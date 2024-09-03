import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

from amd.rocal.plugin.pytorch import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

class NumSamplesAndBatches:
    def __init__(self, num_samples: int = 0, num_batches: int = 0):
        self.num_samples = num_samples
        self.num_batches = num_batches

@dataclass
class DataInfo:
    dataloader: ROCALClassificationIterator
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    num_samples_and_batches: NumSamplesAndBatches = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    print("\n num_shards", num_shards)
    print("\ntotal_size", total_size)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path
    rocal_cpu = True
    num_thread = 1
    valdir = data_path
    crop = 224
    pipe_val = val_pipeline(valdir, args.batch_size, args.device, args.world_size, num_thread, crop, rocal_cpu, wds=False)
    pipe_val.build()
    dataloader = ROCALClassificationIterator(pipe_val, device="cpu" if rocal_cpu else "cuda", device_id = args.device)
    return DataInfo(dataloader=dataloader, sampler=None)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def train_pipeline(data_path, batch_size, local_rank, world_size, num_thread, crop, rocal_cpu):
    print("\n DATASET PATH OF TRAIN PIPELINE", data_path)
    # print("type pf local rank",type(int(local_rank)))
    # print(int(local_rank))
    pipe = Pipeline(batch_size=batch_size, num_threads=8, device_id=torch.distributed.get_rank(), seed=torch.distributed.get_rank()+10, rocal_cpu=rocal_cpu, tensor_dtype = types.FLOAT, tensor_layout=types.NCHW, prefetch_queue_depth = 6, mean = [0.485 * 255,0.456 * 255,0.406 * 255], std = [0.229 * 255,0.224 * 255,0.225 * 255], output_memory_type = types.HOST_MEMORY if rocal_cpu else types.DEVICE_MEMORY)
    with pipe:
        img_raw = fn.readers.webdataset(
        path=data_path, ext=[{'jpg', 'json', 'txt'}], missing_components_behavior = types.SKIP)
        decode = fn.decoders.webdataset(img_raw, file_root=data_path, color_format=types.RGB,max_decoded_width=1510, max_decoded_height=1024, shard_id=torch.distributed.get_rank(), num_shards=world_size, random_shuffle=True)
        rocal_device = 'cpu' if rocal_cpu else 'gpu'
        crop_aspect_ratio = fn.random.uniform(img_raw, range=[0.75, 1.3333])
        crop_area_factor = fn.random.uniform(img_raw, range=[0.9, 1])
        # randomcrop = fn.random_crop(decode, crop_area_factor=[0.9, 1], crop_aspect_ratio=[0.75, 1.3333],
        #         crop_pox_x=0, crop_pox_y=0, num_attempts=20, device=None,
        #         all_boxes_above_threshold=True, allow_no_crop=True, ltrb=True, output_layout=types.NHWC, output_dtype=types.UINT8)
        random_resize_crop = fn.resize_crop(decode,
                        resize_width=224,
                        resize_height=224,
                        crop_area_factor=crop_area_factor,
                        crop_aspect_ratio=crop_aspect_ratio,
                        # x_drift=0.6,
                        # y_drift=0.4,
                        interpolation_type=types.CUBIC_INTERPOLATION,
                        output_layout=types.NHWC, output_dtype=types.UINT8)
        # res = fn.resize(randomcrop, resize_width=224, resize_height=224, output_layout = types.NHWC, output_dtype = types.UINT8, interpolation_type=types.CUBIC_INTERPOLATION)
        # flip_coin = fn.random.coin_flip(probability=0.5)
        cmnp = fn.crop_mirror_normalize(random_resize_crop,
                                        output_layout = types.NCHW,
                                        output_dtype = types.FLOAT,
                                        crop=(crop, crop),
                                        mirror=0,
                                        mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
                                        std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255])
        pipe.set_outputs(cmnp)
    print('rocal "{0}" variant'.format(rocal_device))
    return pipe

def val_pipeline(data_path, batch_size, local_rank, world_size, num_thread, crop, rocal_cpu, wds):
    print("\n DATASET PATH OF VAL PIPELINE", data_path)
    if wds:
        pipe = Pipeline(batch_size=batch_size, num_threads=8, device_id=0, seed=torch.distributed.get_rank() + 10, rocal_cpu=rocal_cpu, tensor_dtype = types.FLOAT, tensor_layout=types.NCHW, prefetch_queue_depth = 6, mean = [0.485 * 255,0.456 * 255,0.406 * 255], std = [0.229 * 255,0.224 * 255,0.225 * 255], output_memory_type = types.HOST_MEMORY if rocal_cpu else types.DEVICE_MEMORY)
    else:
        pipe = Pipeline(batch_size=batch_size, num_threads=8, device_id=0, seed=torch.distributed.get_rank() + 10, rocal_cpu=rocal_cpu, tensor_dtype = types.FLOAT, tensor_layout=types.NCHW, prefetch_queue_depth = 6, mean = [0.485 * 255,0.456 * 255,0.406 * 255], std = [0.229 * 255,0.224 * 255,0.225 * 255], output_memory_type = types.HOST_MEMORY if rocal_cpu else types.DEVICE_MEMORY)

    with pipe:
        rocal_device = 'cpu' if rocal_cpu else 'gpu'
        if wds:
            img_raw = fn.readers.webdataset(
            path=data_path, ext=[{'jpg', 'txt'}], missing_components_behavior = types.SKIP)
            decode = fn.decoders.webdataset(img_raw, last_batch_policy=types.LAST_BATCH_PARTIAL, file_root=data_path, color_format=types.RGB,max_decoded_width=512, max_decoded_height=512, shard_id=0, num_shards=1)
        else:
            jpegs, labels = fn.readers.file(file_root=data_path)
            decode = fn.decoders.image(jpegs,file_root=data_path, max_decoded_width=1000, max_decoded_height=1000, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False, last_batch_policy=types.LAST_BATCH_PARTIAL)
        res = fn.resize(decode, resize_shorter = 224, scaling_mode=types.SCALING_MODE_NOT_SMALLER, interpolation_type=types.CUBIC_INTERPOLATION, output_layout = types.NHWC, output_dtype = types.UINT8)
        centercrop = fn.center_crop(res, crop=[224, 224], output_layout = types.NHWC, output_dtype = types.UINT8)
        cmnp = fn.crop_mirror_normalize(centercrop,
                                        output_layout = types.NCHW,
                                        output_dtype = types.FLOAT,
                                        crop=(224, 224),
                                        mirror=0,
                                        mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
                                        std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255])
        pipe.set_outputs(cmnp)
    print('rocal "{0}" variant'.format(rocal_device))
    return pipe

def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    print("Inside get_wds_dataset")
    input_shards = args.train_data if is_train else args.val_data
    print("input shards", input_shards)
    # exit(0)

    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    print("\n resampled", resampled)
    # exit(0)
    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    rocal_cpu = True
    num_thread = 1
    if is_train:
        traindir = os.path.dirname(input_shards) + "/"
        crop = 224
        pipe_train = train_pipeline(traindir, args.batch_size, args.device, args.world_size, num_thread, crop, rocal_cpu)
        pipe_train.build()
        dataloader = ROCALClassificationIterator(pipe_train, device="cpu" if rocal_cpu else "cuda", device_id = args.device)
    else:
        valdir = os.path.dirname(input_shards) + "/"
        crop = 224
        pipe_val = val_pipeline(valdir, args.batch_size, args.device, args.world_size, num_thread, crop, rocal_cpu, wds=True)
        pipe_val.build()
        dataloader = ROCALClassificationIterator(pipe_val, device="cpu" if rocal_cpu else "cuda", device_id = args.device)
    # dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            print("\n num_shards", num_shards)
            print("\n args.workers", args.workers)
            print("\n args.world_size", args.world_size)
            # assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        # dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)


    print("dataloader", dataloader)
    
    # dataloader.num_batches = num_batches
    # # print("\n num_batches", num_batches)
    # dataloader.num_samples = num_samples
    # print("\n num_samples", num_samples)
    # for data in dataloader:
    #     print(len(data))
    # exit(0)
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, num_samples_and_batches=NumSamplesAndBatches(num_samples=num_samples, num_batches=num_batches))


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    print("Calls the dataset fn")
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    # print("preprocess_train", preprocess_train)
    # print("preprocess_val", preprocess_val)
    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
