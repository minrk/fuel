from __future__ import division
from contextlib import closing
from collections import defaultdict
import gzip
import io
import itertools
import multiprocessing
import os
import logging
import os.path
import tarfile

import h5py
import numpy
from PIL import Image
from scipy.io.matlab import loadmat
import six
from six.moves import zip, xrange
from toolz.itertoolz import partition_all
import zmq

# from fuel.datasets import H5PYDataset
from fuel.server import send_arrays, recv_arrays
from fuel.utils.logging import (SubprocessFailure, zmq_log_and_monitor,
                                configure_zmq_process_logger)
log = logging.getLogger(__name__)

DEVKIT_ARCHIVE = 'ILSVRC2010_devkit-1.0.tar.gz'
DEVKIT_META_PATH = 'devkit-1.0/data/meta.mat'
DEVKIT_VALID_GROUNDTRUTH_PATH = ('devkit-1.0/data/'
                                 'ILSVRC2010_validation_ground_truth.txt')
PATCH_IMAGES_TAR = 'patch_images.tar'
TEST_GROUNDTRUTH = 'ILSVRC2010_test_ground_truth.txt'
TRAIN_IMAGES_TAR = 'ILSVRC2010_images_train.tar'
VALID_IMAGES_TAR = 'ILSVRC2010_images_val.tar'
TEST_IMAGES_TAR = 'ILSVRC2010_images_test.tar'
IMAGE_TARS = TRAIN_IMAGES_TAR, VALID_IMAGES_TAR, TEST_IMAGES_TAR


def make_debug_logging_function(logger, process_type, **additional_kwargs):
    def _debug(status, **kwargs):
        pid = os.getpid()
        message_str = "{process_type}({pid}): {status} ".format(
            process_type=process_type, pid=pid, status=status)
        message_str += " ".join("{key}={val}".format(key=key, val=val)
                                for key, val in kwargs.items())
        kwargs['process_type'] = process_type
        kwargs['status'] = status
        kwargs.update(additional_kwargs)
        logger.debug(message_str, extra=kwargs)
    return _debug


def ilsvrc2010(directory, save_path, image_dim=256, shuffle_train_set=True,
               shuffle_seed=(2015, 4, 1), num_workers=4, chunk_size=128):
    """Converter for the ILSVRC2010 dataset.

    Parameters
    ----------
    directory : str
        Path from which to read raw data files.
    save_path : str
        Path to which to save the HDF5 file.
    image_dim : int, optional
        The number of rows and columns to which images are normalized
        (default 256).
    shuffle_train_set : bool, optional
        If `True` (default), shuffle the training set within the HDF5 file,
        so that a sequential read through the training set is shuffled
        by default.
    shuffle_seed : int or sequence, optional
        Seed for a `numpy.random.RandomState` used to shuffle the training
        set order.
    num_workers : int, optional
        The number of worker processes to deploy.
    chunk_size : int, optional
        The number of images the workers should send to the sink at a
        time.

    """
    debug = make_debug_logging_function(log, 'MAIN')

    # Read what's necessary from the development kit.
    devkit_path = os.path.join(directory, DEVKIT_ARCHIVE)
    synsets, cost_matrix, raw_valid_groundtruth = read_devkit(devkit_path)

    # Mapping to take WordNet IDs to our internal 0-999 encoding.
    wnid_map = dict(zip((s.decode('utf8') for s in synsets['WNID']),
                        xrange(1000)))

    # Mapping to take ILSVRC2010 (integer) IDs to our internal 0-999 encoding.
    # label_map = dict(zip(synsets['ILSVRC2010_ID'], xrange(1000)))
    train, valid, test, patch = [os.path.join(directory, fn)
                                 for fn in IMAGE_TARS + (PATCH_IMAGES_TAR,)]

    # Raw test data groundtruth, ILSVRC2010 IDs.
    raw_test_groundtruth = numpy.loadtxt(
        os.path.join(directory, TEST_GROUNDTRUTH),
        dtype=numpy.int16)

    # Ascertain the number of filenames to prepare appropriate sized
    # arrays.
    n_train = int(synsets['num_train_images'].sum())
    n_valid, n_test = len(raw_valid_groundtruth), len(raw_test_groundtruth)
    n_total = n_train + n_valid + n_test
    log.info("Training set: {} images".format(n_train))
    log.info("Validation set: {} images".format(n_valid))
    log.info("Test set: {} images".format(n_test))
    log.info("Total (train/valid/test): {} images".format(n_total))
    width = height = image_dim
    channels = 3
    with h5py.File(os.path.join(save_path, 'ilsvrc2010.hdf5'), 'w') as f:
        log.info("Creating HDF5 datasets...")
        f.create_dataset('features', shape=(n_total, channels,
                                            height, width),
                         dtype='uint8')
        f.create_dataset('targets', shape=(n_total,), dtype='int16')
        debug(status='STARTED_SET', which_set='train', num_images='n_train')
        process_train_set(f, train, patch, synsets['num_train_images'],
                          wnid_map, n_train, image_dim, num_workers,
                          chunk_size)
        debug(status='FINISHED_SET', which_set='train', num_images='n_train')
        ilsvrc_id_to_zero_based = dict(zip(synsets['ILSVRC2010_ID'],
                                       xrange(len(synsets))))
        valid_groundtruth = [ilsvrc_id_to_zero_based[id_]
                             for id_ in raw_valid_groundtruth]
        debug(status='STARTED_SET', which_set='valid', num_images='n_valid')
        for num_completed in process_other_set(f['features'], f['targets'],
                                               valid, patch, valid_groundtruth,
                                               'valid', chunk_size, image_dim,
                                               n_train):
            debug(status='WRITTEN', which_set='valid',
                  images_written=num_completed)
        debug(status='FINISHED_SET', which_set='valid')
        test_groundtruth = [ilsvrc_id_to_zero_based[id_]
                            for id_ in raw_test_groundtruth]
        debug(status='STARTED_SET', which_set='test', num_images='n_test')
        for num_completed in process_other_set(f['features'], f['targets'],
                                               test, patch, test_groundtruth,
                                               'test', chunk_size, image_dim,
                                               n_train + n_valid):
            debug(status='WRITTEN', which_set='test',
                  images_written=num_completed)
        debug(status='FINISHED_SET', which_set='test')


def process_train_set(hdf5_file, train_archive_path, patch_archive_path,
                      train_images_per_class, wnid_map, n_train, image_dim,
                      num_workers, worker_chunk_size):
    ventilator = multiprocessing.Process(target=train_set_ventilator,
                                         args=(train_archive_path,))
    ventilator.start()
    workers = [multiprocessing.Process(target=train_set_worker,
                                       args=(train_archive_path,
                                             patch_archive_path, wnid_map,
                                             train_images_per_class,
                                             image_dim, worker_chunk_size))
               for _ in xrange(num_workers)]
    for worker in workers:
        worker.start()
    sink = multiprocessing.Process(target=train_set_sink,
                                   args=(hdf5_file, n_train,
                                         train_images_per_class))
    sink.start()
    try:
        context = zmq.Context()
        zmq_log_and_monitor(log, context,
                            processes=[ventilator, sink] + workers,
                            failure_threshold=logging.ERROR)
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received.")
    except SubprocessFailure:
        log.info("One or more substituent processes failed.")
    finally:
        log.info("Shutting down workers and ventilator...")
        for worker in workers:
            worker.terminate()
        ventilator.terminate()
        sink.terminate()
        log.info("Killed child processes.")


def process_other_set(features, targets, archive, patch_archive, groundtruth,
                      which_set, chunk_size, image_dim, offset):
    """Process and convert either the validation set or the test set.

    Parameters
    ----------
    features : array-like
        The array (or array-like container, such as an HDF5 dataset)
        to which images should be written.
    targets : array-like
        The array (or array-like container, such as an HDF5 dataset)
        to which targets should be written.
    archive : str or file-like object
    patch_archive : str or file-like object
    groundtruth : ndarray, 1-dimensional
        Integer targets, with the same length as the number of images
    which_set : str
        One of 'valid' or 'test', used to extract the right patch images.
    chunk_size : int
        The number of examples/targets to write at a time.
    offset : int
        The offset in the `features` and `targets` arrays at which to
        begin writing rows.

    Yields
    ------
    int
        A stream of integers. Each represents the number of examples
        processed so far, in increments of `chunk_size`.
    """
    patch_images = extract_patch_images(patch_archive, which_set)
    with _open_tar_file(archive) as tar:
        images_gen = (_cropped_transposed_patched(tar, filename,
                                                  patch_images, image_dim)
                      for filename in sorted(info.name for info in tar))
        start = offset
        for chunk in partition_all(zip(images_gen, groundtruth), chunk_size):
            images, labels = zip(*chunk)
            this_chunk = len(images)
            features[start:start + this_chunk] = numpy.concatenate(images)
            targets[start:start + this_chunk] = labels
            start += this_chunk
            yield start


def train_set_ventilator(f, ventilator_port=5557, sink_port=5558,
                         logging_port=5559, high_water_mark=10):
    context = zmq.Context()
    configure_zmq_process_logger(log, context, logging_port)
    debug = make_debug_logging_function(log, 'VENTILATOR')
    debug(status='START')
    sender = context.socket(zmq.PUSH)
    sender.hwm = high_water_mark
    sender.bind("tcp://*:{}".format(ventilator_port))
    sink = context.socket(zmq.PUSH)
    sink.connect("tcp://localhost:{}".format(sink_port))
    # Signal the sink to start receiving. Required, according to ZMQ guide.
    sink.send(b'0')
    with _open_tar_file(f) as tar:
        for num, inner_tar in enumerate(tar):
            with closing(tar.extractfile(inner_tar.name)) as f:
                sender.send_pyobj((num, inner_tar.name), zmq.SNDMORE)
                sender.send(f.read())
                debug(status='SENT', tar_filename=inner_tar.name, number=num)
        debug(status='SHUTDOWN')


def train_set_worker(f, patch_images_path, wnid_map, images_per_class,
                     image_dim=256, chunk_size=128, ventilator_port=5557,
                     sink_port=5558, logging_port=5559,
                     receiver_high_water_mark=10, sender_high_water_mark=10):
    context = zmq.Context()

    # Set up logging.
    configure_zmq_process_logger(log, context, logging_port)
    debug = make_debug_logging_function(log, 'WORKER')

    # Set up ventilator->worker socket on the worker end.
    receiver = context.socket(zmq.PULL)
    receiver.hwm = receiver_high_water_mark
    receiver.connect("tcp://localhost:{}".format(ventilator_port))
    debug(status='CONNECTED_VENTILATOR', port=ventilator_port)

    # Set up worker->sink socket on the worker end.
    sender = context.socket(zmq.PUSH)
    sender.hwm = sender_high_water_mark
    sender.connect("tcp://localhost:{}".format(sink_port))
    debug(status='CONNECTED_SINK', port=sink_port)

    # Grab patch images off disk.
    patch_images = extract_patch_images(patch_images_path, 'train')

    while True:
        # Receive a class TAR file.
        num, name = receiver.recv_pyobj()
        label = wnid_map[name.split('.')[0]]
        tar_data = io.BytesIO(receiver.recv())
        debug(status='RECEIVED', tar_filename=name, number=num, label_id=label)

        with tarfile.open(fileobj=tar_data) as tar:
            debug(status='OPENED', tar_filename=name)
            images_gen = (_cropped_transposed_patched(tar, jpeg_info.name,
                                                      patch_images, image_dim)
                          for jpeg_info in tar)
            total_images = 0
            # Send images to sink in batches of at most chunk_size.
            try:
                for images in partition_all(chunk_size, images_gen):
                    sender.send_pyobj(label, zmq.SNDMORE)
                    send_arrays(sender, [numpy.concatenate(images)])
                    total_images += len(images)
                    debug(status='SENT', tar_filename=name, number=num,
                          num_images=len(images), total_so_far=total_images)
            except Exception:
                log.error("WORKER(%d): Encountered error processing "
                          "%s (%d images processed successfully)",
                          os.getpid(), name, total_images,
                          exc_info=1)

        if total_images != images_per_class[label]:
            log.error("WORKER(%d): For class %s (%d), expected %d images but "
                      "only found %d", os.getpid(), name.split('.')[0], label,
                      images_per_class[label], total_images, exc_info=1)
        debug(status='FINISHED_CLASS', tar_filename=name, number=num,
              total=total_images)


def train_set_sink(hdf5_file, num_images, images_per_class,
                   flush_frequency=256, shuffle_seed=(2015, 4, 9),
                   sink_port=5558, logging_port=5559, high_water_mark=10):
    context = zmq.Context()

    # Set up logging.
    configure_zmq_process_logger(log, context, logging_port)
    debug = make_debug_logging_function(log, 'SINK')

    # Create a shuffling order and parcel it up into a list of iterators
    # over class-sized sublists of the list.
    all_order = numpy.random.RandomState(shuffle_seed).permutation(num_images)
    orders = list(map(iter, class_permutations(all_order, images_per_class)))

    # Receive completed batches from the workers.
    receiver = context.socket(zmq.PULL)
    receiver.hwm = 10
    receiver.bind("tcp://*:5558")

    # Synchronize with the ventilator.
    receiver.recv()
    batches_received = 0
    images_sum = None
    images_sq_sum = None
    num_images_remaining = num_images
    num_images_by_label = defaultdict(lambda: 0)

    features = hdf5_file['features']
    targets = hdf5_file['targets']
    try:
        while num_images_remaining > 0:
            # Receive a label and a batch of images.
            label = receiver.recv_pyobj()
            images, = recv_arrays(receiver)
            batches_received += 1

            debug(status='RECEIVED', label=label,
                  num_images=images.shape[0], batch_number=batches_received,
                  num_images_remaining=num_images_remaining)

            # Delay creation of the sum arrays until we've got the first
            # batch so that we can size them correctly.
            if images_sum is None:
                images_sum = numpy.zeros_like(images[0], dtype=numpy.float64)
                images_sq_sum = numpy.zeros_like(images_sum)

            # Grab the next few indices for this label. We partition the
            # indices by class so that no matter which order we receive
            # batches in, the final order is deterministic (because the
            # images within a class always appear in a deterministic order,
            # i.e. the order they are read out of the TAR file).
            indices = sorted(itertools.islice(orders[label], images.shape[0]))
            features[indices] = images
            targets[indices] = label * numpy.ones(images.shape[0],
                                                  dtype=numpy.int16)

            num_images_remaining -= images.shape[0]
            num_images_by_label[label] += images.shape[0]

            debug(status='WRITTEN', label=label,
                  num_images=images.shape[0], batch_number=batches_received,
                  num_images_remaining=num_images_remaining)

            # Accumulate the sum and the sum of the square, for mean and
            # variance statistics.
            images_sum += images.sum(axis=0)
            images_sq_sum += (images.astype(numpy.uint64) ** 2).sum(axis=0)

            # Manually flush file to disk at regular intervals. Unsure whether
            # this is strictly necessary.
            if batches_received % flush_frequency == 0:
                debug(status='FLUSH', hdf5_filename=hdf5_file.filename)
                hdf5_file.flush()
    except Exception:
        log.error("SINK(%d): encountered exception (%d images remaining)",
                  os.getpid(), num_images_remaining, exc_info=1)
        return

    # Compute training set mean and variance.
    features.attrs['training_set_mean'] = mean = images_sum / num_images
    sq_mean = images_sq_sum / num_images
    features.attrs['training_set_std'] = numpy.sqrt(sq_mean - mean**2)
    debug(status='DONE')


def _open_tar_file(f):
    """Open either a filename or a file-like object as a TAR file.

    Parameters
    ----------
    f : str or file-like object
        The filename or file-like object from which to read.

    Returns
    -------
    TarFile
        A `TarFile` instance.

    """
    if isinstance(f, six.string_types):
        return tarfile.open(name=f)
    else:
        return tarfile.open(fileobj=f)


def _imread(f):
    """Simple image reading function.

    Parameters
    ----------
    f : str or file-like object
        Filename or file object from which to read image data.

    Returns
    -------
    image : ndarray, 3-dimensional
        RGB image data as a NumPy array with shape `(rows, cols, 3)`.

    """
    with closing(Image.open(f).convert('RGB')) as f:
        return numpy.array(f)


def _cropped_transposed_patched(tar, jpeg_filename, patch_images, image_dim):
    """Do everything necessary to process a JPEG inside a TAR.

    Parameters
    ----------
    tar : `TarFile` instance
        The tar from which to read `jpeg_filename`.
    jpeg_filename : str
        Fully-qualified path inside of `tar` from which to read a
        JPEG file.
    patch_images : dict
        A dictionary containing filenames (without path) of replacements
        to be substituted in place of the version of the same file found
        in `tar`. Values are in `(width, height, channels)` layout.
    image_dim : int
        The length to which the shorter side of the image should be
        scaled. After cropping the central square, the resulting
        image will be `image_dim` x `image_dim`.
    Returns
    -------
    ndarray
        An ndarray of shape `(1, 3, image_dim, image_dim)` containing
        an image.
    """
    # TODO: make the square_crop configurable from calling functions.
    image = patch_images.get(os.path.basename(jpeg_filename), None)
    if image is None:
        try:
            image = _imread(tar.extractfile(jpeg_filename))
        except (IOError, OSError):
            with gzip.GzipFile(fileobj=tar.extractfile(jpeg_filename)) as gz:
                image = _imread(gz)

    return square_crop(image, dim=image_dim).transpose(2, 0, 1)[numpy.newaxis]


def other_images_generator(f, patch_images, labels, label_map):
    """Generate a stream of images from the valid or test set TAR.


    Parameters
    ----------
    f : str or file-like object
        The filename or file-handle to the images TAR file.
    patch_images : dict
        A dictionary containing filenames (without path) of replacements
        to be substituted in place of the version of the same file found
        in `f`.
    labels : sequence
        A list of integer labels.
    label_map : dict
        Dictionary mapping the integers in `labels` to other integers
        (i.e. from 0-999).

    Yields
    ------
    tuple
        The first element is which is an image represented as an ndarray
        of shape `(1, 256, 256, 3)`, the second element of which is an
        integer label between 0 and 999 inclusive.

    """
    with _open_tar_file(f) as tar:
        filenames = sorted(info.name for info in f
                           if info.name.endswith('.JPEG'))
        for fn, label in zip(filenames, labels):
            image = _cropped_transposed_patched(tar, fn, patch_images)
            yield image, label_map[label]


def class_permutations(order, images_per_class):
    """
    Parameters
    ----------
    order : sequence
        A sequence containing a permutation of the integers from
        0 to `len(order) - 1`.
    images_per_class : sequence
        A sequence containing the number of images in each class,
        with as many elements as there are classes.

    Returns
    -------
    list of lists
        A list the same length as `images_per_class`, except
    """
    if len(order) != sum(images_per_class):
        raise ValueError("images_per_class should sum to the length of order")
    result = []
    for num in images_per_class:
        result, order = result + [order[:num]], order[num:]
    return result


def square_crop(image, dim=256):
    """Crop an image to the central square after resizing it.

    Parameters
    ----------
    image : ndarray, 3-dimensional
        An image represented as a 3D ndarray, with 3 color
        channels represented as the third axis.
    dim : int, optional
        The length of the shorter side after resizing, and the
        length of both sides after cropping. Default is 256.

    Returns
    -------
    cropped : ndarray, 3-dimensional, shape `(dim, dim, 3)`
        The image resized such that the shorter side is length
        `dim`, with the longer side cropped to the central
        `dim` pixels.

    Notes
    -----
    This reproduces the preprocessing technique employed in [Kriz]_.

    .. [Kriz] A. Krizhevsky, I. Sutskever and G.E. Hinton (2012).
       "ImageNet Classification with Deep Convolutional Neural Networks."
       *Advances in Neural Information Processing Systems 25* (NIPS 2012).

    """
    if image.ndim != 3 and image.shape[2] != 3:
        raise ValueError("expected a 3-dimensional ndarray with last axis 3")
    if image.shape[0] > image.shape[1]:
        new_size = int(round(image.shape[0] / image.shape[1] * dim)), dim
        pad = (new_size[0] - dim) // 2
        slices = (slice(pad, pad + dim), slice(None))
    else:
        new_size = dim, int(round(image.shape[1] / image.shape[0] * dim))
        pad = (new_size[1] - dim) // 2
        slices = (slice(None), slice(pad, pad + dim))
    with closing(Image.fromarray(image, mode='RGB')) as pil_image:
        # PIL uses width x height, e.g. cols x rows, hence new_size backwards.
        resized = numpy.array(pil_image.resize(new_size[::-1], Image.BICUBIC))
    out = resized[slices]
    return out


def read_devkit(f):
    """Read relevant information from the development kit archive.

    Parameters
    ----------
    f : str or file-like object
        The filename or file-handle for the gzipped TAR archive
        containing the ILSVRC2010 development kit.

    Returns
    -------
    synsets : ndarray, 1-dimensional, compound dtype
        See :func:`read_metadata` for details.
    cost_matrix : ndarray, 2-dimensional, uint8
        See :func:`read_metadata` for details.
    raw_valid_groundtruth : ndarray, 1-dimensional, int16
        The labels for the ILSVRC2010 validation set,
        distributed with the development kit code.

    """
    with _open_tar_file(f) as tar:
        # Metadata table containing class hierarchy, textual descriptions, etc.
        meta_mat = tar.extractfile(DEVKIT_META_PATH)
        synsets, cost_matrix = read_metadata(meta_mat)

        # Raw validation data groundtruth, ILSVRC2010 IDs. Confusingly
        # distributed inside the development kit archive.
        raw_valid_groundtruth = numpy.loadtxt(tar.extractfile(
            DEVKIT_VALID_GROUNDTRUTH_PATH), dtype=numpy.int16)
    return synsets, cost_matrix, raw_valid_groundtruth


def read_metadata(meta_mat):
    """Read ILSVRC2010 metadata.

    Parameters
    ----------
    meta_mat : str or file-like object
        The filename or file-handle for `meta.mat` from the
        ILSVRC2010 development kit.

    Returns
    -------
    synsets : ndarray, 1-dimensional, compound dtype
        A table containing ILSVRC2010 metadata for the "synonym sets"
        or "synsets" that comprise the classes and superclasses,
        including the following fields:
         * `ILSVRC2010_ID`: the integer ID used in the original
           competition data.
         * `WNID`: A string identifier that uniquely identifies
           a synset in ImageNet and WordNet.
         * `wordnet_height`: The length of the longest path to
           a leaf nodein the FULL ImageNet/WordNet hierarchy
           (leaf nodes in the FULL ImageNet/WordNet hierarchy
           have `wordnet_height` 0).
         * `gloss`: A string representation of an English
           textual description of the concept represented by
           this synset.
         * `num_children`: The number of children in the hierarchy
           for this synset.
         * `words`: A string representation, comma separated,
           of different synoym words or phrases for the concept
           represented by this synset.
         * `children`: A vector of `ILSVRC2010_ID`s of children
           of this synset, padded with -1. Note that these refer
           to `ILSVRC2010_ID`s from the original data and *not*
           the zero-based index in the table.
         * `num_train_images`: The number of training images for
           this synset.
    cost_matrix : ndarray, 2-dimensional, uint8
        A 1000x1000 matrix containing the precomputed pairwise
        cost (based on distance in the hierarchy) for all
        low-level synsets (i.e. the thousand possible output
        classes with training data associated).

    """
    mat = loadmat(meta_mat, squeeze_me=True)
    synsets = mat['synsets']
    cost_matrix = mat['cost_matrix']
    new_dtype = numpy.dtype([
        ('ILSVRC2010_ID', numpy.int16),
        ('WNID', ('S', max(map(len, synsets['WNID'])))),
        ('wordnet_height', numpy.int8),
        ('gloss', ('S', max(map(len, synsets['gloss'])))),
        ('num_children', numpy.int8),
        ('words', ('S', max(map(len, synsets['words'])))),
        ('children', (numpy.int8, max(synsets['num_children']))),
        ('num_train_images', numpy.uint16)
    ])
    new_synsets = numpy.empty(synsets.shape, dtype=new_dtype)
    for attr in ['ILSVRC2010_ID', 'WNID', 'wordnet_height', 'gloss',
                 'num_children', 'words', 'num_train_images']:
        new_synsets[attr] = synsets[attr]
    children = [numpy.atleast_1d(ch) for ch in synsets['children']]
    padded_children = [
        numpy.concatenate((c,
                           -numpy.ones(new_dtype['children'].shape[0] - len(c),
                                       dtype=numpy.int16)))
        for c in children
    ]
    new_synsets['children'] = padded_children
    return new_synsets, cost_matrix


def extract_patch_images(f, which_set=None):
    """Extracts a dict of the "patch images" for ILSVRC2010.

    Parameters
    ----------
    f : str or file-like object
        The filename or file-handle to the patch images TAR file.
    which_set : str
        Which set of images to extract. One of 'train', 'valid', 'test'.

    Returns
    -------
    dict
        A dictionary contains a mapping of filenames (without path) to a
        NumPy array containing the replacement image.

    Notes
    -----
    Certain images in the distributed archives are blank, or display
    an "image not available" banner. A separate TAR file of
    "patch images" is distributed with the corrected versions of
    these. It is this archive that this function is intended to read.

    """
    if which_set not in ('train', 'valid', 'test'):
        raise ValueError("which_set must be one of train, valid, or test")
    which_set = 'val' if which_set == 'valid' else which_set
    patch_images = {}
    with _open_tar_file(f) as tar:
        for info_obj in tar:
            if not info_obj.name.endswith('.JPEG'):
                continue
            # Pretty sure that '/' is used for tarfile regardless of
            # os.path.sep, but I officially don't care about Windows.
            tokens = info_obj.name.split('/')
            file_which_set = tokens[1]
            if file_which_set != which_set:
                continue
            filename = tokens[-1]
            image = _imread(tar.extractfile(info_obj.name))
            patch_images[filename] = image
    return patch_images


if __name__ == "__main__":
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh = logging.FileHandler('log.txt')
    log.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.addHandler(fh)
    while log.root.handlers:
        log.root.handlers.pop()
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    ilsvrc2010('.', '.')
