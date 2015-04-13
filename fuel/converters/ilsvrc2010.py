from __future__ import division

from contextlib import closing
from collections import defaultdict
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

from fuel.server import send_arrays, recv_arrays


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


def ilsvrc2010(directory, save_path, image_dim=256, shuffle_train_set=True,
               shuffle_seed=(2015, 4, 1), num_workers=7):
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

    """
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
    # raw_test_groundtruth = numpy.loadtxt(
    #     os.path.join(directory, TEST_GROUNDTRUTH),
    #     dtype=numpy.int16)

    # Ascertain the number of filenames to prepare appropriate sized
    # arrays.
    # train_files = extract_train_filenames(train)
    print(synsets['num_train_images'].dtype)
    n_train = int(synsets['num_train_images'].sum())
    # with _open_tar_file(valid) as valid_f, _open_tar_file(test) as test_f:
    #     valid_files, test_files = [[sorted(info.name for info in f
    #                                        if info.name.endswith('.JPEG'))]
    #                                for f in (valid_f, test_f)]
    # n_valid, n_test = [len(fn) for fn in
    #                    (valid_files, test_files)]
    n_valid, n_test = 0, 0
    n_total = n_train + n_valid + n_test
    log.info("Training set: {} images".format(n_train))
    log.info("Validation set: {} images".format(n_valid))
    log.info("Test set: {} images".format(n_test))
    log.info("Total (train/valid/test): {} images".format(n_total))
    width = height = image_dim
    channels = 3
    ventilator = multiprocessing.Process(target=train_set_ventilator,
                                         args=(train,))
    ventilator.start()
    workers = [multiprocessing.Process(target=train_set_worker,
                                       args=(train, patch, wnid_map,
                                             synsets['num_train_images']))
               for _ in xrange(num_workers)]
    for worker in workers:
        worker.start()
    with h5py.File(os.path.join(save_path, 'ilsvrc2010.hdf5'), 'w') as f:
        log.info("Creating HDF5 datasets...")
        f.create_dataset('features', shape=(n_total, channels,
                                            height, width),
                         dtype='uint8')
        f.create_dataset('targets', shape=(n_total,), dtype='int16')
        sink = multiprocessing.Process(target=train_set_sink,
                                       args=(f, n_total,
                                             synsets['num_train_images']))
        sink.start()
        try:
            log_messages()
        except KeyboardInterrupt:
            log.info("Shutting down workers and ventilator...")
        finally:
            for worker in workers:
                worker.terminate()
            ventilator.terminate()
            sink.terminate()
            log.info("Killed child processes.")


def log_messages(logging_port=5559):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:{}".format(logging_port))
    while True:
        message = receiver.recv_json()
        type = message.pop('type')
        pid = message.pop('pid')
        status = message.pop('status')
        log.debug('%10s(%d): %10s %s' % (type, pid, status, str(message)))


def train_set_ventilator(f, ventilator_port=5557, sink_port=5558,
                         logging_port=5559):
    pid = os.getpid()
    context = zmq.Context()
    log_socket = context.socket(zmq.PUSH)
    log_socket.connect("tcp://localhost:{}".format(logging_port))

    def log(**kwargs):
        message = {'type': 'VENTILATOR', 'pid': pid}
        message.update(kwargs)
        log_socket.send_json(message)

    log(status='START')
    sender = context.socket(zmq.PUSH)
    sender.hwm = 1  # TODO: make this configurable
    sender.bind("tcp://*:{}".format(ventilator_port))
    sink = context.socket(zmq.PUSH)
    sink.connect("tcp://localhost:{}".format(sink_port))
    # The first message is "0" and signals start of batch (TODO wat)
    sink.send(b'0')
    with _open_tar_file(f) as tar:
        for num, inner_tar in enumerate(tar):
            with closing(tar.extractfile(inner_tar.name)) as f:
                sender.send_pyobj((num, inner_tar.name), zmq.SNDMORE)
                sender.send(f.read())
                log(status='SENT', filename=inner_tar.name, number=num)
        log(status='FINISHED')


def train_set_worker(f, patch_images_path, wnid_map, images_per_class,
                     chunk_size=128, ventilator_port=5557, sink_port=5558,
                     logging_port=5559):
    pid = os.getpid()
    context = zmq.Context()

    log_socket = context.socket(zmq.PUSH)
    log_socket.connect("tcp://localhost:{}".format(logging_port))

    def log(**kwargs):
        message = {'type': 'WORKER', 'pid': pid}
        message.update(kwargs)
        log_socket.send_json(message)

    sender = context.socket(zmq.PUSH)
    patch_images = extract_patch_images(patch_images_path, 'train')
    receiver = context.socket(zmq.PULL)
    receiver.hwm = 1
    receiver.connect("tcp://localhost:{}".format(ventilator_port))
    log(status='CONNECTED_VENTILATOR', port=ventilator_port)
    sender = context.socket(zmq.PUSH)
    sender.hwm = 1
    sender.connect("tcp://localhost:{}".format(sink_port))
    log(status='CONNECTED_SINK', port=sink_port)
    while True:
        num, name = receiver.recv_pyobj()
        label = wnid_map[name.split('.')[0]]
        tar_data = io.BytesIO(receiver.recv())
        log(status='RECEIVED', filename=name, number=num, label_id=label)
        with tarfile.open(fileobj=tar_data) as tar:
            log(status='OPENED', filename=name)
            images_gen = (_cropped_transposed_patched(tar, jpeg_info.name,
                                                      patch_images)
                          for jpeg_info in tar)
            total_images = 0
            for images in partition_all(chunk_size, images_gen):
                sender.send_pyobj(label, zmq.SNDMORE)
                send_arrays(sender, [numpy.concatenate(images)])
                total_images += len(images)
                log(status='SENT', filename=name, number=num,
                    num_images=len(images), total_so_far=total_images)
        if total_images != images_per_class[label]:
            log(status='NOT_ENOUGH_IMAGES', filename=name, number=num,
                total_so_far=total_images)
        log(status='FINISHED', filename=name, number=num, total=total_images)


def class_permutations(order, images_per_class):
    if len(order) != sum(images_per_class):
        raise ValueError("images_per_class should sum to the length of order")
    result = []
    order_iter = iter(order)
    print(images_per_class)
    for num_images in images_per_class:
        num_images = int(num_images)
        result.append(list(itertools.islice(order_iter, num_images)))
    return result


def train_set_sink(hdf5_file, num_images, images_per_class,
                   flush_frequency=256, shuffle_seed=(2015, 4, 9),
                   sink_port=5558, logging_port=5559):
    pid = os.getpid()
    context = zmq.Context()
    log_socket = context.socket(zmq.PUSH)
    log_socket.connect("tcp://localhost:5559")

    def log(**kwargs):
        message = {'type': 'SINK', 'pid': pid}
        message.update(kwargs)
        log_socket.send_json(message)

    receiver = context.socket(zmq.PULL)
    receiver.hwm = 1
    receiver.bind("tcp://*:5558")
    order = numpy.random.RandomState(shuffle_seed).permutation(num_images)
    orders = list(map(iter, class_permutations(order, images_per_class)))
    features = hdf5_file['features']
    targets = hdf5_file['targets']
    # Synchronize.
    receiver.recv()
    batches_received = 0
    images_sum = None
    images_sq_sum = None
    num_images_remaining = num_images
    num_images_by_label = defaultdict(lambda: 0)
    while num_images_remaining > 0:
        label = receiver.recv_pyobj()
        images, = recv_arrays(receiver)
        batches_received += 1
        num_images_remaining -= images.shape[0]
        num_images_by_label[label] += images.shape[0]
        log(status='RECEIVED', label=label,
            num_images=images.shape[0], batch_number=batches_received,
            num_images_remaining=num_images_remaining)
        if images_sum is None:
            images_sum = numpy.zeros_like(images[0], dtype=numpy.float64)
            images_sq_sum = numpy.zeros_like(images[0], dtype=numpy.float64)
        indices = sorted(itertools.islice(orders[label], images.shape[0]))
        features[indices] = images
        targets[indices] = label * numpy.ones(images.shape[0],
                                              dtype=numpy.int16)
        log(status='WRITTEN', label=label,
            num_images=images.shape[0], batch_number=batches_received,
            num_images_remaining=num_images_remaining)
        images_sum += images.sum(axis=0)
        images_sq_sum += (images.astype(numpy.uint64) ** 2).sum(axis=0)
        if batches_received % flush_frequency == 0:
            log(status='FLUSH', hdf5_filename=hdf5_file.filename,
                num_images_by_label=num_images_by_label)
            hdf5_file.flush()
    features.attrs['training_set_mean'] = mean = images_sum / num_images
    sq_mean = images_sq_sum / num_images
    features.attrs['training_set_std'] = numpy.sqrt(sq_mean - mean**2)
    log.status(status='FINISHED', num_images_by_label=num_images_by_label)


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
    with closing(Image.open(f).convert('RGB')) as f:
        return numpy.array(f)


def _cropped_transposed_patched(tar, jpeg_filename, patch_images):
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

    Returns
    -------
    ndarray
        An ndarray of shape `(1, 3, 256, 256)` containing an image.

    """
    # TODO: make the square_crop configurable from calling functions.
    image = patch_images.get(os.path.basename(jpeg_filename), None)
    if image is None:
        image = _imread(tar.extractfile(jpeg_filename))
    return square_crop(image).transpose(2, 0, 1)[numpy.newaxis, ...]


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
        Which set of images to extract. One of 'train', 'val', 'test'.

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
    fh = logging.FileHandler('log.txt')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.addHandler(fh)
    log.setLevel(logging.DEBUG)
    ilsvrc2010('.', '.')
