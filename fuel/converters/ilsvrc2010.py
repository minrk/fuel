from __future__ import division

import itertools
import os.path
import tarfile

import h5py
import numpy
from scipy.misc import imread, imresize
from scipy.io.matlab import loadmat
import six
from six.moves import zip, xrange
from toolz.itertoolz import partition_all


DEVKIT_ARCHIVE = 'devkit-1.0.tar.gz'
DEVKIT_META_PATH = 'data/meta.mat'
DEVKIT_VALID_GROUNDTRUTH_PATH = 'data/ILSVRC2010_validation_ground_truth.txt'
PATCH_IMAGES_TAR = 'patch_images.tar'
TEST_GROUNDTRUTH = 'ILSVRC2010_test_ground_truth.txt'
TRAIN_IMAGES_TAR = 'ILSVRC2010_images_train.tar'
VALID_IMAGES_TAR = 'ILSVRC2010_images_val.tar'
TEST_IMAGES_TAR = 'ILSVRC2010_images_test.tar'
IMAGE_TARS = TRAIN_IMAGES_TAR, VALID_IMAGES_TAR, TEST_IMAGES_TAR


def ilsvrc2010(directory, save_path, shuffle_train_set=True,
               shuffle_seed=(2015, 4, 1)):
    """Converter for the ILSVRC2010 dataset.

    Parameters
    ----------
    directory : str
        Path from which to read raw data files.
    save_path : str
        Path to save
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
    wnid_map = dict(zip(synsets['WNID']), xrange(1000))

    # Mapping to take ILSVRC2010 (integer) IDs to our internal 0-999 encoding.
    label_map = dict(zip(synsets['ILSVRC2010_ID'], xrange(1000)))
    train, valid, test = [os.path.join(directory, fn) for fn in IMAGE_TARS]

    # Raw test data groundtruth, ILSVRC2010 IDs.
    raw_test_groundtruth = numpy.loadtxt(
        os.path.join(directory, TEST_GROUNDTRUTH),
        dtype=numpy.int16)

    # Read in patch_images.
    all_patch_images = extract_patch_images(PATCH_IMAGES_TAR)

    # Ascertain the number of filenames to prepare appropriate sized
    # arrays.
    train_files = extract_train_filenames(train)
    with _open_tar_file(valid) as valid_f, _open_tar_file(test) as test_f:
        valid_files, test_files = [[sorted(info.name for info in f
                                           if info.name.endswith('.JPEG'))]
                                   for f in (valid_f, test_f)]
    n_train, n_valid, n_test = [len(fn) for fn in
                                (train_files, valid_files, test_files)]
    n_total = n_train + n_valid + n_test
    print("n_total:", n_total)
    width = height = 256  # TODO
    channels = 3
    chunk_size = 512
    with h5py.File(os.path.join(save_path, 'ilsvrc2010.hdf5'), 'wb') as f:
        features = f.create_dataset('features', shape=(n_total, channels,
                                                       height, width),
                                    dtype='uint8')
        targets = f.create_dataset('targets', shape=(n_total,), dtype='int16')
        images_iterator = itertools.chain(
            train_images_generator(train, train_files,
                                   all_patch_images['train'], wnid_map),
            other_images_generator(valid, all_patch_images['val'],
                                   raw_valid_groundtruth, label_map),
            other_images_generator(test, all_patch_images['test'],
                                   raw_test_groundtruth, label_map)
        )
        chunk_iterator = partition_all(chunk_size, images_iterator)
        for i, chunk in enumerate(chunk_iterator):
            images, labels = zip(*chunk)
            images = numpy.vstack(images)
            features[i * chunk_size:(i + 1) * chunk_size] = images
            targets[i * chunk_size:(i + 1) * chunk_size] = labels
            print (i + 1) * chunk_size


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
        in `tar`. Values are in `(width, height, channels)` layout
        as returned by `scipy.misc.imread`.

    Returns
    -------
    ndarray
        An ndarray of shape `(1, 3, 256, 256)` containing an image.

    """
    # TODO: make the square_crop configurable from calling functions.
    image = patch_images.get(os.path.basename(jpeg_filename), None)
    if image is None:
        image = imread(tar.extractfile(jpeg_filename))
    transposed = image.transpose(2, 0, 1)[numpy.newaxis, ...]
    return square_crop(transposed)


def extract_train_filenames(f, shuffle_seed=None):
    """Generator that yields a list of files from the training set TAR.

    Parameters
    ----------
    f : str or file-like object
        The filename or file-handle to the training set TAR file.
    shuffle_seed : int or sequence, optional
        A seed to be passed to :class:`numpy.random.RandomState`,
        used to shuffle the order randomly.

    Returns
    -------
    files : list of tuples
        A list of tuples, with each tuple having the form
        `(class_tar_filename, image_filename)`, where `class_tar_filename`
        is the inner TAR archive corresponding to a certain class and
        `image_filename`.

    """
    files = []
    with _open_tar_file(f) as tar:
        for class_info_obj in tar:
            class_fileobj = tar.extract_file(class_info_obj.name)
            with tarfile.TarFile(fileobj=class_fileobj) as class_tar:
                files.extend((class_info_obj.name, jpeg_info.name)
                             for jpeg_info in class_tar)
    if shuffle_seed is not None:
        files = numpy.array(files)
        rng = numpy.random.RandomState(shuffle_seed)
        rng.shuffle(files)
        return files.tolist()
    else:
        return files


def train_images_generator(f, filenames, patch_images, wnid_map):
    """Generate a stream of images from the training set TAR.

    Parameters
    ----------
    f : str or file-like object
        The filename or file-handle to the train images TAR file.
    filenames : list of tuples
        A list of `(inner_tar_filename, jpeg_filename)` tuples as
        returned by :func:`extract_train_filenames`.
    patch_images : dict
        A dictionary containing filenames (without path) of replacements
        to be substituted in place of the version of the same file found
        in `f`.
    wnid_map : dict
        A dictionary mapping WordNet IDs (the pre-suffix part of the
        inner tar filenames) to a numerical index.

    Yields
    ------
    tuple
        A tuple containing an ndarray with shape (1, 3, 256, 256)`,
        representing an image, and an integer class label.

    """
    inner_tar_handles = {}
    max_handles = 50

    with _open_tar_file(f) as f:
        try:
            for inner_tar, jpeg in filenames:
                if inner_tar not in inner_tar_handles:
                    fobj = f.extractfile(inner_tar)
                    inner_tar_handles[inner_tar] = tarfile.open(fileobj=fobj)
                handle = inner_tar_handles[inner_tar]
                image = _cropped_transposed_patched(handle, jpeg, patch_images)
                label = wnid_map[inner_tar[:-4]]
                yield image, label
                # Super-duper crude resource management.
                if len(inner_tar_handles) > max_handles:
                    inner_tar_handles.pop(next(iter(inner_tar_handles)))
        finally:
            for t in inner_tar_handles.values():
                t.close()


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


def square_crop(image, dim=256, interp='bicubic'):
    """Crop an image to the central square after resizing it.

    Parameters
    ----------
    image : ndarray, 3-dimensional
        An image represented as a 3D ndarray, with 3 color
        channels represented as the third axis.
    dim : int, optional
        The length of the shorter side after resizing, and the
        length of both sides after cropping. Default is 256.
    interp : str, optional
        The interpolation mode passed to `scipy.misc.imresize`.
        Defaults to `'bicubic'`.

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
    resized = imresize(image, new_size, interp=interp)
    return resized[slices]


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
        numpy.concatenate(c, -numpy.ones(synsets.shape[0] - len(c),
                                         dtype=numpy.int16))
        for c in children
    ]
    new_synsets['children'] = padded_children
    return new_synsets, cost_matrix


def extract_patch_images(f):
    """Extracts a dict of dicts of the "patch images" for ILSVRC2010.

    Parameters
    ----------
    f : str or file-like object
        The filename or file-handle to the patch images TAR file.

    Returns
    -------
    dict
        A dict containing three keys: 'train', 'val', 'test'. Each dict
        contains a mapping of filenames (without path) to a NumPy array
        containing the replacement image.

    Notes
    -----
    Certain images in the distributed archives are blank, or display
    an "image not available" banner. A separate TAR file of
    "patch images" is distributed with the corrected versions of
    these. It is this archive that this function is intended to read.

    """
    patch_images = {'train': {}, 'val': {}, 'test': {}}
    with _open_tar_file(f) as tar:
        for info_obj in tar:
            if not info_obj.name.endswith('.JPEG'):
                continue
            _, which_set, _, filename = os.path.split(info_obj.name)
            image = imread(tar.extractfile(info_obj.name))
            patch_images[which_set][filename] = image
    return patch_images
