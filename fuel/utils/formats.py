"""Low-level utilities for reading a variety of source data storage formats."""
import os
import struct

import numpy
import six
from six.moves import cPickle, reduce

from fuel.utils import open_if_filename


def _unpickle_py2(f):
    """Wrapper that unpickles Python 2.x-pickled files correctly.

    Parameters
    ----------
    f : file-like object
        An open file-like object containing a stream of latin1-encoded
        pickled data.

    Returns
    -------
    obj : object
        The deserialized data.
    """
    if six.PY3:
        return cPickle.load(f, encoding='latin1')
    else:
        return cPickle.load(f)


def load_cifar_batch(f):
    """Load images and labels from a single CIFAR{10,100} pickle file.

    Parameters
    ----------
    f : str or file-like object
        If `f` is a string it is presumed to be a filename on disk.
        Otherwise, `f` is presumed to be a file-like object, open
        in mode 'rb'.

    Returns
    -------
    images : ndarray, 2-dimensional
        A 2-dimensional ndarray of shape (10000, 3072) containing
        pixel data for the given batch.
    labels : ndarray, 1-dimensional
        A 1-dimensional ndarray of shape (10000,) containing the
        label for each example.
    """
    with open_if_filename(f, 'rb'):
        batch = _unpickle_py2(f)
        images = batch['data'].reshape((10000, 3, 32, 32))
        labels = numpy.asarray(batch['labels'], dtype='uint8')
        return images, labels


def load_cifar_data(base_path, which_set):
    """Utility function for loading CIFAR10 and CIFAR100 pickled data.

    Parameters
    ----------
    base_path : str
        Path on the filesystem where the pickle files can be found.
    which_set : str
        A string indicating which set to load, 'train' or 'test'.

    Returns
    -------
    images : ndarray, 4-dimensional, uint8
        The first axis indexes examples, the second indexes colour
        channels, the third and fourth index rows and columns,
        respectively.
    labels : ndarray, 1-dimensional, uint8
        Integer labels corresponding to the class of each example
        in `images`.
    label_names : list of str
        Human-interpretable label names for each integer label.

    """
    files = {'train': ['data_batch_{}'.format(i) for i in range(1, 6)],
             'test': ['test_batch']}
    num_examples = 50000 if which_set == 'train' else 10000
    image_shape = (3, 32, 32)
    images = numpy.zeros((num_examples,) + image_shape,
                         dtype='uint8')
    labels = numpy.zeros((num_examples,), dtype='uint8')
    batches = [load_cifar_batch(os.path.join(base_path, fname))
               for fname in files[which_set]]
    images, labels = [numpy.concatenate(arrays) for arrays in zip(*batches)]
    with open(os.path.join(base_path, 'batches.meta'), 'rb') as f:
        label_names = _unpickle_py2(f)['label_names']
    return images, labels, label_names


def load_idx_file(f, byte_order='>'):
    """Load files stored in 'IDX' format.

    Parameters
    ----------
    fname : str or file-like object
        If `f` is a string it is presumed to be a filename on disk.
        Otherwise, `f` is presumed to be a file-like object, open
        in mode 'rb'.
    byte_order : str, optional
        '>' for big-endian (default), '<' for little-endian,
        '=' for platform-native order (not recommended).

    Returns
    -------
    array : ndarray
        A multi-dimensional array, with number of dimensions,
        size and dtype determined by the metadata present in
        the IDX file header.

    Notes
    -----
    This format is used to distribute the original files of the
    MNIST handwritten digit database. This implementation is based
    on the description provided on the `MNIST web page`_.

    .. _MNIST web page: http://yann.lecun.com/exdb/mnist/
    """
    if byte_order not in ('>', '<', '='):
        raise ValueError("Invalid byte order: {}".format(byte_order))
    dtype_mappings = {0x08: 'uint8', 0x09: 'int8',
                      0x0B: 'int16', 0x0C: 'int32',
                      0x0D: 'float32', 0x0E: 'float64'}
    with open_if_filename(f, 'rb') as f:
        magic = f.read(4)
        b1, b2, dtype_byte, ndim = struct.unpack('BBBB', magic)
        if b1 != 0 or b2 != 0:
            raise IOError("Invalid header in file: {}".format(f))
        if dtype_byte not in dtype_mappings:
            raise ValueError("Unknown data type specifier: {}".format(
                hex(dtype_byte)))
        dims = struct.unpack(byte_order + 'i' * ndim,
                             f.read(struct.calcsize('i' * ndim)))
        return numpy.fromfile(f, count=reduce(lambda x, y: x * y, dims),
                              dtype=dtype_mappings[dtype_byte]).reshape(dims)
