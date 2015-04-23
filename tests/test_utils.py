import errno
import pickle
from six.moves import range
from zmq import ZMQError

from fuel.utils import do_not_pickle_attributes
from fuel.utils.zmq import uninterruptible


@do_not_pickle_attributes("non_pickable", "bulky_attr")
class TestClass(object):
    def __init__(self):
        self.load()

    def load(self):
        self.bulky_attr = list(range(100))
        self.non_pickable = lambda x: x


def test_do_not_pickle_attributes():
    cl = TestClass()

    dump = pickle.dumps(cl)

    loaded = pickle.loads(dump)
    assert loaded.bulky_attr == list(range(100))
    assert loaded.non_pickable is not None


def test_uninterruptible():
    foo = []

    def interrupter(a, b):
        if len(foo) < 3:
            foo.append(0)
            raise ZMQError(errno=errno.EINTR)
        return (len(foo) + a) / b

    def noninterrupter():
        return -1

    assert uninterruptible(interrupter, 5,  2) == 4
