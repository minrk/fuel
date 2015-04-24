import errno
import pickle
from six.moves import range
import zmq

from fuel.utils import do_not_pickle_attributes
from fuel.utils.zmq import uninterruptible
from fuel.utils.zmq import (DivideAndConquerVentilator, DivideAndConquerSink,
                            DivideAndConquerWorker,
                            LocalhostDivideAndConquerManager)


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
            raise zmq.ZMQError(errno=errno.EINTR)
        return (len(foo) + a) / b

    def noninterrupter():
        return -1

    assert uninterruptible(interrupter, 5,  2) == 4


class DummyVentilator(DivideAndConquerVentilator):
    def send(self, socket, number):
        socket.send_pyobj(number)

    def produce(self):
        for i in range(50):
            yield i


class DummyWorker(DivideAndConquerWorker):
    def recv(self, socket):
        return socket.recv_pyobj()

    def send(self, socket, number):
        socket.send_pyobj(number)

    def process(self, number):
        return number ** 2


class DummySink(DivideAndConquerSink):
    def __init__(self, result_port):
        self.result_port = result_port
        self.messages_received = 0
        self.sum = 0

    def recv(self, socket):
        return socket.recv_pyobj()

    def done(self):
        return self.messages_received >= 50

    def setup_sockets(self, context, *args, **kwargs):
        super(DummySink, self).setup_sockets(context, *args, **kwargs)
        self.result_socket = context.socket(zmq.PUSH)
        self.result_socket.connect('tcp://localhost:{}'.format(
            self.result_port))

    def process(self, number_squared):
        self.sum += number_squared

    def shutdown(self):
        self.result_socket.send_pyobj(self.sum)


def test_localhost_divide_and_conquer_manager():
    result_port = 59591
    ventilator_port = 59592
    sink_port = 59593
    manager = LocalhostDivideAndConquerManager(DummyVentilator(),
                                               DummySink(result_port),
                                               [DummyWorker(), DummyWorker()],
                                               ventilator_port, sink_port)
    context = zmq.Context()
    result_socket = context.socket(zmq.PULL)
    result_socket.bind('tcp://*:{}'.format(result_port))
    manager.launch()
    manager.wait_for_sink()
    result = result_socket.recv_pyobj()
    assert result == sum(i ** 2 for i in xrange(50))

