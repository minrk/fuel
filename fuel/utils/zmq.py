from abc import ABCMeta, abstractmethod
import errno

import six
import zmq


def uninterruptible(f, *args, **kwargs):
    """Run a function, catch & retry on interrupted system call errors."""
    while True:
        try:
            return f(*args, **kwargs)
        except zmq.ZMQError as e:
            if e.errno == errno.EINTR:
                # interrupted, try again
                continue
            else:
                # real error, raise it
                raise


class DivideAndConquerBase(object):
    """Base class for divide-and-conquer-over-ZMQ components."""
    @abstractmethod
    def setup_sockets(self, context):
        """Set up the receiver and sender sockets given a ZeroMQ context.

        Parameters
        ----------
        context : zmq.Context
            A ZeroMQ context.

        """

    @abstractmethod
    def run(self, context=None):
        """Start doing whatever this component needs to be doing.

        Parameters
        ----------
        context : zmq.Context
            A ZeroMQ context to use for creating sockets. If one is not
            provided, one will be created. Note that you should use at
            most one context at a time per process.

        """

        pass

    @staticmethod
    def as_spec(spec, port_template):
        """Format a valid socket specification from (optionally) a port.

        Parameters
        ----------
        spec : str or int
            If string, interpreted as a full socket spec used by
            a connect or bind call. If spec, interpreted as a port
        port_template : str
            A string formatting template with `{}` where a port
            number should be placed.

        Returns
        -------
        str
            Either `spec` directly, if `spec` was initially a string,
            or `port_template` with `spec` (presumed integer)
            substituted in where `{}` appears (i.e. by `.format()`).

        """

        if isinstance(spec, six.string_types):
            return spec
        else:
            return port_template.format(spec)


class DivideAndConquerVentilator(DivideAndConquerBase):
    """The ventilator serves tasks on a PUSH socket to workers.

    Parameters
    ----------
    sender_spec : str or int
        The address spec (e.g. `tcp://*:9534`) or port (as an
        integer) on which the ventilator should listen for worker
        connections and send messages.
    sink_spec : str or int
        The address (e.g. `tcp://somehost:5678`) or port (as an
        integer) on which the ventilator should connect to the
        sink in order to synchronize the start of work.
    sender_hwm : int, optional
        High water mark to set on the sender socket. Default
        is to not set one.

    """
    def __init__(self, sender_spec, sink_spec, sender_hwm=None):
        self.sender_spec = sender_spec
        self.sink_spec = sink_spec
        self.sender_hwm = sender_hwm

    def setup_sockets(self, context):
        self._sender = context.socket(zmq.PUSH)
        if self._sender.hwm is not None:
            self._sender.hwm = self.hwm
        self._sender.bind(self.as_spec(self.sender_spec, 'tcp://*:{}'))
        self._sink = context.socket(zmq.PUSH)
        self._sink.connect(self.as_spec(self.sink_spec, 'tcp://localhost:{}'))

    @abstractmethod
    def send(self, socket, *items):
        """Send produced batch of work over the socket.

        Parameters
        ----------
        socket : zmq.Socket
            The socket on which to send.
        \*items
            Arguments representing a batch of work as yielded by
            :method:`produce`.

        """

    @abstractmethod
    def produce(self):
        """Generator that yields batches of work to send."""

    def run(self, context=None):
        context = context if context is not None else zmq.Context()
        try:
            self.setup_sockets()
            self._sink.send(b'0')
            for batch in self.produce():
                self.send(self._sender, *batch)
        finally:
            context.destroy()


@six.add_metaclass(ABCMeta)
class DivideAndConquerWorker(DivideAndConquerBase):
    """A worker receives tasks from a ventilator, sends results to a sink.

    Parameters
    ----------
    receiver_spec : str or int
        The address (e.g. `tcp://somehost:9534`) or port (as an
        integer) on which the worker should listen for jobs
        from the ventilator.
    sender_spec : str or int
        The address (e.g. `tcp://somehost:9534`) or port (as an
        integer) on which the worker should connect to the sink.
    receiver_hwm : int, optional
        High water mark to set on the receiver socket. Default
        is to not set one.
    sender_hwm : int, optional
        High water mark to set on the sender socket. Default
        is to not set one.

    """
    def __init__(self, receiver_spec, sender_spec,
                 receiver_hwm=None, sender_hwm=None):
        self.receiver_spec = receiver_spec
        self.sender_spec = sender_spec
        self.receiver_hwm = receiver_hwm
        self.sender_hwm = sender_hwm

    @abstractmethod
    def recv(self, socket):
        """Receive a message [from the ventilator] and return it."""
        pass

    @abstractmethod
    def send(self, socket, *to_sink):
        """Send a group of messages over a socket [to the sink]."""
        pass

    @abstractmethod
    def process(self, *received):
        """Generator that turns a received chunk into one or more outputs.

        Parameters
        ----------
        \*received
            A sequence of arguments as returned by :method:`recv`.

        Yields
        ------
        tuple
            Tuples of batches to be sent to the sink, used as arguments
            to :method:`send`.

        """
        pass

    def setup_sockets(self, context):
        self._receiver = context.socket(zmq.PULL)
        self._receiver.bind(self.as_spec(self.receiver_spec, 'tcp://*:{}'))
        self._sender = context.socket(zmq.PUSH)
        if self.sender_hwm is not None:
            self._sender.hwm = self.sender_hwm
        self._sender.connect(self.as_spec(self.sender_spec,
                                          'tcp://localhost:{}'))

    def done(self):
        """Indicate whether the worker should terminate.

        Notes
        -----
        Usually, a worker *can't* know that no further work batches will
        be dispatched, as it has no idea what other workers have done.
        However there are restricted cases where it is predictable, and
        one could potentially build in a mechanism for the ventilator
        to communicate this information. THe default implementation
        returns `False` unconditionally.

        """
        return False

    def work_loop(self):
        """Loop indefinitely receiving, processing and sending."""
        while not self.done():
            received = self.recv(self._receiver)
            for output in self.process(*received):
                self.send(self._sender, *output)

    def run(self, context=None):
        self.setup_sockets(context if context is not None else zmq.Context())
        self.work_loop()


class DivideAndConquerSink(DivideAndConquerBase):
    """A sink receives incoming results from workers and processes them.

    Parameters
    ----------
    receiver_spec : str or int
        The address (e.g. `tcp://somehost:9534`) or port (as an
        integer) on which the receiver should listen for worker
        results.
    receiver_hwm : int
        High water mark to set on the receiver socket. Default
        is to not set one.

    """
    def __init__(self, receiver_spec, receiver_hwm):
        self.receiver_spec = receiver_spec
        self.receiver_hwm = receiver_hwm

    @abstractmethod
    def recv(self, socket):
        """Receive and return results from a worker."""
        pass

    @abstractmethod
    def process(self, *results):
        """Process a batch of results as returned by :method:`recv`."""
        pass

    def done(self):
        """Indicate whether or not the sink should terminate."""
        return False

    def setup_sockets(self, context):
        self._receiver = context.socket(zmq.PULL)
        self._receiver.hwm = self.receiver_hwm
        self._receiver.bind(self.as_spec(self.receiver_spec))

    def run(self, context=None):
        self.setup_sockets(context if context is not None else zmq.context())
        while not self.done():
            self.process(self.recv(self._receiver))
