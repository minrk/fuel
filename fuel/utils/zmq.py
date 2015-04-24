from abc import ABCMeta, abstractmethod
import errno
from multiprocessing import Process

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

def bind_to_address_port_or_range(socket, addr_or_port, default_addr='tcp://*',
                                  max_retries=100):
    """Bind to an address, port or random port in a range.

    Parameters
    ----------
    socket : zmq.Socket
        The socket to bind.
    addr_or_port : str, int, or tuple
        If string, this is interpeted as a fully-qualified address
    default_addr : str
        The address (with protocol, no port) to use if a port or
        port range is specified.
    max_retries : int, optional
        The maximum number of retries to perform in the case of
        selecting randomly from a port range.

    Returns
    -------
    port : int
        The port on which the socket was bound.

    """
    if isinstance(addr_or_port, (list, tuple)):
        min_port, max_port = addr_or_port
        return socket.bind_to_random_port(min_port, max_port, max_retries)
    else:
        if isinstance(addr_or_port, six.string_types):
            port = int(addr_or_port.split(':')[-1])
            socket.bind(addr_or_port)
            return port
        else:
            port = addr_or_port
        return port

@six.add_metaclass(ABCMeta)
class DivideAndConquerBase(object):
    """Base class for divide-and-conquer-over-ZMQ components."""

    setup_done = False

    @abstractmethod
    def setup_sockets(self, context):
        """Set up the receiver and sender sockets given a ZeroMQ context.

        Parameters
        ----------
        context : zmq.Context
            A ZeroMQ context.

        """
        self.setup_done = True
        self.context = context

    @abstractmethod
    def run(self, context=None):
        """Start doing whatever this component needs to be doing."""

    def check_setup(self):
        """Check that sockets have been set up, raise an error if not."""
        if not self.setup_done:
            raise ValueError('setup_sockets() must be called before run()')

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


@six.add_metaclass(ABCMeta)
class DivideAndConquerVentilator(DivideAndConquerBase):
    default_addr = 'tcp://*'

    """The ventilator serves tasks on a PUSH socket to workers."""
    def setup_sockets(self, context, sender_spec, sender_hwm, sink_spec):
        """Set up sockets for task dispatch.

        Parameters
        ----------
        sender_spec : str, int, or tuple
            The address spec (e.g. `tcp://*:9534`), port (as an
            integer), or port range (e.g `(9000, 9050)` on which the
            ventilator should listen for worker connections and send
            messages. If a port range is specified,
        sender_hwm : int, optional
            High water mark to set on the sender socket. Default
            is to not set one.
        sink_spec : str or int
            The address (e.g. `tcp://somehost:5678`) or port (as an
            integer) on which the ventilator should connect to the
            sink in order to synchronize the start of work.

        """
        self._sender = context.socket(zmq.PUSH)
        if self._sender.hwm is not None:
            self._sender.hwm = self.hwm
            if isinstance(sender_spec, (list, tuple)):
                self.port = self._sender.bind_to_random_port(*sender_spec)
            else:
                sender_spec = self.as_spec(sender_spec,
                                           self.default_addr + ':{}')
                self._sender.bind(sender_spec)
                if isinstance(self.sender_spec, six.string_types):
                    self.port = int(self.sender_spec.split(':')[-1])
                else:
                    self.port = self.sender_spec

        self._sink = context.socket(zmq.PUSH)
        self._sink.connect(self.as_spec(sink_spec, 'tcp://localhost:{}'))
        super(DivideAndConquerVentilator, self).setup_sockets(context)

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

    def run(self):
        try:
            self.check_setup()
            self._sink.send(b'0')
            for batch in self.produce():
                self.send(self._sender, *batch)
        finally:
            self.context.destroy()


@six.add_metaclass(ABCMeta)
class DivideAndConquerWorker(DivideAndConquerBase):
    """A worker receives tasks from a ventilator, sends results to a sink.

    """
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

    def setup_sockets(self, context, receiver_spec, receiver_hwm,
                      sender_spec, sender_hwm):
        """Set up sockets for receiving tasks and sending results.

        Parameters
        ----------
        receiver_spec : str or int, optional
            The address (e.g. `tcp://somehost:9534`) or port (as an
            integer) on which the worker should listen for jobs
            from the ventilator.
        sender_spec : str or int, optional
            The address (e.g. `tcp://somehost:9534`) or port (as an
            integer) on which the worker should connect to the sink.
        receiver_hwm : int, optional
            High water mark to set on the receiver socket. Default
            is to not set one.
        sender_hwm : int, optional
            High water mark to set on the sender socket. Default
            is to not set one.

        """

        self._receiver = context.socket(zmq.PULL)
        if receiver_hwm is not None:
            self._receiver.hwm = receiver_hwm
        self._receiver.bind(self.as_spec(receiver_spec, 'tcp://*:{}'))
        self._sender = context.socket(zmq.PUSH)
        if sender_hwm is not None:
            self._sender.hwm = sender_hwm
        self._sender.connect(self.as_spec(sender_spec, 'tcp://localhost:{}'))
        super(DivideAndConquerWorker, self).setup_sockets(context)

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

    def run(self):
        self.check_setup()
        self.work_loop()


@six.add_metaclass(ABCMeta)
class DivideAndConquerSink(DivideAndConquerBase):
    """A sink receives results from workers and processes them."""
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

    def setup_sockets(self, context, receiver_spec, receiver_hwm):
        """Set up sockets for receiving results from workers.

        Parameters
        ----------
        receiver_spec : str or int
            The address (e.g. `tcp://somehost:9534`) or port (as an
            integer) on which the receiver should listen for worker
            results.
        receiver_hwm : int, optional
            High water mark to set on the receiver socket. Default
            is to not set one.

        """
        self._receiver = context.socket(zmq.PULL)
        if receiver_hwm is not None:
            self._receiver.hwm = receiver_hwm
        self._receiver.bind(self.as_spec(receiver_spec, 'tcp://*:{}'))
        super(DivideAndConquerSink, self).setup_sockets(context)

    def run(self):
        self.check_setup()
        try:
            while not self.done():
                self.process(self.recv(self._receiver))
        finally:
            self.shutdown()

    def shutdown(self):
        """Called just before :method:`run` terminates.

        Notes
        -----
        This is called even in case of error, via a `finally` block.

        """


class LocalhostDivideAndConquerManager(object):
    """Manages a ventilator, sink and workers running locally.

    Parameters
    ----------
    ventilator : DivideAndConquerVentilator
        Instance of a class derived from
        :class:`DivideAndConquerVentilator`.
    sink : DivideAndConquerSink
        Instance of a class derived from :class:`DivideAndConquerSink`.
    workers : list of DivideAndConquerWorkers
        A list of instances of a class derived from
        :class:`DivideAndConquerWorker`.
    ventilator_port : int
        The port on which the ventilator will communicate with
        workers.
    sink_port : int, optional
        The port on which the workers and ventilator will communicate
        with the sink.
    ventilator_hwm : int, optional
        The high water mark to set on the ventilator's PUSH socket.
        Default is to leave the high water mark unset.
    worker_receiver_hwm : int, optional
        The high water mark to set on each worker's PULL socket.
        Default is to leave the high water mark unset.
    worker_sender_hwm : int, optional
        The high water mark to set on each worker's PUSH socket.
        Default is to leave the high water mark unset.
    sink_hwm : int, optional
        The high water mark to set on the sink's PULL socket.
        Default is to leave the high water mark unset.

    """
    def __init__(self, ventilator, sink, workers,
                 ventilator_port, sink_port, ventilator_hwm=None,
                 worker_receiver_hwm=None, worker_sender_hwm=None,
                 sink_hwm=None):
        self.ventilator = ventilator
        self.sink = sink
        self.workers = workers
        self.processes = []
        self.ventilator_port = ventilator_port
        self.sink_port = sink_port
        self.ventilator_hwm = ventilator_hwm
        self.worker_receiver_hwm = worker_receiver_hwm
        self.worker_sender_hwm = worker_sender_hwm
        self.sink_hwm = sink_hwm

    def launch_worker(self, worker):
        """Launch a worker.

        Parameters
        ----------
        worker : DivideAndConquerWorker
            An object representing the worker to be run.

        Notes
        -----
        Intended to be run inside a forked process.

        """
        context = zmq.Context()
        worker.setup_sockets(context, self.ventilator_port,
                             self.worker_receiver_hwm, self.sink_port,
                             self.worker_sender_hwm)
        worker.run()

    def launch_ventilator(self):
        """Launch the ventilator.

        Notes
        -----
        Intended to be run inside a forked process.

        """
        context = zmq.Context()
        self.ventilator.setup_sockets(context, self.ventilator_port,
                                      self.ventilator_hwm, self.sink_port)
        self.ventilator.run()

    def launch_sink(self):
        """Launch the sink.

        Notes
        -----
        Intended to be run inside a forked process.

        """
        context = zmq.Context()
        self.sink.setup_sockets(context, self.sink_port, self.sink_hwm)
        self.sink.run()

    def launch(self):
        """Launch ventilator, workers and sink in separate processes."""
        ventilator_process = Process(target=self.launch_ventilator)
        worker_processes = [Process(target=self.launch_worker, args=(worker,))
                            for worker in self.workers]
        sink_process = Process(target=self.launch_sink)
        for process in [ventilator_process, sink_process] + worker_processes:
            process.start()

        # Attribute assignment after all the processes are started, so that
        # process handles don't get copied to any of the other processes.
        self.ventilator_process = ventilator_process
        self.worker_processes = worker_processes
        self.sink_process = sink_process
        self.processes.extend([self.ventilator_process, self.sink_process] +
                              self.worker_processes)

    def cleanup(self):
        """Kill any launched processes that are still alive."""
        for process in self.processes:
            if process.is_alive():
                process.terminate()

    def wait_for_sink(self):
        """Wait for the sink process to terminate, then clean up."""
        self.sink_process.join()
        self.cleanup()
