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


def from_port_or_addr(addr_or_port, default_addr):
    """Wrapper that accepts an address or a port number.

    Parameters
    ----------
    addr_or_port : str or int
        Fully qualified address (e.g. `tcp://somehost:5932`) or port
        (as integer).
    default_addr : str
        The address (with leading protocol, but no port) to use when
        a port `addr_or_port` is a port number.

    Returns
    -------
    result_addr : str
        Either `addr_or_port` itself (if string) or `default_addr`
        concatenated with it (if a port number).

    """
    return (addr_or_port if isinstance(addr_or_port, six.string_types)
            else ':'.join([default_addr, str(addr_or_port)]))


def bind_to_addr_port_or_range(socket, addr_or_port, default_addr,
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
        # Port range.
        min_port, max_port = addr_or_port
        return socket.bind_to_random_port(min_port, max_port, max_retries)
    else:
        socket.bind(from_port_or_addr(addr_or_port, default_addr))
        if isinstance(addr_or_port, six.string_types):
            # Fully-qualified address.
            port = int(addr_or_port.split(':')[-1])
        else:
            # Port number as integer.
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


@six.add_metaclass(ABCMeta)
class DivideAndConquerVentilator(DivideAndConquerBase):
    """The ventilator serves tasks on a PUSH socket to workers."""

    default_addr = 'tcp://*'

    def __init__(self):
        self.port = None

    def setup_sockets(self, context, sender_spec, sender_hwm=None,
                      sink_spec=None):
        """Set up sockets for task dispatch.

        Parameters
        ----------
        sender_spec : str, int, or tuple
            The address spec (e.g. `tcp://*:9534`), port (as an
            integer), or port range (e.g `(9000, 9050)` on which the
            ventilator should listen for worker connections and send
            messages. If a port range is specified, a random port
            in the range will be bound within that range.
        sender_hwm : int, optional
            High water mark to set on the sender socket. Default
            is to not set one.
        sink_spec : str or int
            The address (e.g. `tcp://somehost:5678`) or port (as an
            integer) on which the ventilator should connect to the
            sink in order to synchronize the start of work.

        Raises
        ------
        ZMQBindError
            If the worker socket cannot be bound.

        """
        self._sender = context.socket(zmq.PUSH)
        # if self._sender.hwm is not None:
        #     self._sender.hwm = sender_hwm
        self.port = bind_to_addr_port_or_range(self._sender, sender_spec,
                                               self.default_addr)
        self._sink = context.socket(zmq.PUSH)
        self._sink.connect(from_port_or_addr(sink_spec, 'tcp://localhost'))
        super(DivideAndConquerVentilator, self).setup_sockets(context)

    @abstractmethod
    def send(self, socket, batch):
        """Send produced batch of work over the socket.

        Parameters
        ----------
        socket : zmq.Socket
            The socket on which to send.
        batch : object
            Object representing a batch of work as yielded by
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
                self.send(self._sender, batch)
        finally:
            self.context.destroy()


@six.add_metaclass(ABCMeta)
class DivideAndConquerWorker(DivideAndConquerBase):
    """A worker receives tasks from a ventilator, sends results to sink."""

    default_addr = 'tcp://localhost'

    @abstractmethod
    def recv(self, socket):
        """Receive a message [from the ventilator] and return it."""
        pass

    @abstractmethod
    def send(self, socket, result):
        """Send results over a socket [to the sink].

        Parameters
        ----------
        socket : zmq.Socket
            Socket on which to send results.
        results : object
            Object representing results as yielded by :func:`process`.

        """
        pass

    @abstractmethod
    def process(self, received):
        """Generator that turns a received chunk into one or more outputs.

        Parameters
        ----------
        received : object
            A received object representing a batch of work as returned by
            :method:`recv`.

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
        self._receiver.connect(from_port_or_addr(receiver_spec,
                                                 self.default_addr))
        self._sender = context.socket(zmq.PUSH)
        if sender_hwm is not None:
            self._sender.hwm = sender_hwm
        self._sender.connect(from_port_or_addr(sender_spec, self.default_addr))
        super(DivideAndConquerWorker, self).setup_sockets(context)

    def done(self):
        """Indicate whether the worker should terminate.

        Notes
        -----
        Usually, a worker *can't* know that no further work batches will
        be dispatched, as it has no idea what other workers have done.
        However there are restricted cases where it is predictable, and
        one could potentially build in a mechanism for the ventilator
        to communicate this information. The default implementation
        returns `False` unconditionally.

        """
        return False

    def work_loop(self):
        """Loop indefinitely receiving, processing and sending."""
        while not self.done():
            received = self.recv(self._receiver)
            for output in self.process(received):
                self.send(self._sender, output)

    def run(self):
        self.check_setup()
        self.work_loop()


@six.add_metaclass(ABCMeta)
class DivideAndConquerSink(DivideAndConquerBase):
    """A sink receives results from workers and processes them."""

    default_addr = 'tcp://*'

    def __init__(self):
        self.port = None

    @abstractmethod
    def recv(self, socket):
        """Receive and return results from a worker."""
        pass

    @abstractmethod
    def process(self, results):
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
        self.port = bind_to_addr_port_or_range(self._receiver, receiver_spec,
                                               self.default_addr)
        super(DivideAndConquerSink, self).setup_sockets(context)

    def run(self):
        self.check_setup()
        # Synchronize with the ventilator.
        self._receiver.recv()
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
