import logging
import zmq


class SubprocessFailure(Exception):
    """Raised by :func:`zmq_log_and_monitor` upon unrecoverable error."""
    pass


class ZMQLoggingHandler(logging.Handler):
    """A `logging.Handler` subclass that sends records over a ZMQ socket.

    Parameters
    ----------
    socket : zmq.Socket instance
        The socket over which to send `LogRecord` instances.
    level : int, optional
        The log level for this handler. Defaults to `logging.DEBUG`,
        so everything right down to debug messages gets forwarded
        through the socket.
    formatter : object, optional
        An object that provides a `formatException` method to be used
        to cache exception text before serialization (since traceback
        objects cannot be pickled). If not provided, a `logging.Formatter`
        instance is created and used.

    Notes
    -----
    A reasonable way to use this in a subprocess being driven by
    ZMQ is to create a logging socket connection to whatever process
    will be handling logging, and then installing this on the module
    logger from within the subprocess with `propagate` set to False
    to silence any default handlers on the root logger.

    """
    def __init__(self, socket, level=logging.DEBUG, formatter=None):
        super(ZMQLoggingHandler, self).__init__(level=level)
        self.socket = socket
        self.formatter = (logging.Formatter() if formatter is None
                          else formatter)

    def emit(self, record):
        try:
            # Tracebacks aren't picklable, so cache the traceback text
            # and then throw away the traceback object. This seems to
            # allow the text to still be displayed by the default Formatter.
            if record.exc_info:
                record.exc_text = self.formatter.formatException(
                    record.exc_info)
                record.exc_info = record.exc_info[:2] + (None,)
            self.socket.send_pyobj(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def zmq_log_and_monitor(logger, context, processes=(), logging_port=5559,
                        failure_threshold=logging.CRITICAL):
    """Feed `LogRecord`s received on a ZeroMQ socket to a logger.

    Parameters
    ----------
    logger : object
        Logger-like object with a `handle()` method similar that
        accepts :class:`logging.LogRecord` instances.
    processes : sequence, optional
        Collection containing :class:`multiprocessing.Process` objects.
        The loop will continue until none of these processes is alive.
        If empty (default), this
    logging_port : int, optional
        The port on which to initiate a ZeroMQ PULL socket and receive
        :class:`logging.LogRecord` messages.
    failure_threshold : int, optional
        Log-level at or above which a :class:`SubprocessFailure` should
        be raised. This allows processes to signal to initiate a
        shutdown of the whole system.

    Raises
    ------
    SubprocessFailure
        When a log message is received on the ZeroMQ socket with log
        level at or greater than `failure_threshold`.

    Notes
    -----
    This function is most useful when run from a process that has
    launched several worker processes..

    """
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:{}".format(logging_port))
    while len(processes) == 0 or any(p.is_alive() for p in processes):
        message = receiver.recv_pyobj()
        logger.handle(message)
        if message.levelno >= logging.ERROR:
            raise SubprocessFailure


def configure_zmq_process_logger(logger, context, logging_port):
    """Configures a logger object to log to a ZeroMQ socket.

    Parameters
    ----------
    logger : :class:`logging.Logger`
        A logger object, as returned by :func:`logging.getLogger`.
    context : :class:`zmq.Context`
        A ZeroMQ context.
    logging_port : int
        The port on localhost on which to open a `PUSH` socket
        for sending :class:`logging.LogRecord`s.

    Notes
    -----
    Mutates the logger object by removing any existing handlers,
    setting the `propagate` attribute to `False`, and adding a
    :class:`ZMQLoggingHandler` set up to log messages to a socket
    connected on `logging_port`.

    """
    logger = logging.getLogger(__name__)
    logger.propagate = False
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:{}".format(logging_port))
    while logger.handlers:
        logger.handlers.pop()
    logger.addHandler(ZMQLoggingHandler(socket))
