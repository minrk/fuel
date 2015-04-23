import logging
import os

import progressbar
import zmq


class ProgressBarHandler(logging.Handler):
    """A :class:`~logging.Handler` that drives a progress bar.

    Parameters
    ----------
    max_val_attr : str
        The attribute name used for the maximum value of the
        progress bar, used to trigger a new progress bar being started.
    curr_val_attr : str
        The attribute name used for the current value of the progress
        bar, used to trigger an update to the progress bar.
    start_predicate : callable, optional
        A callable that returns `True` if a given `LogRecord` signals
        the start of a progress bar. If unspecified, the presence of
        `max_val_attr` is used as the test.
    update_predicate : callable
        A callable that returns `True` if a given `LogRecord` signals
        an update of a progress bar. If unspecified, the presence of
        `curr_val_attr` is used as the test.
    level : int, optional
        The level of messages expected to contain progress bar signaling.
        Defaults to `logging.DEBUG`.

    """
    def __init__(self, max_val_attr, curr_val_attr, widgets=None,
                 start_predicate=None, update_predicate=None,
                 level=logging.DEBUG):
        super(ProgressBarHandler, self).__init__(level)
        self.max_val_attr = max_val_attr
        self.curr_val_attr = curr_val_attr
        self.widgets = widgets
        self.start_predicate = (start_predicate if start_predicate is not None
                                else lambda x: hasattr(x, self.max_val_attr))
        self.update_predicate = (update_predicate
                                 if update_predicate is not None
                                 else lambda x: hasattr(x, self.curr_val_attr))
        self.level = level

    def handle(self, record):
        """Handle a `LogRecord`, updating a progress bar if necessary."""
        if self.start_predicate(record):
            maxval = getattr(record, self.max_val_attr)
            self.progress_bar = progressbar.ProgressBar(
                maxval=maxval,
                widgets=self.widgets).start()
        elif self.update_predicate(record):
            currval = getattr(record, self.curr_val_attr)
            self.progress_bar.update(currval)


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
        """Send a `LogRecord` over the socket."""
        try:
            # Tracebacks aren't picklable, so cache the traceback text
            # and then throw away the traceback object. This seems to
            # allow the text to still be displayed by the default Formatter.
            if record.exc_info:
                record.exc_text = (
                    "Traceback (most recent call last):\n" +
                    "".join(traceback.format_tb(record.exc_info[2]))
                )
                record.exc_info = record.exc_info[:2] + (None,)
            self.socket.send_pyobj(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


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
    logger.propagate = False
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:{}".format(logging_port))
    while logger.handlers:
        logger.handlers.pop()
    logger.addHandler(ZMQLoggingHandler(socket))


def make_debug_logging_function(logger, process_type):
    """Return a callable for logging keyword argument-based debug messages.

    Parameters
    ----------
    logger : object
        Logger-like object with a `debug()` method matching the
        signature of the same method from :class:`logging.Logger`.
    process_type : str
        A string that will be included at the beginning of each message,
        along with the PID in parentheses.

    Returns
    -------
    debug_log_fn : callable
        A function that expects one mandatory argument `status`, and
        as many keyword arguments as desired. The keywords and their
        values will appear in the log message as *key=value* (sorted
        by key), and will also be available as attributes on the
        corresponding `LogRecord` object.

    """
    def _debug(status, **kwargs):
        pid = os.getpid()
        message_str = '{process_type}({pid}): {status} '.format(
            process_type=process_type, pid=pid, status=status)
        message_str += ' '.join('{key}={val}'.format(key=key, val=kwargs[key])
                                for key in sorted(kwargs))
        kwargs['process_type'] = process_type
        kwargs['status'] = status
        logger.debug(message_str, extra=kwargs)
    return _debug


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
        If empty (default), this loops forever until interrupted.
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
    launched several worker processes. They should each set up a
    logger with a :class:`ZMQLoggingHandler` (e.g., by using
    :func:`configure_zmq_process_logger`).

    """
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:{}".format(logging_port))
    while len(processes) == 0 or any(p.is_alive() for p in processes):
        try:
            message = receiver.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.ZMQError as exc:
            if exc.errno == zmq.EAGAIN:
                continue
            else:
                raise
        levelno = message.levelno
        logger.handle(message)
        if levelno >= failure_threshold:
            raise SubprocessFailure
