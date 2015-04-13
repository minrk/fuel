import logging


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
