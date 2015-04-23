import errno
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
