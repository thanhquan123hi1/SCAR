"""Text logging for evaluation progress, metrics, and failures."""
from contextlib import contextmanager
import logging
import sys


@contextmanager
def evaluation_logger(path):
    logger = logging.Logger('scar.evaluate', level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    try:
        for handler in (logging.FileHandler(path, encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        yield logger
    except BaseException:
        logger.exception('Evaluation interrupted/failed.')
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
