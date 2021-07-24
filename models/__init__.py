import logging
logger = logging.getLogger('base')


def create_model(opt):
    # image restoration
    from .Video_base_model import VideoBaseModel as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
