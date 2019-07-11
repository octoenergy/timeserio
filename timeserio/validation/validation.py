import logging
from ..utils.pickle import loadf

logger = logging.getLogger()


PIPELINE_ERRORS = (ValueError, AttributeError)
PICKLE_ERRORS = (KeyError,)


def is_valid_transformer(transformer):
    """"""
    try:
        transformer.required_columns
        transformer.transformed_columns(transformer.required_columns)
        return True
    except PIPELINE_ERRORS:
        return False


def is_valid_pipeline(pipeline):
    """"""
    is_valid = is_valid_transformer(pipeline)
    returns_array = False
    if is_valid:
        transformed_columns = \
            pipeline.transformed_columns(pipeline.required_columns)
        if transformed_columns == {None}:
            returns_array = True
    return is_valid and returns_array


def is_valid_multipipeline(multipipeline):
    """"""
    all_valid = True
    for name, pipeline in multipipeline.pipelines.items():
        is_valid = is_valid_pipeline(pipeline)
        if not is_valid:
            logger.error('Invalid pipeline "%s"', name)
        all_valid &= is_valid
    return all_valid


def is_valid_multimodel(multimodel):
    """"""
    pipes_valid = is_valid_multipipeline(multimodel.multipipeline)
    return pipes_valid


def is_valid_pickle(path):
    """"""
    try:
        multimodel = loadf(path)
        return is_valid_multimodel(multimodel)
    except PICKLE_ERRORS:  # FixMe
        return False
