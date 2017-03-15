class InvalidTagmeKeyException(Exception):
    """The provided Tagme key is not valid

    See http://tagme.di.unipi.it/tagme_help.html for further information on how to get a Tagme key and to access Tagme
    API documentation"""
    pass


class NumberOfCirclesOutOfRangeException(Exception):
    """The number of requested ego network circles is outside the range of accepted values

    The number of circles must be greater than 0 (a negative or null number of circles does not make sense) and lesser
    than 100 (ego networks are expected to have a small number of circles)"""
    pass


class InvalidTimestampException(Exception):
    """The provided timestamp is not valid

    When two timestamps are requested, so as to define a time range, the first timestamp (usually *from_timestamp*)
    must be lesser than the second one (*to_timestamp*)"""
    pass


class InvalidIdException(Exception):
    """The value of the provided id is not valid

    ids for egos and alters must be positive ``int``"""
    pass


class InvalidIntValueException(Exception):
    """The value is not valid

    The ``int`` value provided is outside the accepted range"""
    pass


class InvalidSimilarityTypeException(Exception):
    """The requested similarity type is unknown"""
    pass


class InvalidNormalParamValueException(Exception):
    """The parameter value is not valid

    The parameter value is outside the range [0,1]"""
    pass

