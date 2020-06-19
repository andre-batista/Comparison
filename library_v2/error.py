"""Error classes which may be risen."""


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        """Save a message and the wrong code expression."""
        self.expression = expression
        self.message = message
