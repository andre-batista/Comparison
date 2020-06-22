"""Error classes which may be risen."""


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class MissingInputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        function_name -- a string containing the name of the function.
        input_names -- a string or a list of string with the names of
            the missing inputs.
        message -- explanation of the error
    """

    def __init__(self, function_name, input_names):
        """Save the name of the function and the missing inputs."""
        self.function_name = function_name
        self.input_names = input_names

        if isinstance(input_names, str):
            self.message = (
                'The argument ' + input_names + ' is missing at the '
                + 'function ' + function_name + '!'
            )

        else:
            n = len(input_names)
            self.message = (
                'The following arguments are missing at the function '
                + function_name + ': '
            )
            for i in range(n):
                self.message = self.message + input_names[i] + ' '

        super().__init__(self.message)


class ExcessiveInputsError(Error):
    """An error exception for excessive inputs.

    Attributes:.
        function_name -- a string with the name of the function.
        input_names -- a list of strings with the input names.
    """

    def __init__(self, function_name, input_names):
        """Save the name of the function and inputs."""
        self.function_name = function_name
        self.input_names = input_names
        self.message = 'You must given only one of the following inputs: '
        self.message = self.message + self.input_names[0]
        for i in range(1, len(input_names)):
            self.message = self.message + ' or ' + self.input_names[i]
        super().__init__(self.message)


class MissingAttributesError(Error):
    """Exception raised when some attribute is missing within an object.

    Attributes:
        class_name -- a string with the name of the class.
        attribute_name -- the name of the missing attribute.
    """

    def __init__(self, class_name, attribute_name):
        """Save the class of the object and the missing attribute."""
        self.class_name = class_name
        self.attribute_name = attribute_name
        super().__init__('Attribute ' + self.attribute_name + ' of class '
                         + self.class_name + ' is missing!')
