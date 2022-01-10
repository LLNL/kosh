def auto_valid(validation):
    """auto_valid converts class type
    to their 'isinstance' function in the future other
    keyword maybe added to the list

    :param validation: class def or type
    :type validation: type
    :return: isinstance lambda function or validation function
    :rtype: function
    """
    if isinstance(validation, type):
        def lmb(x): return isinstance(x, validation)
        return lmb
    return validation


def validate_value(value, validation):
    """validate_value Validates a value for a given validation function (or list of)

    :param value: value to validate
    :type value: any
    :param validation: validation function (or list of)
    :type validation: callable
    :raises ValueError: value does not vaidate
    :return: True if validates
    :rtype: bool
    """
    if validation is None:
        return True
    validation = auto_valid(validation)
    if callable(validation):
        res = validation(value)
        if not res:
            raise ValueError("value {value} failed validation".format(value=value))
    elif isinstance(validation, list):
        success = False
        for item in validation:  # Loop thru elements to get a possible success
            item = auto_valid(item)
            if callable(item):
                try:
                    res = item(value)
                    success += res
                    if success:
                        break  # no need to check further
                except Exception:
                    pass
            elif value == item:
                success += True
                break  # no need to check further

        if not success:  # All options failed
            raise ValueError("Could not validate value '{value}'".format(value=value))
    return True


class KoshSchema(object):
    def __init__(self, required={}, optional={}):
        """Schema for Kosh objects
        :param required: Dictionary of required keys and their validation function(s).
        :type required: dict
        :param optional: Dictionary of optional keys and their validation function(s)
        :type optional: dict
        Validation functions must be pickable.
        If multiple validation functions are provided (in a list),
        then the attribute is valid if ANY function return True.
        Validation function must return True, False or raise an Exception.
        """
        self.required = required
        self.optional = optional

    def __str__(self):
        """string representation"""
        st = """Kosh Validation Object
        Required attributes and their validations:
        {}
        Optional attributes and their validations:
        {}""".format(self.required, self.optional)
        return st

    def validate(self, obj):
        """validate an object through a schema
        Checks that the obj has all the required attribute
        and that both required and present optional attributes pass
        validation functions.

        :param obj: object to validate
        :type obj: str
        :raises ValueError: obj does not validate through the schema
        :return: True if validates
        "rtype: bool
        """
        # First check all the required keys
        req_errors = {}
        for k, v in self.required.items():
            try:
                value = getattr(obj, k)
                validate_value(value, v)
            except Exception as err:
                req_errors[k] = err
        opt_errors = {}
        for k, v in self.optional.items():
            try:
                value = getattr(obj, k)
            except AttributeError:
                continue
            try:
                validate_value(value, v)
            except Exception as err:
                opt_errors[k] = err

        if len(req_errors) != 0 or len(opt_errors) != 0:
            raise ValueError(
                "Could not validate {}\n"
                "{} required attribute errors: {}\n"
                "{} optional attributes errors: {}".format(
                    obj.id, len(req_errors), req_errors, len(opt_errors), opt_errors))
        return True

    def validate_attribute(self, attribute, value):
        """validate_attribute validates a value for a specific attribute

        :param attribute: attribute to validate
        :type attribute: str
        :param value: value to validate
        :type value: any
        :return: True or False
        "rtype: bool
        """
        if attribute in self.required:
            result = validate_value(value, self.required[attribute])
        elif attribute in self.optional:
            result = validate_value(value, self.optional[attribute])
        else:
            result = True
        return result
