def auto_valid(validation):
    if validation in [int, float, str]:
        return lambda x: isinstance(x, validation)
    return validation


def validate_value(value, validation):
    if validation is None:
        return
    validation = auto_valid(validation)
    if callable(validation):
        res = validation(value)
        if not res:
            raise ValueError(f"value {value} failed validation")
    elif isinstance(validation, list):
        success = False
        for item in validation:  # Loop thru elements to get a possible success
            if callable(item):
                try:
                    res = item(value)
                    success += res
                    break  # no need to check further
                except Exception:
                    pass
            elif value == item:
                success += True
                break  # no need to check further

        if success is False:  # All options failed
            raise ValueError(f"Could not validate value '{value}'")


class KoshSchema(object):
    def __init__(self, required={}, optional={}):
        """Schema for Kosh objects"""
        self.required = required
        self.optional = optional

    def validate(self, data):
        # First check all the required keys
        req_errors = {}
        for k, v in self.required.items():
            try:
                value = getattr(data, k)
                validate_value(value, v)
            except Exception as err:
                req_errors[k] = err
        opt_errors = {}
        for k, v in self.optional.items():
            try:
                value = getattr(data, k)
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
                    data.__id__, len(req_errors), req_errors, len(opt_errors), opt_errors))

    def validate_attribute(self, attribute, value):
        if attribute in self.required:
            validate_value(value, self.required[attribute])
        elif attribute in self.optional:
            validate_value(value, self.optional[attribute])
