class KoshSchema(object):
    def __init__(self, name, valid_keys={"dataset": [
                 "name", ], "file": ["uri", "type"]}):
        if not isinstance(valid_keys, dict):
            raise TypeError("valid_key must be a dictionary")
        for key in ["dataset", ]:
            if "dataset" not in valid_keys:
                raise KeyError("valid_key must contain {} key".format(key))
        for key in valid_keys:
            if not isinstance(valid_keys[key], (list, tuple)):
                raise TypeError("valid_key items must be list or tuples")
        self.name = name
        self.valid_keys = valid_keys
