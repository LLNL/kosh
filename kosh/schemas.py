class KoshSchema(object):
    def __init__(self, name, valid_keys={"dataset":["name",], "file":["uri", "type"]}):
        if not isinstance(valid_key, dict):
            raise TypeError("valid_key must be a dictionary")
        for key in ["dataset", ]:
            if not "dataset" in valid_keys:
                raise KeyError("valid_key must contain {} key".format(key))
        for key in valid_key:
            if not isinstance(valid_key[key], (list, tuple)):
                raise TypeError("valid_key items must be list or tuples")
        self.name = name
        self.valid_keys = valid_keys
