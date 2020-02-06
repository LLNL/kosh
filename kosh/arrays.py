
class KoshAxis(object):
    def __init__(self, id, values):
        self.id = id
        self.__values = values

    def __getitem__(self, key):
        return self.__values[key]

    def __setitem__(self, key, value):
        self.__values[key] = value

    def __len__(self):
        return len(self.__values)
