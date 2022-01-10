import json
from .core import KoshLoader


class JSONLoader(KoshLoader):

    types = {"json": ["any", "dict", "list", "str"]}

    def list_features(self):
        with open(self.uri) as json_file:
            content = json.load(json_file)
        if isinstance(content, dict):
            return sorted(["content", ] + list(content.keys()))
        else:
            return ["content", ]

    def extract(self):
        features = self.feature
        if not isinstance(features, list) and features != "content":
            features = [self.feature, ]

        with open(self.uri) as json_file:
            content = json.load(json_file)

        if features == "content":
            out = content
        else:
            if self.format == "dict":
                out = {}
            else:
                out = []
            for feature in features:
                if self.format == "dict":
                    out[feature] = content[feature]
                else:
                    out.append(content[feature])
        if isinstance(features, list) and len(features) == 1 and self.format != "dict":
            return out[0]
        else:
            return out
