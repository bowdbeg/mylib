from pathlib import Path
import json

legal_type = [int, str, float, bool, list, tuple, dict, set]


class Namespace:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def json(self):
        dic = self.__dict__
        for key in dic.keys():
            if not any([isinstance(dic[key], t) for t in legal_type]):
                dic[key] = str(dic[key])
        return json.dumps(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    def load(self, json_file: Path):
        self.__dict__.update(json.loads(json_file.read_text()))
        return self

    def __repr__(self) -> str:
        return self.json()

    def dict(self):
        return self.__dict__