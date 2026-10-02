from dataclasses import dataclass


@dataclass
class DataClassDict:
    def __getitem__(self, key):
        return getattr(self, key)
