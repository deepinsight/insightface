import numpy as np
from numpy.linalg import norm as l2norm


class Face(dict):
    __slots__ = ("__dict__",)

    def __init__(self, d=None, **kwargs):
        super().__init__()
        if d:
            self.update(d)
        if kwargs:
            self.update(kwargs)

    def __setattr__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Face):
            value = Face(value)
        elif isinstance(value, (list, tuple)):
            value = [
                Face(v) if isinstance(v, dict) else v
                for v in value
            ]
        super().__setattr__(key, value)
        super().__setitem__(key, value)

    __setitem__ = __setattr__

    def __getattr__(self, key):
        # avoid AttributeError cost
        return self.get(key, None)

    @property
    def embedding_norm(self):
        emb = self.get("embedding")
        if emb is None:
            return None
        return l2norm(emb)

    @property
    def normed_embedding(self):
        emb = self.get("embedding")
        if emb is None:
            return None
        n = l2norm(emb)
        if n == 0:
            return emb
        return emb / n

    @property
    def sex(self):
        g = self.get("gender")
        if g is None:
            return None
        return "M" if g == 1 else "F"
