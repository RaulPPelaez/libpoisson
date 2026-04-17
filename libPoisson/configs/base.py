from dataclasses import dataclass as _dataclass

@_dataclass
class BaseConfig:
    """
    Parameters
    ----------
    charge_radius: float
        Charge radius, associated with gaussian charge distribution.
    perm: float
        Permittivity of the medium.
    """
    charge_radius: float
    perm: float

def dataclass(*args, **kwargs):
    def wrapper(cls):
        cls = _dataclass(cls, **kwargs)

        docs = []

        for base in reversed(cls.__mro__):
            if base.__doc__:
                docs.append(base.__doc__.strip())

        cls.__doc__ = "\n\n".join(docs)

        return cls

    if len(args) == 1 and callable(args[0]):
        return wrapper(args[0])

    return wrapper
