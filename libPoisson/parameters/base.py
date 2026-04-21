from dataclasses import dataclass as _dataclass

@_dataclass(kw_only=True)
class BaseParameters:
    """
    charge_radius: float
        Charge radius, associated with gaussian charge distribution.

    permittivity: float
        Permittivity of the medium.

    needs_complex: bool, optional
        Whether the potential needs to be complex-valued. Default is False.
    """
    charge_radius: float
    permittivity: float
    need_complex: bool = False

def child_dataclass(*args, **kwargs):
    def wrapper(cls):
        cls = _dataclass(cls, **kwargs, kw_only=True)

        docs = []

        base = cls.__bases__[0]
        cls.__doc__ = base.__doc__ + "\n" + cls.__doc__

        return cls

    if len(args) == 1 and callable(args[0]):
        return wrapper(args[0])

    return wrapper
