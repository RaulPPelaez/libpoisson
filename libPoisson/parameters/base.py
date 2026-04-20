from dataclasses import dataclass as _dataclass

@_dataclass(kw_only=True)
class BaseParameters:
    """
    Parameters
    ----------
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

def dataclass(*args, **kwargs):
    def wrapper(cls):
        cls = _dataclass(cls,**kwargs, kw_only=True)

        docs = []

        for base in reversed(cls.__mro__):
            if base.__doc__:
                docs.append(base.__doc__.strip())

        cls.__doc__ = "\n\n".join(docs)

        return cls

    if len(args) == 1 and callable(args[0]):
        return wrapper(args[0])

    return wrapper
