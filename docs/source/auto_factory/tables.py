from pathlib import Path
import os
import sys
from libPoisson.registry.core import _SOLVER_REGISTRY as REGISTRY
from libPoisson.parameters.base import BaseParameters
from dataclasses import fields, is_dataclass
from dataclasses import fields, MISSING
from typing import get_origin, get_args, Union

def is_optional_type(tp):
    origin = get_origin(tp)
    return origin is Union and type(None) in get_args(tp)


def format_default(f):
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return None


def extract_params(cls, base_cls):
    base_fields = set()

    for b in base_cls.__mro__:
        if hasattr(b, "__dataclass_fields__"):
            base_fields.update(b.__dataclass_fields__.keys())

    params = []

    for f in fields(cls):
        if f.name in base_fields:
            continue

        tp = f.type
        type_name = getattr(tp, "__name__", str(tp))

        optional = (
            is_optional_type(tp)
            or f.default is not MISSING
            or f.default_factory is not MISSING
        )

        if optional:
            default = format_default(f)
            params.append(f"{f.name}: {type_name} = {default}")
        else:
            params.append(f"{f.name}: {type_name}")

    return params


def extract_user_params(cls):
    if not is_dataclass(cls):
        return ""

    base_fields = set()
    if is_dataclass(BaseParameters):
        base_fields = {f.name for f in fields(BaseParameters)}

    result = []
    for f in fields(cls):
        if f.name in base_fields:
            continue

        tp = f.type
        type_name = getattr(tp, "__name__", str(tp))
        optional = (
            is_optional_type(tp)
            or f.default is not MISSING
            or f.default_factory is not MISSING
        )
        if optional:
            default = format_default(f)
            result.append(f"{f.name}: {type_name} = {default}")
        else:
            result.append(f"{f.name}: {type_name}")

    if result == []:
        return ""
    return ".. code-block:: python\n\n\t\t"+",\n\t\t".join(result)

def make_solver_link(cls):
    full = f"{cls.__module__}.{cls.__name__}"
    return f":ref:`{full}`"

def build_solver_tables():
    default_rows = []
    all_rows = []

    def style(string):
        return f'.. code-block:: python\n\n\t\t"{string}"'
    def style_bc(bc):
        bcx = f'"{bc.x.value}"'
        bcy = f'"{bc.y.value}"'
        bcz = f'"{bc.z.value}"'
        return f'.. code-block:: python\n\n\t\t({bcx},\n\n\t\t {bcy},\n\n\t\t {bcz})'

    for key, solver in REGISTRY.items():
        params_str = extract_user_params(solver.parameters_cls)
        solver_cls = solver.solver_cls
        if len(key) == 2:
            bc, device = key
            default_rows.append((style_bc(bc), style(device.value), make_solver_link(solver_cls), params_str))

        else:
            bc, device, impl = key
            all_rows.append((style_bc(bc), style(device.value), style(impl), make_solver_link(solver_cls), params_str))


        def table(title, headers, rows):
            lines = [title, "-" * len(title), ""]
            lines.append(".. list-table::")
            lines.append("   :header-rows: 1\n")

            lines.append("   * - " + "\n     - ".join(headers))

            for row in rows:
                lines.append("   * - " + "\n     - ".join(map(str, row)))

            return "\n".join(lines)

        content = "\n\n".join([
            table("Default Solvers",
                  ["Boundary Conditions", "Device", "Solver", "Extra Parameters"], default_rows),

            table("All Solvers",
                  ["Boundary Conditions", "Device", "Implementation", "Solver", "Extra Parameters"], all_rows),
        ])

        out = Path(__file__).parent/"../solvers_tables.rst"
        out.write_text(content)
