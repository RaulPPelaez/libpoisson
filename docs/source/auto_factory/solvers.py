from libPoisson.registry.core import _SOLVER_REGISTRY as REGISTRY
from pathlib import Path


def make_solver_rst(solver_cls):
    title = solver_cls.__name__
    full_name = f"{solver_cls.__module__}.{solver_cls.__name__}"

    format_title = f".. _{full_name}:\n\n{title}\n{'^' * len(title)}\n\n"
    autodoc = f".. autoclass:: {full_name}\n\t:noindex:\n\n"

    return format_title + autodoc


def build_solver_classes():
    full_page = ""
    page = {}
    for key, solver in REGISTRY.items():
        if len(key) != 3:
            continue
        device = key[1]
        if device not in page:
            page[device] = ""
        docu = make_solver_rst(solver.solver_cls)
        page[device] += docu

    for device, content in page.items():
        page[device] = f"{device.value}\n{'-' * len(device.value)}\n\n" + content
        full_page += page[device] + "\n\n"
    out = Path(__file__).resolve().parent.parent / "solvers/all_solvers.rst"
    out.write_text(full_page, encoding="utf-8")
