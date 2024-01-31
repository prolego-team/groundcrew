import ast

from dataclasses import dataclass

from radon.visitors import ComplexityVisitor


@dataclass
class Import:
    name: str
    asname: str


def get_imports_from_code(code: str) -> list[Import]:
    """Returns a list of Import objects for the given code."""
    imports = []

    parsed = ast.parse(code)
    assert isinstance(parsed, ast.Module)
    for line in parsed.body:
        if isinstance(line, ast.Import):
            for imp in line.names:
                module_name = imp.name
                alias = imp.asname if imp.asname is not None else module_name
                imports.append(Import(module_name, alias))
        elif isinstance(line, ast.ImportFrom):
            module_name = line.module
            for entity in line.names:
                name = entity.name
                alias = entity.asname if entity.asname is not None else name
                imports.append(Import(f'{module_name}.{name}', alias))
                if line.level > 0:
                    print('Warning: import with level > 0')

    return imports


def imports_entity(
        imports: list[Import],
        importable_object: str,
    ) -> bool:
    """Returns True if the imports list imports the entity from the module.

    Imports are checked by comparing the first n period-separated elements of
    the importable_object where n is the number of overlapping elements.
    For example:

    - importable_object = 'numpy.random.rand'
        - 'import numpy' in file returns True
        - 'import numpy.random' in file returns True
        - 'import numpy.random.rand' in file returns True
        - 'import numpy.random.randn' in file returns False
        - 'import random.rand' in file returns False
    - importable_object = 'torch.nn'
        - 'import torch' in file returns True
        - 'import torch.nn' in file returns True
        - 'import torch.nn.functional' in file returns True
        - 'import nn.functional' in file returns False

    Each import in the imports list should contain the fully qualified name
    of import objects, resolving aliases as needed.  c.f. get_imports_from_code.
    """
    test = importable_object.split('.')
    for imp in imports:
        imp_name = imp.name.split('.')
        n = min(len(imp_name), len(test))
        if test[:n] == imp_name[:n]:
            return True

    return False


def import_called_as(
        imports: list[Import],
        importable_object: str,
    ) -> list[str]:
    """Returns the names of the importable_object if the imports list imports that
    object.  The names are returned as they are referenced in the code, accounting
    for import aliases.

    A list of names is returned because the same object could be accessed by multiple
    imports (e.g., import numpy as np; import numpy.random as npr would
    For example:
    - importable_object = 'numpy.random.rand'
    - imports = get_imports_from_code('import numpy as np; import numpy.random as npr')
    - return value = ['np.random.rand', 'npr.rand']
    """
    test = importable_object.split('.')
    calls = []
    for imp in imports:
        imp_name = imp.name.split('.')
        n = min(len(imp_name), len(test))
        if test[:n] == imp_name[:n]:
            call = imp.asname
            if n < len(test):
                call = '.'.join([call] + test[n:])
            calls.append(call)

    return calls


def cyclomatic_complexity(code: str) -> dict:
    """Compute the cyclomatic complexity of a piece of code."""
    v = ComplexityVisitor.from_code(code)

    output = {
        func.name: {
            'object': 'function',
            'complexity': func.complexity
        }
        for func in v.functions
    }

    output.update({
        clss.name: {
            'object': 'class',
            'complexity': clss.complexity,
            'methods': {
                meth.name: {
                    'complexity': meth.complexity
                }
                for meth in clss.methods
            }
        }
        for clss in v.classes
    })

    return output
