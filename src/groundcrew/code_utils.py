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

    If entity is None, returns True if the imports list imports the module.
    If entity is not None, returns True if the imports list imports the entity from
    the module."""
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
    ) -> str:
    """Returns the name of the entity if the imports list imports the entity.

    The returned string provides the name of the entity as it is called in the code,
    accounting for import aliases. If the entity is None, the name of the module is
    returned as it is aliased in the code.

    If the entity or module is not found, None is returned."""
    test = importable_object.split('.')
    calls = []
    for imp in imports:
        imp_name = imp.name.split('.')
        n = min(len(imp_name), len(test))
        if test[:n] == imp_name[:n]:
            call = imp.asname
            if n < len(test):
                call += '.' + '.'.join(test[n:])
            calls.append(call)

    return calls


def cyclomatic_complexity(code: str) -> dict:
    """Compute the cyclomatic complexity of a piece of code."""
    v = ComplexityVisitor.from_code(code)
    output = {}
    for func in v.functions:
        output[func.name] = {
            'object': 'function',
            'complexity': func.complexity
        }

    for clss in v.classes:
        output[clss.name] = {
            'object': 'class',
            'complexity': clss.complexity,
            'methods': {}
        }
        for meth in clss.methods:
            output[clss.name]['methods'][meth.name] = {
                'complexity': meth.complexity
            }

    return output
