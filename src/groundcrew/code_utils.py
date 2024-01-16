import ast

from dataclasses import dataclass


@dataclass
class Import:
    name: str
    asname: str | None


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

    return imports


def get_imports_from_file(filepath: str) -> list[Import]:
    """Returns a list of Import objects for the given file."""
    with open(filepath, 'r') as f:
        code = f.read()
    return get_imports_from_code(code)


def get_imports_from_files(filepaths: list[str]) -> dict[str, Import]:
    """Returns a list of Import objects for the given files."""
    imports = {}
    for file in filepaths:
        imports[file] = get_imports_from_file(file)
    return imports


def imports_entity(imports: list[Import], module: str, entity: str) -> bool:
    """Returns True if the imports list imports the entity from the module."""
    for imp in imports:
        if imp.name.endswith(module):
            return True
        elif imp.name.endswith(f'{module}.{entity}'):
            return True

    return False


def import_called_as(imports: list[Import], module: str, entity: str) -> str:
    """Returns the name of the entity if the imports list imports the entity.

    The returned string provides the name of the entity as it is called in the code,
    accounting for import aliases.  If the entity is not found, None is returned."""
    for imp in imports:
        if imp.name.endswith(module):
            return f'{imp.asname}.{entity}'
        elif imp.name.endswith(f'{module}.{entity}'):
            return imp.asname

    return None
