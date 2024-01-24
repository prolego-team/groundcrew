import groundcrew.code_utils as cu

TEST_CODE = """
import numpy as np
import torch_fake
from torch.functional import sigmoid
from torch.nn import Module, Linear
from foo.bar import a as b, c
import baz.utils as bu
import baz

def dummy(y):
    y_t = torch_fake.tensor(y)
    z = sigmoid(y_t)
    return z + np.random.rand()

class Network(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

"""

def test_get_imports():
    imports = cu.get_imports_from_code(TEST_CODE)
    assert len(imports) == 9
    assert imports[0]==cu.Import(
        name='numpy',
        asname='np',
    )
    assert imports[1]==cu.Import(
        name='torch_fake',
        asname='torch_fake',
    )
    assert imports[2]==cu.Import(
        name='torch.functional.sigmoid',
        asname='sigmoid',
    )
    assert imports[3]==cu.Import(
        name='torch.nn.Module',
        asname='Module',
    )
    assert imports[4]==cu.Import(
        name='torch.nn.Linear',
        asname='Linear',
    )
    assert imports[5]==cu.Import(
        name='foo.bar.a',
        asname='b',
    )
    assert imports[6]==cu.Import(
        name='foo.bar.c',
        asname='c',
    )
    assert imports[7]==cu.Import(
        name='baz.utils',
        asname='bu',
    )
    assert imports[8]==cu.Import(
        name='baz',
        asname='baz',
    )


def test_import_module_entity():
    imports = cu.get_imports_from_code(TEST_CODE)
    assert cu.imports_entity(imports, 'numpy')
    assert cu.imports_entity(imports, 'numpy', 'random')
    assert cu.imports_entity(imports, 'numpy', 'random.rand')
    assert cu.imports_entity(imports, 'numpy.random', 'rand')
    # assert cu.uses_module(imports, 'numpy')
    # assert not cu.uses_module(imports, 'numpy.random')

    assert cu.imports_entity(imports, 'torch_fake')
    assert cu.imports_entity(imports, 'torch_fake.tensor')
    assert cu.imports_entity(imports, 'torch_fake.functional.sigmoid')
    assert cu.imports_entity(imports, 'torch.functional.sigmoid')
    assert not cu.imports_entity(imports, 'torch.functional.softmax')
    assert not cu.imports_entity(imports, 'functional.sigmoid')
    assert cu.imports_entity(imports, 'torch.nn.Module')
    assert cu.imports_entity(imports, 'torch.nn.Linear')
    assert cu.imports_entity(imports, 'foo.bar.a')
    assert not cu.imports_entity(imports, 'foo.bar', 'b')
    assert cu.imports_entity(imports, 'foo.bar.c')
    assert cu.imports_entity(imports, 'baz.utils')


def test_import_called_as():
    imports = cu.get_imports_from_code(TEST_CODE)
    assert cu.import_called_as(imports, 'numpy') == ['np']
    assert cu.import_called_as(imports, 'numpy.random') == ['np.random']
    assert cu.import_called_as(imports, 'numpy.random.rand') == ['np.random.rand']
    assert cu.import_called_as(imports, 'torch_fake.tensor') == ['torch_fake.tensor']
    assert cu.import_called_as(imports, 'torch.functional.sigmoid') == ['sigmoid']
    assert cu.import_called_as(imports, 'torch.functional.softmax') == []
    assert cu.import_called_as(imports, 'functional.sigmoid') == []
    assert cu.import_called_as(imports, 'torch.nn.Module') == ['Module']
    assert cu.import_called_as(imports, 'torch.nn.Linear') == ['Linear']
    assert cu.import_called_as(imports, 'foo.bar.a') == ['b']
    assert cu.import_called_as(imports, 'foo.bar.b') == []
    assert cu.import_called_as(imports, 'foo.bar.c') == ['c']
    assert cu.import_called_as(imports, 'baz.utils') == ['bu', 'baz.utils']
    assert cu.import_called_as(imports, 'baz.utils.some_func') == ['bu.some_func', 'baz.utils.some_func']


def test_cyclomatic_complexity():
    code = (
        'def foo(x):\n'
        '    if x > 0:\n'
        '        return x\n'
        '    else:\n'
        '        return -x\n'
    )
    cc = cu.cyclomatic_complexity(code)
    assert cc['foo']['object'] == 'function'
    assert cc['foo']['complexity'] == 2

    code = (
        'class Foo:\n'
        '    def __init__(self):\n'
        '        self.x = 0\n'
        '    def bar(self):\n'
        '        if self.x > 0:\n'
        '            return self.x\n'
        '        else:\n'
        '            return -self.x\n'
    )
    cc = cu.cyclomatic_complexity(code)
    assert cc['Foo']['object'] == 'class'
    assert cc['Foo']['complexity'] == 3
