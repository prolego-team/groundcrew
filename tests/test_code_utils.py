import groundcrew.code_utils as cu

TEST_CODE = """
import numpy as np
import torch
from torch.functional import sigmoid
from torch.nn import Module, Linear
from ..foo.bar import a as b, c

def dummy(y):
    y_t = torch.tensor(y)
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
    assert len(imports) == 7
    assert imports[0]==cu.Import(
        name='numpy',
        asname='np',
    )
    assert imports[1]==cu.Import(
        name='torch',
        asname='torch',
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


def test_import_entity():
    imports = cu.get_imports_from_code(TEST_CODE)
    assert cu.imports_entity(imports, 'numpy', 'random')
    assert cu.imports_entity(imports, 'numpy', 'random.rand')
    assert cu.imports_entity(imports, 'torch', 'tensor')
    assert cu.imports_entity(imports, 'torch.functional', 'sigmoid')
    assert cu.imports_entity(imports, 'functional', 'sigmoid')
    assert cu.imports_entity(imports, 'torch.nn', 'Module')
    assert cu.imports_entity(imports, 'torch.nn', 'Linear')
    assert cu.imports_entity(imports, 'foo.bar', 'a')
    assert not cu.imports_entity(imports, 'foo.bar', 'b')
    assert cu.imports_entity(imports, 'foo.bar', 'c')

def test_import_called_as():
    imports = cu.get_imports_from_code(TEST_CODE)
    assert cu.import_called_as(imports, 'numpy', 'random') == 'np.random'
    assert cu.import_called_as(imports, 'numpy', 'random.rand') == 'np.random.rand'
    assert cu.import_called_as(imports, 'torch', 'tensor') == 'torch.tensor'
    assert cu.import_called_as(imports, 'torch.functional', 'sigmoid') == 'sigmoid'
    assert cu.import_called_as(imports, 'torch.nn', 'Module') == 'Module'
    assert cu.import_called_as(imports, 'torch.nn', 'Linear') == 'Linear'
    assert cu.import_called_as(imports, 'foo.bar', 'a') == 'b'
    assert cu.import_called_as(imports, 'foo.bar', 'b') is None
    assert cu.import_called_as(imports, 'foo.bar', 'c') == 'c'
