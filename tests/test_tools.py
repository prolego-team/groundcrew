from groundcrew.tools import cyclomatic_complexity

def test_cyclomatic_complexity():
    code = (
        'def foo(x):\n'
        '    if x > 0:\n'
        '        return x\n'
        '    else:\n'
        '        return -x\n'
    )
    cc = cyclomatic_complexity(code)
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
    cc = cyclomatic_complexity(code)
    assert cc['Foo']['object'] == 'class'
    assert cc['Foo']['complexity'] == 3