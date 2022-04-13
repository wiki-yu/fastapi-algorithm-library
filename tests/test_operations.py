
from operationss.operations import multiply,square_root,divide


def test_multiply ():
    a,b=3,2
    val=multiply(a,b)
    assert val==a*b

def test_square_root():
    a=4
    val=square_root(a)
    assert val==a**0.5

def test_divide():
    a,b=10,5
    val=divide(a,b)
    assert val==a/b if a>b else b/a