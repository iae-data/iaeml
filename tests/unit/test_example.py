import pytest
from iaeml.example import add


def test_add_positive_numbers():
    assert add(1, 2) == 3, "Deveria retornar 3 para a soma de 1 e 2"

def test_add_negative_numbers():
    assert add(-1, -2) == -3, "Deveria retornar -3 para a soma de -1 e -2"

def test_add_mixed_numbers():
    assert add(-1, 2) == 1, "Deveria retornar 1 para a soma de -1 e 2"

def test_add_zero():
    assert add(0, 0) == 0, "Deveria retornar 0 para a soma de 0 e 0"