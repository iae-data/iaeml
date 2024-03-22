# example.pyx

# Definindo uma função em Cython
cpdef int add(int a, int b):
    """
    Calcula a soma de dois números inteiros.

    Parâmetros:
    a (int): Primeiro número.
    b (int): Segundo número.

    Retorna:
    int: A soma de a e b.
    """
    return a + b
