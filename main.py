from Matrix import Matrix
import copy
import random
from Solver import Jacobi_SLAE_Solver, SOR_SLAE_Solver
print('Практическое задание №6 Обухов Назар ПМИ-41')

def norma(x):
    return sum([x.data[0][i]**2 for i in range(x.rows)])**0.5

#Вариант 15
a, b = -0.497, 0.937
for n in [5]:
    x = Matrix(data=[[1]*n])
    A = Matrix(data=[[0]*n for i in range(n)])
    for j in range(n): #заполенение матрицы
        for i in range(n):
            if i != j:
                A.data[i][j] = 1+a*(i+1)+b*(j+1)
            else:
                A.data[i][j] = n
    f = A.multiply_matrix(x) #вектор свободных членов

    Aj = copy.deepcopy(A)
    fj = copy.deepcopy(f)
    xj = Matrix(data=[[random.randint(1, 10) for i in range(n)]])
    result = Jacobi_SLAE_Solver(Aj, fj, xj)
    result.print_matrix()

    """
    As = copy.deepcopy(A)
    fs = copy.deepcopy(f)
    xs = Matrix(data=[[random.randint(1, 20) for i in range(n)]])
    result = SOR_SLAE_Solver(As, fs, xs, 1.75)
    result.print_matrix()
"""
