from Matrix import Matrix
import copy
import random
from Solver import Jacobi_SLAE_Solver, SOR_SLAE_Solver

print('Практическое задание №6 Обухов Назар ПМИ-41')


def norma(x):
    return sum([x.data[0][i] ** 2 for i in range(x.rows)]) ** 0.5

def best_omega(A, F, x0):
    mass = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    x = Matrix(data=[[1] * n])
    delta_min = 1e10
    omega = 1.0
    for w in mass:
        result = SOR_SLAE_Solver(A, f, x, w)
        r = Matrix(data=result.data)
        for i in range(n):
            r.data[0][i] = x.data[0][i] - result.data[0][i]
        delta = norma(r) / norma(x)
        if delta < delta_min:
            delta_min = delta
            omega = w
    print('omega:', omega)
    return delta_min


# Вариант 15
a, b = -0.497, 0.937

for n in [500, 1000, 2000]:
    x = Matrix(data=[[1] * n])
    A = Matrix(data=[[0] * n for i in range(n)])

    for j in range(n):  # j — столбец
        for i in range(n):  # i — строка
            if i != j:
                A.data[j][i] = 1 + a * (i + 1) + b * (j + 1)
            else:
                A.data[j][i] = n

    f = A.multiply_matrix(x)  # вектор свободных членов

    print(f'Порядок {n}')

    #Якоби
    print('Метод Якоби')
    Aj = copy.deepcopy(A)
    fj = copy.deepcopy(f)
    xj = Matrix(data=[[random.uniform(0.9, 1.1) for i in range(n)]])
    result = Jacobi_SLAE_Solver(Aj, fj, xj)
    r = Matrix(data=result.data)
    for i in range(n):
        r.data[0][i] = x.data[0][i] - result.data[0][i]
    delta = norma(r)/norma(x)
    print(f'Погрешность {delta}')

    # SOR
    print('Метод релаксации')
    As = copy.deepcopy(A)
    fs = copy.deepcopy(f)
    xs = Matrix(data=[[random.uniform(0.9, 1.1) for i in range(n)]])
    delta = best_omega(As, fs, xs)
    print(f'Погрешность {delta}')
    print()