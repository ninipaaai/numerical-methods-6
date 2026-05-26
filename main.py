from Matrix import Matrix
import copy
import random
import time
from Solver import Jacobi_SLAE_Solver, SOR_SLAE_Solver

print('Практическое задание №6 Обухов Назар ПМИ-41')

def norma(x):
    return sum([x.data[0][i] ** 2 for i in range(x.rows)]) ** 0.5

def Gershgorin(A, n): #оценивает наибольщий радиус круга Гершгорина
    if A.cols == 0:
        raise ValueError('Gershgorin: the matrix is empty')
    if A.rows != A.cols:
        raise ValueError('Gershgorin: the matrix is not square')

    summ = []
    for i in range(A.rows):
        s = 0
        for j in range(A.cols):
            if i != j:
                s += abs(A.data[j][i])
        summ.append(s)
    return (max(summ), n)

# Вариант 15
a, b = -0.497, 0.937
challenge = 10 #количество замеров времени

for n in [500, 1000, 2000]:
    x = Matrix(data=[[1] * n]) #точное решение
    A = Matrix(data=[[0] * n for i in range(n)])

    for i in range(n):
        for j in range(n):
            if i != j:
                A.data[j][i] = a * (i + 1) + b * (j + 1)
            else:
                A.data[j][i] = n

    f = A.multiply_matrix(x)  # вектор свободных членов

    print(f'Порядок {n}')

    print(f"Радиус наибольшего круга Гершгорина: {Gershgorin(A, n)[0]}")

    #Якоби
    print('Метод Якоби')
    Aj = copy.deepcopy(A)
    fj = copy.deepcopy(f)
    xj = Matrix(data=[[random.randint(1, 5) for i in range(n)]])
    time1 = []
    for i in range(challenge):
        start = time.time()
        result = Jacobi_SLAE_Solver(Aj, fj, xj)
        end = time.time()
        time1.append(end - start)
    print(f"Среднее время работы {sum(time1)/challenge}")
    r = Matrix(data=result.data)
    for i in range(n):
        r.data[0][i] = x.data[0][i] - result.data[0][i]
    delta = norma(r)/norma(x)
    print(f'Погрешность {delta}')

    # SOR
    print('Метод релаксации')
    As = copy.deepcopy(A)
    fs = copy.deepcopy(f)
    xs = Matrix(data=[[random.randint(1, 5) for i in range(n)]])
    time2 = []
    for i in range(challenge):
        start = time.time()
        result = SOR_SLAE_Solver(As, fs, xs, 0.01)
        end = time.time()
        time2.append(end - start)
    print(f"Среднее время работы {sum(time2)/challenge}")
    r = Matrix(data=result.data)
    for i in range(n):
        r.data[0][i] = x.data[0][i] - result.data[0][i]
    delta = norma(r)/norma(x)
    print(f'Погрешность {delta}')
    print()