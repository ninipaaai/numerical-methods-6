from Matrix import Matrix
import copy
import math

#Метод Якоби
def Jacobi_SLAE_Solver(A, f, x0, max_iter=10000, eps=1e-12):
    M = A.cols
    if M==0:
        raise ValueError('Jacobi_SLAE_Solver: Matrix is empty')
    if A.cols != A.rows:
        raise ValueError('Jacobi_SLAE_Solver: Matrix must be square')

    # Проверка на нулевые диагональные элементы
    for i in range(M):
        if abs(A.data[i][i]) < 1e-15:
            raise ValueError(f'Jacobi_SLAE_Solver: Нулевой диагональный элемент в позиции {i}')

    x = Matrix(data=[[1]*A.rows])
    x_new = Matrix(data=[[1]*A.rows])
    Norm_Xnew_Xold = float('inf')
    iter = 0

    while iter < max_iter and Norm_Xnew_Xold > eps:
        Norm_Xnew_Xold = 0.0
        for i in range(M):
            F_Ax = f.data[0][i]
            for j in range(i):
                F_Ax -= A.data[j][i]*x.data[0][j]
            for j in range(i+1, M):
                F_Ax -= A.data[j][i]*x.data[0][j]

            x_new.data[0][i] = F_Ax/A.data[i][i]
            Norm_Xnew_Xold += (x.data[0][i]-x_new.data[0][i])**2
        x = copy.deepcopy(x_new) #полноценное копирование
        Norm_Xnew_Xold = math.sqrt(Norm_Xnew_Xold)
        if Norm_Xnew_Xold > 3: #Проверка на расходимость
            break
        iter += 1

    return x

#Метод релаксации
def SOR_SLAE_Solver(A, f, x0, w, max_iter=10000, eps=1e-12):
    M = A.cols
    if M==0:
        raise ValueError('SOR_SLAE_Solver: Matrix is empty')
    if A.cols != A.rows:
        raise ValueError('SOR_SLAE_Solver: Matrix must be square')

    x = Matrix(data=[[1] * A.rows])
    x_new = Matrix(data=[[1] * A.rows])
    Norm_Xnew_Xold = float('inf')
    iter = 0

    while iter < max_iter and Norm_Xnew_Xold > eps:
        Norm_Xnew_Xold = 0.0
        for i in range(M):
            F_Ax = f.data[0][i]
            for j in range(i):
                F_Ax -= A.data[j][i]*x_new.data[0][j]
            for j in range(i+1, M):
                F_Ax -= A.data[j][i]*x.data[0][j]

            F_Ax /= A.data[i][i]
            x_new.data[0][i] = (1.0-w)*x.data[0][i] + w*F_Ax
            Norm_Xnew_Xold += math.pow(x.data[0][i]-x_new.data[0][i], 2)

        x = copy.deepcopy(x_new)
        Norm_Xnew_Xold = math.sqrt(Norm_Xnew_Xold)
        if Norm_Xnew_Xold > 3:
            break
        iter += 1

    return x


