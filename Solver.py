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

    x = Matrix(data=[[1]*A.rows])
    x_new = Matrix(data=[[1]*A.rows])
    Norm_Xnew_Xold = 0.0
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
        iter += 1

    return x




