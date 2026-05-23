from Matrix import Matrix
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

    A.print_matrix()

