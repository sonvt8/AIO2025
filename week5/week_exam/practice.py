import numpy as np
from tools import ClassicTool, QuickTool

classicTool = ClassicTool()
quickTool = QuickTool()

# câu 1
print("Kết quả câu 1 là:")
vector = np.array([-2, 4, 9, 21])
assert classicTool.vec_len(vector) == quickTool.vec_len(vector)
print(quickTool.vec_len(vector))


# câu 2
print("*"*25, "\nKết quả câu 2 là:")
v1 = np.array([0, 1, -1, 2])
v2 = np.array([2, 5, 1, 0])
assert np.array_equal(
    classicTool.dot_product(v1, v2),
    quickTool.dot_product(v1, v2)
)
print(quickTool.dot_product(v1, v2))

# câu 3
print("*"*25, "\nKết quả câu 3 là:")
matrix = np.array([[-1, 1, 1], [0, -4, 9]])
vector = np.array([0, 2, 1])
assert np.array_equal(
    classicTool.multi_vector(matrix, vector),
    quickTool.dot_product(matrix, vector)
)
print(quickTool.dot_product(matrix, vector))

# câu 4
print("*"*25, "\nKết quả câu 4 là:")
m1 = np.array([[0, 1, 2], [2, -3, 1]])
m2 = np.array([[1, -3], [6, 1], [0, -1]])
assert np.array_equal(
    classicTool.matrix_multi_matrix(m1, m2),
    quickTool.dot_product(m1, m2)
)
print(quickTool.dot_product(m1, m2))

# câu 5
print("*"*25, "\nKết quả câu 5 là:")
x = np.array([1, 2, 3, 4])
y = np.array([1, 0, 3, 0])

print(round(quickTool.cosine(x, y), 3))

# câu 6
print("*"*25, "\nKết quả câu 6 là:")
arr = np.arange(0, 10)

print(arr[arr % 2 == 1])

# câu 7
print("*"*25, "\nKết quả câu 6 là:")
arr = np.arange(0, 10)

print(arr[arr % 2 == 1])

# câu 8
print("*"*25, "\nKết quả câu 8 là:")
arr = np.arange(0, 10)

print(np.where(arr % 2 == 1, -1, arr))

# câu 9
print("*"*25, "\nKết quả câu 9 là:")
arr1 = np.arange(10).reshape(2, -1)
arr2 = np.ones((2, 5), dtype=int)

print(np.concatenate((arr1, arr2), axis=0))

# câu 10
print("*"*25, "\nKết quả câu 10 là:")
arr1 = np.arange(10).reshape(2, -1)
arr2 = np.ones((2, 5), dtype=int)

print(np.concatenate((arr1, arr2), axis=1))

# câu 11
print("*"*25, "\nKết quả câu 11 là:")
a = np.array([2, 6, 1, 9, 10, 3, 27])
index = np.where((a >= 5) & (a <= 10))[0]
print(a[index])
print(a[(a >= 5) & (a <= 10)])
