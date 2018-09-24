import numpy as np

matA = [[1,2],[2,3]]
matB = [[4,5],[6,7]]

matC = np.outer(matA, matB)
print(matC)