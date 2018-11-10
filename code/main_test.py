import numpy as np
import numpy as np
import helper
# import submission
import matplotlib.pyplot as plt
import scipy.optimize
import scipy
import sympy as sp
a = sp.Symbol('a')
f1 = np.random.rand(3,3)
f2 = np.random.rand(3,3)


F = f1*a+f2*(1+a)
M = sp.Matrix(F)
print(M)

det = M.det()
print(det)


s = sp.solve(det,a)

print(s)




