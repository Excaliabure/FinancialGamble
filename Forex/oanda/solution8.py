import numpy as np
# from scipy.integrate import odeint
# import torch.nn as nn


n = 2 # bodies 

G = 1
m = np.array([100, 1])
x = np.array([[1,2], # x1
     [2,1] # x2
     ])

a = np.zeros(n)

print("Calculating A..")
for i in range(n):
    for j in range(n):
        if j != i:
            a[i] = G * sum(m[i] * (x[j] - x[i]) / np.pow(np.linalg.norm(x[j] - x[i]),3))


print(a)
