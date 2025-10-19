import numpy as np
from scipy.integrate import solve_ivp
# from scipy.integrate import odeint
# import torch.nn as nn


n = 2 # bodies 

G = 1

# Mass1, x1, y1, z1, Mass2, x2, y2, z2, ...

# Note only doing positions
ic = np.array([
     ]) 
     

def f(t, y, dims=3):
    # Central Body at 0,0,0 and mass 100 
    
    
    pos = y[:n*dims].reshape(n, dims)
    vel  = y[:n*dims].reshape(n, dims)
    acc = np.zeros(vel.shape[0])
    print(vel.shape)
    a = np.zeros(len(masses))

    for i in range(len(masses)):
        for j in range(len(masses)):
            if j != i:
                r = pos[j] - pos[i]
                acc[i] =  (masses[i] *  r / np.pow(np.linalg.norm(r), 3)).sum()
                
    
    return a


t_span = (0,20)
dt = 20

# print(solve_ivp(f,t_span=t_span, y0=ic))
print(f(1,ic))

