import forex as fx
import matplotlib.pyplot as plt
import numpy as np
<<<<<<< Updated upstream
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
=======
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
import scipy.optimize as opt
import numpy as np
import torch.nn as nn
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




_data = fx.min("EURUSD").data[0].to_numpy()
data = _data[:,2]

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ========== Hyperparameters ===========
BODIES = 400         # number of bodies
DIMS = 2            # dimensions (2D)
G = 1.0             # gravitational constant
T0, TF = 0, 60
GRANULARITY = 120
PLOT = True


# ========== Initial Conditions ===========
masses = np.abs(np.random.randn(BODIES)) + 1  # positive random masses

# Random initial positions and velocities
r0 = np.random.randn(BODIES, DIMS) * 5
r0 = np.clip(r0, -1, 1)
v0 = np.random.randn(BODIES, DIMS)
v0 = np.clip(v0, -1, 1)

# --- Conserve momentum (so center of mass doesn’t drift) ---
total_mass = np.sum(masses)
v0 -= np.sum(v0 * masses[:, np.newaxis], axis=0) / total_mass

# Flatten state vector
y0 = np.hstack([r0.flatten(), v0.flatten()])

t_span = (T0, TF)
t_eval = np.linspace(T0, TF, GRANULARITY)


# ========== Acceleration and Derivative Functions ===========
def compute_accelerations_fast(y, masses, G=1.0, DIMS=2, epsilon=0.1):
    n = len(masses)
    pos = y[:n*DIMS].reshape(n, DIMS)

    # r_ij = r_j - r_i
    r = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # shape (n, n, DIMS)
    
    # squared distance + softening term
    dist_sq = np.sum(r**2, axis=2) + epsilon**2

    # 1 / (r^2 + eps^2)^(3/2)
    inv_dist3 = 1.0 / np.power(dist_sq, 1.5)

    # Zero out self-interaction
    np.fill_diagonal(inv_dist3, 0.0)

    # compute acceleration
    acc = G * np.sum(r * inv_dist3[:, :, np.newaxis] * masses[np.newaxis, :, np.newaxis], axis=1)
    return acc.reshape(-1)


def f(t, y):
    n = BODIES
    pos = y[:n*DIMS].reshape(n, DIMS)
    vel = y[n*DIMS:].reshape(n, DIMS)
    acc = compute_accelerations_fast(y, masses, G=G, DIMS=DIMS)
    dydt = np.hstack([vel.flatten(), acc.flatten()])
    return dydt


# ========== Objective Function (optional optimizer use) ===========
data = data[:GRANULARITY]

def objective_func(x, validate=False):
    soln = solve_ivp(
        f,
        t_span=t_span,
        y0=x,
        method='Radau',
        t_eval=t_eval,
        rtol=1e-4,
        atol=1e-6,
        max_step=1.0
    )
    

    arr = soln.y
    n = len(masses)
    pos = arr[:n*DIMS, :].reshape(n, DIMS, -1)

    # pairwise distances over time
    r = pos[:, np.newaxis, :, :] - pos[np.newaxis, :, :, :]
    dists = np.linalg.norm(r, axis=2)

    # mean_dist_over_time = np.mean(dists, axis=(0, 1))
    # min_len = min(mean_dist_over_time.shape[0], data.shape[0])
    # mse = np.mean((mean_dist_over_time[:min_len] - data[:min_len])**2)
    mse_matrix = []
    pair_indices = []
    for i in range(n):
        for j in range(n):
            if i != j:
                min_len = min(dists.shape[2], len(data))
                mse_ij = np.mean((dists[i, j, :min_len] - data[:min_len])**2)
                mse_matrix.append(mse_ij)
                pair_indices.append((i, j))

    mse_matrix = np.array(mse_matrix)
    best_mse_idx = np.argmin(mse_matrix)
    best_mse = mse_matrix[best_mse_idx]
    mse = best_mse

    print(f"Best MSE: {mse}, Best Pair: {pair_indices[best_mse_idx]}")

    if validate:
        return dists, mse, pos, pair_indices[best_mse_idx]

    return mse


# ========== Run Simulation ===========
# dists, mse, pos = objective_func(y0, validate=True)

# ========== Run Simulation ===========
soln = minimize(objective_func, y0, options={"maxiter" : 2})


# ========== Save IC ===========

np.save("IC.npy", soln.x)

print("Complete! Saved IC")
>>>>>>> Stashed changes

