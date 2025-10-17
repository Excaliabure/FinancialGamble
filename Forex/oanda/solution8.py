import forex as fx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# from numba import njit
import scipy.optimize as opt
import numpy as np
import torch.nn as nn
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


_data = fx.min("EURUSD").data[0].to_numpy()
data = _data[:,2]


import torch
from torchdiffeq import odeint

def sim_torch(x, t_span=(0,28), dt=28, G=1.0, softening=0.05):
    """
    Differentiable N-body simulator in 3D.
    
    x: 1D tensor of shape (n_bodies * 7,)
       Format per body: [mass, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, ...]
    t_span: tuple (start_time, end_time)
    dt: number of time steps
    G: gravitational constant
    softening: softening to avoid singularities

    Returns:
        pos_sol: (n_bodies, 3, dt)
        distances: (n_bodies, dt) relative to first body
        masses: (n_bodies,) tensor of masses
    """
    n_bodies = x.shape[0] // 7

    # Extract masses, positions, velocities
    masses = torch.abs(x[::7])  # every 7th element starting at 0
    pos_list = [x[7*i + 1 : 7*i + 4] for i in range(n_bodies)]
    vel_list = [x[7*i + 4 : 7*i + 7] for i in range(n_bodies)]

    pos0 = torch.stack(pos_list, dim=0)  # (n_bodies, 3)
    vel0 = torch.stack(vel_list, dim=0)  # (n_bodies, 3)

    # Flatten for ODE solver
    y0 = torch.cat([pos0.flatten(), vel0.flatten()])

    t_eval = torch.linspace(t_span[0], t_span[1], dt)

    def n_body_odes(t, y):
        pos = y[:3*n_bodies].reshape(n_bodies, 3)
        vel = y[3*n_bodies:].reshape(n_bodies, 3)

        # Compute Newtonian accelerations
        acc = torch.zeros_like(pos)
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    r_vec = pos[j] - pos[i]
                    r = torch.norm(r_vec) + softening
                    acc[i] += G * masses[j] * r_vec / r**3

        dydt = torch.cat([vel.flatten(), acc.flatten()])
        return dydt

    sol = odeint(n_body_odes, y0, t_eval)  # (dt, 6*n_bodies)

    # Reshape positions and velocities
    pos_sol = sol[:, :3*n_bodies].reshape(dt, n_bodies, 3).permute(1, 2, 0)  # (n_bodies, 3, dt)
    centroid_pos = pos_sol[0]  # reference body
    distances = torch.norm(pos_sol - centroid_pos[None, :, :], dim=1)  # (n_bodies, dt)

    return pos_sol, distances, masses



softening = 0.05
G = 1.0

# -------------------------------
# PINN Acceleration
# -------------------------------
class PINN_Acceleration(nn.Module):
    def __init__(self, n_bodies, G=1.0, softening=0.05):
        super().__init__()
        self.n_bodies = n_bodies
        self.G = G
        self.softening = softening

        # Neural network to learn corrections to physics
        self.net = nn.Sequential(
            nn.Linear(n_bodies * 3, 64),
            nn.Tanh(),
            nn.Linear(64, n_bodies * 3)
        )

    def forward(self, pos, masses):
        """
        pos: (n_bodies, 3)
        masses: (n_bodies,)
        returns: (n_bodies, 3) accelerations
        """
        a_phys = torch.zeros_like(pos)

        # Newtonian gravity
        for i in range(self.n_bodies):
            for j in range(self.n_bodies):
                if i != j:
                    r_vec = pos[j] - pos[i]             # vector from i to j
                    r = torch.norm(r_vec) + self.softening  # distance with softening
                    a_phys[i] += self.G * masses[j] * r_vec / r**3

        # Neural network correction
        a_nn = self.net(pos.flatten()).reshape(self.n_bodies, 3)

        # Total acceleration
        return a_phys + a_nn

# -------------------------------
# ODE function for PINN
# -------------------------------
def pinn_nbody_odes(t, y, nn_model, n_bodies, masses):
    pos = y[:3*n_bodies].reshape(n_bodies, 3)
    vel = y[3*n_bodies:].reshape(n_bodies, 3)
    acc = nn_model(pos, masses)
    dydt = torch.cat([vel.flatten(), acc.flatten()])
    return dydt

# -------------------------------
# nn_odeint wrapper
# -------------------------------
def nn_odeint(x, nn_model, t_span=(0,28), dt=28):
    n_bodies = x.shape[0] // 7
    masses = torch.abs(x[::7])

    pos_list = [x[7*i+1:7*i+4] for i in range(n_bodies)]
    pos0 = torch.stack(pos_list, dim=0)
    vel_list = [x[7*i+4:7*i+7] for i in range(n_bodies)]
    vel0 = torch.stack(vel_list, dim=0)

    y0 = torch.cat([pos0.flatten(), vel0.flatten()])
    t_eval = torch.linspace(t_span[0], t_span[1], dt)

    sol = odeint(lambda t, y: pinn_nbody_odes(t, y, nn_model, n_bodies, masses),
                 y0, t_eval, method='dopri5')

    pos_sol = sol[:, :3*n_bodies].reshape(dt, n_bodies, 3).permute(1,2,0)
    vel_sol = sol[:, 3*n_bodies:].reshape(dt, n_bodies, 3).permute(1,2,0)
    distances = torch.norm(pos_sol - pos_sol[:, :, 0:1], dim=1)
    accelerations = nn_model(pos_sol[:, :, -1], masses)
    return pos_sol, vel_sol, distances, accelerations

# -------------------------------
# 3-body system: Center + 2 moving bodies
# -------------------------------
dt = 28
t_span = (0, dt)
n_bodies = 3  # Center + 2 moving

# Initial guess x: [mass, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z] * n_bodies
x = torch.zeros(n_bodies*7, dtype=torch.float32)

# Center body: fixed at origin, mass = 10
x[0] = 10.0  # mass
x[1:4] = 0.0  # pos
x[4:7] = 0.0  # vel

# Moving bodies: random initial guess
for i in range(1, 3):
    idx = i*7
    x[idx] = 1.0  # mass = 1
    x[idx+1:idx+4] = torch.randn(3) * 5.0  # random pos
    x[idx+4:idx+7] = torch.randn(3) * 0.1  # small random vel

# Reference distances (fake for demo, replace with actual if available)
reference_distances = torch.rand(n_bodies, dt)

x.requires_grad_()

# -------------------------------
# Train PINN
# -------------------------------
nn_model = PINN_Acceleration(n_bodies)
optimizer = torch.optim.Adam(list(nn_model.parameters()) + [x], lr=1e-2)
loss_fn = nn.MSELoss()
epochs = 5

for epoch in tqdm.tqdm(range(epochs)):
    optimizer.zero_grad()
    pos_sol, vel_sol, distances, accelerations = nn_odeint(x, nn_model, t_span, dt)
    loss = loss_fn(distances, reference_distances)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# -------------------------------
# Plot PINN
# -------------------------------
def plot_trajectories(pos_sol, title="Trajectories"):
    n_bodies = pos_sol.shape[0]
    dt = pos_sol.shape[2]
    colors = ['r','b','g','c','m','y']
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n_bodies):
        ax.plot(pos_sol[i,0,:].detach().numpy(),
                pos_sol[i,1,:].detach().numpy(),
                pos_sol[i,2,:].detach().numpy(),
                color=colors[i%6], label=f'Body {i}')
        ax.scatter(pos_sol[i,0,0].detach().numpy(),
                   pos_sol[i,1,0].detach().numpy(),
                   pos_sol[i,2,0].detach().numpy(),
                   color=colors[i%6], marker='o', s=50)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.title("Trajectory")
    plt.savefig("/mnt/c/Users/excal/trajectory.png")
    plt.close()

plot_trajectories(pos_sol, title="Trained PINN Trajectories")

# -------------------------------
# Plot actual physics solution
# -------------------------------
def nbody_odes_physics(t, y, n_bodies, masses):
    pos = y[:3*n_bodies].reshape(n_bodies, 3)
    vel = y[3*n_bodies:].reshape(n_bodies, 3)
    acc = torch.zeros_like(pos)
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                r_vec = pos[j] - pos[i]
                r2 = (r_vec**2).sum() + softening**2
                acc[i] += G * masses[j] * r_vec / (r2 * torch.sqrt(r2))
    dydt = torch.cat([vel.flatten(), acc.flatten()])
    return dydt

def odeint_physics(x, t_span=(0,28), dt=28):
    n_bodies = x.shape[0] // 7
    masses = torch.abs(x[::7])
    pos0 = torch.stack([x[7*i+1:7*i+4] for i in range(n_bodies)], dim=0)
    vel0 = torch.stack([x[7*i+4:7*i+7] for i in range(n_bodies)], dim=0)
    y0 = torch.cat([pos0.flatten(), vel0.flatten()])
    t_eval = torch.linspace(t_span[0], t_span[1], dt)

    sol = odeint(lambda t,y: nbody_odes_physics(t,y,n_bodies,masses),
                 y0, t_eval, method='dopri5')

    pos_sol = sol[:, :3*n_bodies].reshape(dt,n_bodies,3).permute(1,2,0)
    vel_sol = sol[:, 3*n_bodies:].reshape(dt,n_bodies,3).permute(1,2,0)
    return pos_sol, vel_sol



pos_actual, distances, masses = sim_torch(x.detach(), t_span, dt)
plot_trajectories(pos_actual, title="Actual Physics Trajectories")
