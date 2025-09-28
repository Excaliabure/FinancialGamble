# %%time
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3d plots
"""
f = lambda x: 0.990/x
"""

def sim(bank=1000, steps=1000, risk=0.01, multiplier=1.01):
    rng = np.random.default_rng()
    prob = 0.99 / multiplier
    
    b = bank
    y = np.zeros(steps)
    x = np.arange(0,steps)
    
    for i in range(steps):
        
        bet = bank * risk
        bank -= bet
        if rng.random() < prob:
            bank += bet * multiplier

        else:
            bank -= bet
        y[i] = bank
    m,b = np.polyfit(x,y, 1)
    return m


BANK = 1000
STEPS = 100
RISKS = np.linspace(0.01, 1, num=100) # 1% Risk
MULTIPLIERS = np.linspace(1.01,100,num=100)
SIM_AMT = 100

    
def gain(risk,mult, amt=1000,bank=1000):
    # slope(slope(amt * sim(risk,mult)))

    y = np.zeros(amt)
    for i in range(amt):
        y[i] = sim(bank=bank,steps=amt,risk=risk,multiplier=mult)
    
    m = y.sum()

    return m

x = np.arange(0,SIM_AMT)
z = np.arange(0,SIM_AMT)
y = np.zeros((SIM_AMT,SIM_AMT))

for i in tqdm(range(len(MULTIPLIERS))):
    for j in range(len(RISKS)):
        y[i,j] = sim(risk=RISKS[j],multiplier=MULTIPLIERS[i])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, z, y, cmap='viridis')

ax.set_xlabel('X axis')
ax.set_ylabel('Z axis')
ax.set_zlabel('Y value')

plt.title('3D Surface Plot')
plt.show()