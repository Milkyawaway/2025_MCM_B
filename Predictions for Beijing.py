import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data_J = pd.read_excel("./data_Juneau.xlsx")
data_B = pd.read_excel("./data_Beijing.xlsx")

r = 0.06 # Original tax rate, obtained through inquiry

# Calculate coefficients, delta
paras = np.empty((7,2))
paras[:,0] = data_B["I"]/(1+r)
paras[:,1] = data_B["AQI"]
paras

x, residuals, rank, s = np.linalg.lstsq(paras, data_B["dE"], rcond=None)

# Calculate parameter ranges

I_min = 0; I_max = 32853.7
r_min = 0; r_max = 1
k_min = 0; k_max = 1
mu = 0.00387671
dalta = -0.20875312
g = -8.23106*mu

def bar(x, max_val, min_val):
    return (x - min_val) / (max_val - min_val)

def B(I, r):
    return I * r / (r + 1)

def E(I, r, k, g, mu, dalta):
    return -(mu + g * k * r) * I / dalta / (1 + r)

def G(I, r):
    return I - B(I, r)

# Calculate the range of E
I_array = np.linspace(I_min, I_max, num=1000)
r_array = np.linspace(r_min, r_max, num=1000)
k_array = np.linspace(k_min, k_max, num=1000)
E_matrix = np.empty((1000,1000,1000))

for ind in range(len(I_array)):
    if ind % 100 == 0:
        print(ind)
    for rnd in range(len(r_array)):
        
        for knd in range(len(k_array)):
            E_matrix[ind][rnd][knd] = E(I_array[ind], r_array[rnd], k_array[knd], g, mu, dalta)

E_max = np.max(E_matrix)
E_min = np.min(E_matrix)

# Calculate B, G ranges
B_min = B(I_min, r_min) 
B_max = B(I_max, I_max)

G_min = G(I_min, r_min)
G_max = G(I_max, r_max)

def S(E, G, B, k, E_max=E_max, G_max=G_max, E_min=E_min, G_min=G_min):
    a = 0.24 / 1.2
    b = 0.4 / 1.2
    c = 0.56 / 1.2
    return a * bar(E, E_max, E_min) + b * bar(G, G_max, G_min) + c * (1 - k) * B

# Calculate S range
S_max = S(1, 1, 1, 0)
S_min = S(0, 0, 0, 1)

# Define the objective function: Weighted sum method
def objective(x):
    I, r, k = x
    e = E(I, r, k, g, mu, dalta)
    g_val = G(I, r)
    # Objective: Maximize the weighted sum of E and G
    return -(0.5 * bar(e, E_max, E_min) + 0.5 * bar(g_val, G_max, G_min))  # Negative sign indicates maximization
    #return -(0.5 * e + 0.5 * g_val)
# Define constraint: S >= 0.5
def constraint(x):
    I, r, k = x
    b_val = B(I, r)
    e = E(I, r, k, g, mu, dalta)
    g_val = G(I, r)
    return S(e, g_val, b_val, k) - 0.5
    #return S(bar(e, E_max, E_min), bar(g_val, G_max, G_min), bar(b_val, B_max, B_min), bar(k, k_max, k_min)) - 0.5
    
# Initial guess values
x0 = [I_max/19.14, r_max/514, k_max/0.2]

# Bounds for variables
bounds = [(0, I_max), (0, r_max), (0, k_max)]

# Define constraint
con = {'type': 'ineq', 'fun': constraint}

# Use optimization algorithm to find the solution
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=con)

# Output results
I_opt, r_opt, k_opt = solution.x
e_opt = E(I_opt, r_opt, k_opt, g, mu, dalta)
g_opt = G(I_opt, r_opt)
s_opt = S(e_opt, g_opt, B(I_opt, r_opt), k_opt)

print(f"Optimal I: {I_opt}")
print(f"Optimal r: {r_opt}")
print(f"Optimal k: {k_opt}")
print(f"Optimal E: {e_opt}")
print(f"Optimal S: {s_opt}")
