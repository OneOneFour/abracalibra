# %% [markdown]
# ## Lorenz63 Benchmarks

# %%
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from lorenz import * 

# %%

from abracalibra.forward_model import ForwardModel
from torch.distributions import MultivariateNormal

true_params = {
    'sigma':10,
    'beta':8/3,
    'rho':28
}
T = 25
dt = 0.02
spinup = 20
x0 = torch.rand((50,3))
l63 = Lorenz63(**true_params)

r = l63(x0,T,dt)



# %%
l63_half = l63.half()
r_float= l63_half(x0.half(),T,dt)

l63_double = l63.double()
r_double= l63_half(x0.double(),T,dt)

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
with torch.no_grad():
    ax.plot(r[0,:,0],r[0,:,1],r[0,:,2],label='Single',alpha=0.5)
    ax.plot(r_double[0,:,0],r_double[0,:,1],r_double[0,:,2],label='Double',linestyle='--',alpha=0.5)
    ax.plot(r_float[0,:,0],r_float[0,:,1],r_float[0,:,2],label='Half',linestyle=':',alpha=0.5)
ax.legend()
plt.show()
# %%
l63avg = TimeAveragedFeatures(l63,spinup=spinup)
true_stats = l63avg(x0,T,dt)

true_mean_stats = true_stats.mean(dim=0)    
true_covar_stats = torch.cov(true_stats.T)

# %%
rprime = r - r.mean(dim=1,keepdim=True)

# %%
eigenvalues,eigenvectors = torch.linalg.eigh(rprime[0].T@rprime[0]/(T/dt))

# %%


eigenvalues

# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
with torch.no_grad():
    ax.plot(rprime[0,:,0],rprime[0,:,1],rprime[0,:,2],label='Single',alpha=0.5)
    ax.quiver(np.zeros(3),np.zeros(3),np.zeros(3),eigenvectors[0]*eigenvalues/10,eigenvectors[1]*eigenvalues/10,eigenvectors[2]*eigenvalues/10,color='r')
plt.show()
# %%
eigenvalues,eigenvectors = torch.linalg.eigh(rprime[0].T@rprime[0]/(T/dt))

# %%
principle_components = rprime[0]@eigenvectors

# %%
principle_components.shape

# %%
for i in range(3):
    plt.plot(torch.arange(0,T,dt),principle_components[:,i].detach())
