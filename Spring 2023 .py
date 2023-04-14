#!/usr/bin/env python
# coding: utf-8

# ## Writing Functions to Facilitate Benchmarking

# In[1]:


import scqubits as scq
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from time import time
import time as tm
from tqdm.notebook import tqdm


# In[2]:


import primme as pr


# In[3]:


ZeroPi = scq.ZeroPi.create()


# In[4]:


full = scq.FullZeroPi.create()


# In[5]:


def scipy_option_tester(qubit, dims, N):
    # qubit: sparse qubit to be tested for SA vs LM
    # dims: numpy array 
    # N: averaging
    times = np.zeros_like(dims, dtype=float)
    times_1 = np.zeros_like(dims, dtype=float)
    times_2 = np.zeros_like(dims, dtype=float)
    for i, dim in enumerate(dims):
        hamiltonian = qubit.hamiltonian()
        start_time_1 = time()
        for j in range(N):
            sp.sparse.linalg.eigsh(hamiltonian, k=6, sigma = 0.0, which="LM")
        end_time_1 = time()
        times_1[i] = (end_time_1-start_time_1)/N
        start_time_2 = time()
        for value in range(N):
            sp.sparse.linalg.eigsh(hamiltonian, k=6, which="SA")
        end_time_2 = time()
        times_2[i] = (end_time_2-start_time_2)/N
        times[i] = times_2[i]/times_1[i]
        print(f"Time ratio (SA/LM) {times[i]:.2f} for {dim:.0f}x{dim:.0f}")
    return times


# In[6]:


dims = np.array([10,20,30,40,50,60,70])
full = scipy_option_tester(ZeroPi, dims, 1)


# In[ ]:





# In[ ]:


def generate_plot(times, dims):
    plt.plot((dims*2+1)**3, times, 'o-')
    plt.xlabel("Hilbert Space Dimension")
    plt.ylabel("Ratio of Time of SA to Time of LM")
    plt.title("Hilbert Dimension vs Time Ratio for SA and LM")
    plt.savefig("benchmark SA vs LM.png")
    plt.show()
    return None


# In[ ]:


plt.plot((dims*2+1)**3, times, 'o-')
plt.xlabel("Hilbert Space Dimension")
plt.ylabel("Ratio of Time of SA to Time of LM")
plt.title("Hilbert Dimension vs Time Ratio for SA and LM (Transmon)")
plt.savefig("Tmon benchmark SA vs LM.png")
plt.show()


# In[6]:


def primme_heat_map(qubit, shape, N, energies, ncuts):
    # shape: tuple with size of matrix (m,n)
    # N: averaging
    times_a = np.empty(shape = shape)
    times_b = np.empty(shape = shape)
    for index_1, energy in tqdm(enumerate(energies)):
        qubit.EJ = energy
        for index_2, dim in enumerate(ncuts):
            qubit.ncut = dim
            hamiltonian = qubit.hamiltonian()
            start_time_1 = time()
            for _ in range(N):
                result = pr.eigsh(hamiltonian, k=6, sigma = 0.0, which="LM")
            end_time_1 = time()
            times_a[index_1][index_2] = (end_time_1-start_time_1)/N
            start_time = time()
            for _ in range(N):
                result = sp.sparse.linalg.eigsh(hamiltonian, k=6, sigma = 0.0, which="LM")
            end_time = time()
            times_b[index_1][index_2] = (end_time-start_time)/N
    big_times = times_a / times_b        
    return big_times


# In[6]:


energies = energies = np.linspace(13.5, 16.5, 10)
ncuts = ncuts = np.array([20,30,40,50,60])
Zero_pi_heatmap = primme_heat_map(ZeroPi, (10, 5), 1, energies, ncuts)
np.save("heat_map_matrix", Zero_pi_heatmap)


# In[7]:


A = np.load("heat_map_matrix.npy")
A = np.load("heat_map_matrix.npy")
im = plt.imshow(A, extent = [ncuts.min(), ncuts.max(), energies.min(), energies.max()], aspect = 10)
plt.title(r"Heat Map with Hamiltonian Dim, $E_J$, and Relative Time")
plt.xlabel("")
plt.colorbar(im, label = "Ratio of PRIMME and Eigsh Time")
plt.xlabel(r"$n_{cut}$", size = 20)
plt.ylabel(r"$E_{J}$ (GHz)", size = 20)
plt.savefig('heat_map_primme_0pi.png')


# In[7]:


energies = np.linspace(.125, .75, 10)
ncuts = ncuts = np.array([20,30,40,50,60])


# In[8]:


Full_Zero_pi_heatmap = primme_heat_map(full, (10, 5), 1, energies, ncuts)
np.save("full_heat_map_matrix", Full_Zero_pi_heatmap)


# In[9]:


A = np.load("full_heat_map_matrix.npy")
im = plt.imshow(A, extent = [ncuts.min(), ncuts.max(), energies.min(), energies.max()], aspect = 10)
plt.title(r"Heat Map with Hamiltonian Dim, $E_J$, and Relative Time")
plt.xlabel("")
plt.colorbar(im, label = "Ratio of PRIMME and Eigsh Time")
plt.xlabel(r"$n_{cut}$", size = 20)
plt.ylabel(r"$E_{J}$ (GHz)", size = 20)
plt.savefig('heat_map_primme_full.png')


# In[ ]:




