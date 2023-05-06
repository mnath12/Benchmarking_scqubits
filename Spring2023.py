# ## Writing Functions to Facilitate Benchmarking
import scqubits as scq
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from time import time
import time as tm
from tqdm.notebook import tqdm
import primme as pr


def scipy_option_tester(qubit, dims, N):
    # qubit: sparse qubit to be tested for SA vs LM
    # dims: numpy array 
    # N: averaging
    times_ratio = np.zeros_like(dims, dtype=float)
    times_1 = np.zeros_like(dims, dtype=float)
    times_2 = np.zeros_like(dims, dtype=float)
    for i, dim in enumerate(dims):
        qubit.ncut = dims
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
        times_ratio[i] = times_2[i]/times_1[i]
        print(f"Time ratio (SA/LM) {times[i]:.2f} for {dim:.0f}x{dim:.0f}")
    return times_ratio

def generate_plot(times, dims, x_label, y_label, title, file_name):
    plt.plot((dims*2+1)**3, times, 'o-') # How does this work for each qubit? Dims vs grid pt ct...?
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(file_name)
    plt.show()
    return None

def primme_benchmarker(qubit, shape, N, energies, ncuts):
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
    times_ratio = times_a / times_b        
    return times_ratio         


# Generates a heatmap of PRIMME performance given an appropriate matrix
def generate_heatmap(matrix, ncuts, energies, title, xlabel, ylabel, heatLabel, filename):
    im = plt.imshow(matrix, extent = [ncuts.min(), ncuts.max(), energies.min(), energies.max()], aspect = 10)
    plt.title(title)
    plt.xlabel("")
    plt.colorbar(im, label = heatLabel)
    plt.xlabel(xlabel, size = 20)
    plt.ylabel(ylabel, size = 20)
    plt.savefig(filename)
    return None
