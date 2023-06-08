# # # Writing Functions to Facilitate Benchmarking
import scqubits as scq
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from time import time
import time as tm
from tqdm.notebook import tqdm
import primme as pr
from itertools import product



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
        print(f"Time ratio (SA/LM) {times_ratio[i]:.2f} for {dim:.0f}x{dim:.0f}")
    return times_ratio

def generate_plot(times, dims, x_label, y_label, title, file_name):
    plt.plot((dims*2+1)**3, times, 'o-') # How does this work for each qubit? Dims vs grid pt ct...?
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(file_name)
    plt.show()
    return None

# def primme_benchmarker(qubit, shape, avg_num, param_1, a, b, param_2, c, d, file_name):
    # shape: tuple with size of matrix (m,n)
    # avg_num: number of iterations for averaging
    # param_1: first parameter to be varied
    # a: cutoff for p1
    # b: cutoff for p1
    # param_2:
    # c: cutoff
    # d: cutoff
    # file_name:
    times_a = np.empty(shape = shape)
    times_b = np.empty(shape = shape)
    param_1_list = np.linspace(a, b, shape[0])
    param_2_list = np.linspace(c, d, shape[1])
    for index_1, energy in tqdm(enumerate(param_1_list)):
        qubit.__dict__[param_1] = energy
        for index_2, p in enumerate(param_2):
            qubit.__dict__[param_2] = p
            #for ncut in qubit.cutoff_names:
                #qubit.__dict__["_" + ncut] = dim
            hamiltonian = qubit.hamiltonian
            start_time_1 = time()
            for _ in range(avg_num):
                result = pr.eigsh(hamiltonian, k=6, sigma = 0.0, which="LM")
            end_time_1 = time()
            times_a[index_1][index_2] = (end_time_1-start_time_1)/avg_num
            start_time = time()
            for _ in range(avg_num):
                result = sp.sparse.linalg.eigsh(hamiltonian, k=6, sigma = 0.0, which="LM")
            end_time = time()
            times_b[index_1][index_2] = (end_time - start_time)/avg_num
    times_ratio = times_a / times_b    
    np.save(file_name, times_ratio)    
    return times_ratio         

# def primme_benchmarker(qubit, cutoff_list, param_list, avg_num):
    ''' qubit: a qubit class instance
        cutoff_list: list of tuples {(cutoff_name, cutoff_array), ...} 
        param_list: list of tuples tuple {(param_name, param_vals_array), ... } 
        Note: the param_vals and cutoff arrays can be created using np.linspace
        Based on these parameters, primme_benchmarker will return a multi-dimesnional numpy array
        containing benchmark results for primme vs eigsh'''
    arr_dim = len(cutoff_list) + len(param_list)
    # Create appropriate number of empty lists to initialize arrays, using a dictionary 
    big_list = {}
    # for cutoff_tuple in cutoff_list:
    #   big_list.append(cutoff_tuple[1])
    
    for param_tuple in param_list:
        big_list.append(param_tuple[1])

    big_array = np.meshgrid(*big_list)
    # big_array[0,1,2,etc] = arrays corresponding to different param vals
    result = np.zeros(big_array.shape)

    for index, x in np.ndenumerate(big_array):
        param_name = 0
        setattr(qubit, param_name, param_val)

        result[index] = 0
   
    """ for i, cutoff_tuple in enumerate(cutoff_list):
        cutoff_name = cutoff_tuple[0]
        cutoff_array = cutoff_tuple[1]
        for cutoff in cutoff_array:
            setattr(qubit, cutoff_name, cutoff)
            for param_tuple in param_list:
                param_name = param_tuple[0]
                param_vals_array = param_tuple[1]
                for param_val in param_vals_array:
                    setattr(qubit, param_name, param_val)
                    hamiltonian = qubit.hamiltonian
                    start_time = time()
                    for _ in range(avg_num):
                        result = pr.eigsh(hamiltonian, k=6, sigma = 0.0, which="LM")
                    end_time = time()
                    # times_a[index_1][index_2] = (end_time-start_time)/avg_num
                    start_time_1 = time()
                    for _ in range(avg_num):
                        result = sp.sparse.linalg.eigsh(hamiltonian, k=6, sigma = 0.0, which="LM")
                    end_time_1 = time()
                    # times_b[index_1][index_2] = (end_time_1 - start_time_1)/avg_num
                    # Goal: create two ndimensional arrays of appropriate size and store the benchmark results at the 
                    # appropriate index
                    # np.meshgrid(*xi, indexing='ij') """

def primme_benchmarker(qubit, names_list, vals_list, avg_num):
    ''' qubit: class instance of a qubit like a transmon or cos2phi
        names_list: a list of length N containing the names of the parameters or cutoffs to be varied
        vals_list: a list of length N containing lists of values of the parameters
        avg_num: increase or decreasing averaging, integer between 1 and infinity '''
    len_vals_list = [len(list) for list in vals_list]
    items = np.prod(len_vals_list)
    print(f"There should be {items} iterations")
    time_ratios = np.zeros(items)
    array_shape = tuple([len(list) for list in vals_list])
        
    for i, vals_set in enumerate(tqdm(product(*vals_list), leave=True)):
        for j, value in enumerate(vals_set):
            setattr(qubit, names_list[j], value)

        hamiltonian = qubit.hamiltonian()

        start_time = time()

        for _ in range(avg_num):
            result = pr.eigsh(hamiltonian, k=6, sigma = 0.0, which="LM")
        
        end_time = time()
        time_a = (end_time-start_time)/avg_num
        start_time_1 = time()
        for _ in range(avg_num):
            result = sp.sparse.linalg.eigsh(hamiltonian, k=6, sigma = 0.0, which="LM")
        end_time_1 = time()
        time_b = (end_time_1 - start_time_1)/avg_num
        time_ratios[i] = time_a/time_b
    
    res_array = np.reshape(time_ratios, array_shape) 

    return res_array
     

# Generates a heatmap of PRIMME performance given an appropriate matrix
def generate_heatmap(array, title, x_vals, y_vals, xlabel, ylabel, heatLabel, filename):
   
    """ im = plt.imshow(matrix, extent = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], aspect = aspect)
    plt.title(title)
    plt.xlabel("")
    plt.colorbar(im, label = heatLabel)
    plt.xlabel(xlabel, size = 20)
    plt.ylabel(ylabel, size = 20)
    plt.savefig(filename)
 """
    x, y = np.meshgrid(x_vals, y_vals)
    z = array
    fig, ax = plt.subplots()
    ax.set_title(title)
    plt.pcolormesh(x, y, z, cmap='OrRd', vmin = 0, 
                                        vmax = np.abs(z).max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(label = heatLabel)
    plt.savefig(filename)
    plt.show()
    return None
