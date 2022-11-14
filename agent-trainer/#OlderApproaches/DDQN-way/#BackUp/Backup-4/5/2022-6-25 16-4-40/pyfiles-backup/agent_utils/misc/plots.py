import numpy as np
import matplotlib.pyplot as plt

def plot_time_mem_curves(factor,sizes,exp=1.0,berryfieldside=20000,n=1000):
    arr = [[0]*n for l in sizes]
    for i in range(n-1):
        for j,l in enumerate(sizes):
            delta = factor*l/berryfieldside
            arr[j][i+1] = arr[j][i]*(1-delta) + delta
    for j in range(len(sizes)):
        plt.plot(np.array(arr[j])**exp, label=f'{sizes[j]}->{berryfieldside//sizes[j]}')
    plt.grid()
    plt.legend()
    plt.show()
