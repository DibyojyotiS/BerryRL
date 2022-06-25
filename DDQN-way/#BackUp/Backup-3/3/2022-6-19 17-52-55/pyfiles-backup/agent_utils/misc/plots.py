import matplotlib.pyplot as plt

def plot_time_mem_curves(factor,sizes,berryfieldside=20000,n=1000):
    sizes = [50,100,200]
    arr = [[0]*n for l in sizes]
    for i in range(n-1):
        for j,l in enumerate(sizes):
            delta = factor*l/berryfieldside
            arr[j][i+1] = arr[j][i]*(1-delta) + delta
    for j in range(len(sizes)):
        plt.plot(arr[j], label=f'{sizes[j]}->{berryfieldside//l}')
    plt.legend()
    plt.show()
