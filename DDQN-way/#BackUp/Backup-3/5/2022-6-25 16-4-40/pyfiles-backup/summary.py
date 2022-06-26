import os
import matplotlib.pyplot as plt
from utils import plot_berries_picked

# if __name__ == "__main__":
#     for logdir in os.listdir('.temp'):
#         print(logdir)
#         plot_berries_picked(f'.temp/{logdir}')
#         plt.show()

if __name__ == "__main__":
    LOG_DIR = '..'
    plot_berries_picked(LOG_DIR)
    plt.show()