
import os
import matplotlib.pyplot as plt

def plot_berries_picked(LOG_DIR):
    """ plots the number of berries picked in 
    train and eval versus the episode number """

    def get_data_points(path):
        if not os.path.exists(path): return
        data_nberries = []
        episodes = list(map(int,
            [x for x in os.listdir(path) 
            if os.path.isfile(f'{path}/{x}/results.txt')]))
        for i in sorted(episodes):
            pt = f'{path}/{i}/results.txt'
            with open(pt,'r') as f: 
                line = f.readlines()[-2]
                nberries = int(line.split(':')[-1].strip())
            data_nberries.append(nberries)    
        return data_nberries

    # for train
    data1 = get_data_points(f'{LOG_DIR}/analytics-berry-field/')
    
    # for eval
    data2 = get_data_points(f'{LOG_DIR}/eval/analytics-berry-field/')

    # plot stuff
    if data1: 
        plt.plot(data1)
        plt.title('berries collected during train')
        plt.show()
    if data2:
        plt.plot(data2)
        plt.title('berries collected during eval')
        plt.show()

# if __name__ == "__main__":
#     LOG_DIR = 'D:\Machine_Learning\RL\Foraging-in-a-field\DDQN-way\BackUp\\backup-1\\2022-6-15 19-44-17' 
#     LOG_DIR = 'D:\Machine_Learning\RL\Foraging-in-a-field\DDQN-way\.temp\\2022-6-16 8-25-6'
#     plot_berries_picked(LOG_DIR)