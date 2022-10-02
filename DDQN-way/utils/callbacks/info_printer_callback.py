import torch
import numpy as np

class InfoPrinterCallback:
    def __init__(self, ddqntraininer, tstrat, lr_scheduler, buffer) -> None:
        self.ddqntraininer = ddqntraininer
        self.tstrat = tstrat
        self.lr_scheduler = lr_scheduler
        self.buffer = buffer

    def __call__(self, info_dict):
        tolist = lambda t: t.tolist()
        eps = 10E-6
        ssize = self.ddqntraininer.batchSize
        try: skipsteps = self.ddqntraininer.skipSteps(self.ddqntraininer.current_episode)
        except Exception as e: skipsteps = self.ddqntraininer.skipSteps
        print('\t| epsilon:', self.tstrat.epsilon)
        if self.lr_scheduler is not None: print('\t| lr:', self.lr_scheduler.get_lr())
        if self.buffer.buffer is not None:
            positive_idx = np.argwhere((self.buffer.buffer['reward'] > eps).cpu().squeeze())
            a,count = torch.unique(self.buffer.buffer['action'][positive_idx], return_counts=True)
            sample = self.buffer.sample(ssize)[0]
            positive_idxsmp = np.argwhere((sample["reward"]>0).cpu().squeeze())
            asmp,countsmp = torch.unique(sample['action'][positive_idxsmp], return_counts=True)
            # print the stuffs!
            print('\t| skipsteps:', skipsteps)
            print('\t| positive-in-buffer:', positive_idx.shape[-1],
                f'| amount-filled: {100*len(self.buffer)/self.buffer.bufferSize:.2f}%')
            print('\t| action-stats: ', tolist(a), tolist(count))
            print(f'\t| approx positives in sample {ssize}: {positive_idxsmp.shape[-1]}')
            print(f'\t| approx action-dist in sample {ssize}: {tolist(asmp)} {tolist(countsmp)}')
        return info_dict