import numpy as np

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    def x(a, l=1000):
        INVSQRT2 = 0.5**0.5
        pa = np.zeros(8)
        pa[a]=1
        action_switcher = {
            0: (0, 1), # N
            1: (INVSQRT2, INVSQRT2), # NE
            2: (1, 0), # E
            3: (INVSQRT2, -INVSQRT2), # SE
            4: (0, -1), # S
            5: (-INVSQRT2, -INVSQRT2), # SW
            6: (-1, 0), # W
            7: (-INVSQRT2, INVSQRT2), # NW
        }
        x=10000; y=10000
        pth = [(x,y)]
        ah = []
        tmep = 0.09
        for i in range(l):
            # probs = softmax(from_numpy(pa),dim=-1)
            # ac = np.random.choice(8, p=probs)
            probs = pa/sum(pa)
            ac = np.random.choice(8, p=probs)
            # dx,dy = action_switcher[ac]
            # x += dx; y+=dy
            for i in range(10):
                dx,dy = action_switcher[ac]
                x += dx; y+=dy
            pa[a]=pa[(a-1)%8]=pa[(a+1)%8]=0
            pa[ac] =  0.994
            pa[(ac+1)%8]= pa[(ac-1)%8] = (1-pa[ac])/2
            # pa[ac] = 1/tmep
            # pa[(ac+1)%8]= pa[(ac-1)%8] = 0.1/tmep
            a = ac
            pth.append((x,y))
            # ah.append(probs.numpy())
            ah.append(probs)

        return pth, ah

    # np.random.seed(0)

    p,ah = x(0,1000)
    p = np.array(p)
    # plt.ylim(0,20000)
    # plt.xlim(0,20000)
    plt.plot(p[:,0],p[:,1])
    plt.show()
    plt.plot(ah)