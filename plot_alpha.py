import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as grd
import h5py
import sys

n_avg = 32

if __name__ == '__main__':
    if len(sys.argv) != 2: 
        print "Usage: %s <file.h5>" % (sys.argv[0],)
        sys.exit(-1)

    idx = -1
    h5f = h5py.File(sys.argv[1], 'r')

    test_loss = h5f['test_loss'][:]

    for i in range(len(h5f['test_alpha'])-1, -1, -1):
        idx = i
        alpha = h5f['test_alpha'][idx]
        alpha_shp = h5f['test_alpha_shp'][idx]

        alpha.shape = alpha_shp

        x = h5f['test_x'][idx]
        x.shape = h5f['test_x_shp'][idx]
        y = h5f['test_y'][idx]

        '''
        loss = np.convolve(
            train_loss, np.ones((n_avg,))/n_avg, mode='valid')
        plt.plot(loss)
        plt.show()
        '''

        f, (a0, a1) = plt.subplots(2)
        gs = grd.GridSpec(2,1, wspace=0.01) #, height_ratios=[1, 4])
        a0 = plt.subplot(gs[0])

        a0.matshow(x.T, cmap=plt.cm.Greys_r) #, aspect='auto')

        probs = np.zeros_like(alpha)
        for i in range(len(alpha)):
            probs[i] = np.convolve(
                alpha[i], np.ones((2,))/2., mode='same')

        a1.matshow(alpha, interpolation='none', aspect='auto')
        xticks = np.argmax(probs, axis=1)

        a1.set_xticks(xticks)
        a1.set_xticklabels(y, fontsize=16)
        a1.grid(which='both') 
        #plt.tight_layout()
        plt.subplots_adjust(top=None, bottom=None, wspace=0.05, hspace=0.05)

        plt.show()
        
    h5f.close()

