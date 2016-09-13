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

    h5f = h5py.File(sys.argv[1], 'r')

    f, (a0, a1) = plt.subplots(2,1)
    a0.plot(h5f['train_loss'], label='train_loss')
    a0.plot(h5f['test_loss'], label='test_loss')
    a0.legend()
    a1.plot(h5f['test_cer'], label='1-CER')
    a1.legend()
    plt.show()

    h5f.close()

