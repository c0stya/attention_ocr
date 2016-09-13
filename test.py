from deepml.solvers import rmsprop, sgd, adadelta
from deepml.activations import relu, tanh, softmax, sigmoid
from deepml.utils import floatX

import theano

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as grd
import time

import cPickle as pickle
import os

from utils import batched_wer, filter_blanks
from gen_text import tokens, render_batch, render_single, encode


### RND ###
#srng = RandomStreams()

### DATA ###

monitor_file = 'stats.h5'

### META-PARAMETERS ###

#n_in = train_x_shp[0,1]
blank_symbol = len(tokens)

seq_len = (10,20)

### ROUTINES ###

def plot_alpha(alpha, x, y):

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
    plt.subplots_adjust(top=None, bottom=None, wspace=0.05, hspace=0.05)

    plt.show()

def decode(y):
   return np.array([tokens[t] for t in y])

### MODEL ###

def test(model, text):

    [x, y], out, cost, params, alpha = model

    tester = theano.function(inputs=[x,y], outputs=[cost, out, alpha])

    t0 = time.time()

    test_costs = []
    t1 = time.time()
    cer = []

    if text:
        bx = [render_single(text)]
        bx = np.array(bx)

        by = [encode(text)]
    else:
        bx, by = render_batch(seq_len, 1)

    bx_ = np.float32(bx.transpose(0,2,1))/255.
    by_ = by

    loss, y_pred, alpha = tester(bx_, by_)

    y_filt = filter_blanks(y_pred, blank_symbol)
    wer = batched_wer(by, y_filt)

    test_costs.append(loss)
    cer.append(1-wer)

    y_given_decoded = decode(by_[0])
    y_pred_decoded = decode(y_filt[0]) 

    print 'Actual/Predicted:\n%s (%d)\n%s (%d)' % (
        "".join(y_given_decoded), 
        len(y_given_decoded),
        "".join(y_pred_decoded),
        len(y_pred_decoded),
    )

    print 'time: %.4f, test loss: %.8f, CER*: %.4f' % (time.time() - t0, np.mean(test_costs), np.mean(cer))

    plot_alpha(alpha[0], bx_[0], y_pred_decoded)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True, 
        help='Pre-trained model')
    parser.add_argument('--text', '-t', default=None, 
        help='Text to render')

    args = parser.parse_args()
    with open(args.model, 'r') as h:
        model = pickle.load(h)

    test(model, args.text)

