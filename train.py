from deepml.solvers import rmsprop, sgd, adadelta
from deepml.activations import relu, tanh, softmax, sigmoid
from deepml.utils import floatX

import theano

import numpy as np
import pylab as pl
import time

import cPickle as pickle
import os

from utils import align_x, align_y, iter_batches, batched_wer, filter_blanks

from h5monitor import dump_h5, dump_h5_var

import h5py

from arsg_cnn_bi_y import create_model
from gen_text import tokens, render_batch

### RND ###
#srng = RandomStreams()

### DATA ###

monitor_file = 'stats.h5'

### META-PARAMETERS ###

#n_in = train_x_shp[0,1]
n_in = 20
n_hid = 128
n_cyc = 512
n_enc = 64
n_out = len(tokens)+1
blank_symbol = len(tokens)

batch_size = 32
test_batch_size = 128

seq_len = (10,35)
CER_RATE = 1.

### ROUTINES ###

def decode(y):
   return np.array([tokens[t] for t in y])

### MODEL ###

def train(model):
    global seq_len

    [x, y], out, cost, params, alpha = model

    tester = theano.function(inputs=[x,y], outputs=[cost, out, alpha])

    grad_updates = adadelta(
        eps=1e-9,
        cost = cost,
        params = params,
        #lr=3.,
        #grad_norm=1.
    )

    solver = theano.function(
        inputs=[x, y],
        outputs=cost,
        updates = grad_updates,
    )

    # alpha: bs, len_y, len_x


    costs = []
    t0 = time.time()

    for i in range(100000):

        bx, by = render_batch(seq_len, batch_size=batch_size)


        bx_ = np.float32(bx.transpose(0,2,1))/255.
        #bx_, _ = align_x(bx)
        by_, _ = align_y(by, filler=blank_symbol)

        loss = solver(bx_, by_)
        costs.append(loss)

        # monitoring training process

        if i and i%100 == 0:
            print 'Iteration %d, time: %.4f, loss %.8f' % (
                i, time.time() - t0, np.mean(costs))

            dump_h5(monitor_file, prefix='train_loss', 
                data=[np.mean(costs)])

            costs = []
            t0 = time.time()

        
        # testing

        if i and i%100 == 0:

            test_costs = []
            t1 = time.time()
            cer = []

            for j in range(32):
                bx, by = render_batch(seq_len, batch_size=batch_size)

                #bx_, _ = align_x(bx)
                bx_ = np.float32(bx.transpose(0,2,1))/255.
                by_, _ = align_y(by, filler=blank_symbol)

                loss, y_pred, alpha = tester(bx_, by_)

                y_filt = filter_blanks(y_pred, blank_symbol)
                wer = batched_wer(by, y_filt)

                test_costs.append(loss)
                cer.append(1-wer)

            with open('model.pkl', 'w') as h:
                pickle.dump(model, h)
            
            dump_h5(monitor_file, prefix='test_loss', 
                data=[np.mean(test_costs)])
            dump_h5(monitor_file, prefix='test_cer', data=[np.mean(cer)])

            dump_h5_var(monitor_file, 
                prefix='test_alpha', 
                prefix_shape='test_alpha_shp', 
                data=alpha[0:1])
            dump_h5_var(monitor_file, 
                prefix='test_x',
                prefix_shape='test_x_shp',
                data=[bx_[0]])
            dump_h5_var(monitor_file, 
                prefix='test_y',
                prefix_shape='test_y_shp',
                data=[decode(y_filt[0])])

            print 'Iteration: %d, time: %.4f, test loss: %.8f, CER*: %.4f' % (i, time.time() - t1, np.mean(test_costs), np.mean(cer))

            if np.mean(cer) > CER_RATE:
                seq_len = (seq_len[0] + 1, seq_len[1] + 2)
                print 'Increasing sequence length:', seq_len

        i+=1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default=None, 
        help='Pre-trained model')

    args = parser.parse_args()
    if args.model is None:
        #model = create_model(n_in, n_out, n_enc, n_hid, n_cyc, batch_size)
        model = create_model(n_in, n_out, n_enc, n_hid, n_cyc)
    else:
        with open(args.model, 'r') as h:
            model = pickle.load(h)

    train(model)

