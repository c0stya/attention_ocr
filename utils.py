from __future__ import division
import editdistance
import numpy as np
import h5py
import random

import random as rng

#rng.seed(42)

def filter_blanks(y_pred, blank_symbol):
    # y_pred: <bs, seq_len, n_tokens+1>

    Y = []
    for i in range(len(y_pred)):
        y = np.argmax(y_pred[i], axis=1)
        y = y[y < blank_symbol]    # filter out blanks
        Y.append(y)

    return Y 

def filter_repeats(y_pred):
    Y = []
    for i in range(len(y_pred)):
        y_ = y_pred[i]
        y = [y_[j] for j in range(len(y_)) 
                if j == 0 or y_[j-1] != y_[j] ]
        Y.append(y)
    return Y

def batched_wer(ref, hyp):
    ''' Computes mean WER 

    ref: list of references
    hyp: list of corresponding hypotheses
    
    '''
    assert len(ref) == len(hyp)

    wer = 0.
    for r,f in zip(ref, hyp):
        rate = editdistance.eval(r, f) / len(r)
        wer += rate
    
    return wer/len(ref)

def truncate_y(by, sep, n_words):
    truncated_y = []
    for y in by:
        indices = np.argwhere(np.equal(y, sep))
        if len(indices) < n_words:
            truncated_y.append(y)
        else:
            arr = np.split(y, indices[n_words-1])
            truncated_y.append(arr[0])

    return truncated_y

def align_x(sequences, filler=0, max_len=0):
    item = sequences[0]
    n_dim = item.shape[1]
    dtype = item.dtype

    if not (max_len > 0): 
        max_len = np.max([len(s) for s in sequences])
    aligned = np.zeros((len(sequences), max_len, n_dim)).astype(dtype)
    #aligned.fill(filler)

    mask = np.zeros((len(sequences), max_len), dtype=dtype)

    for i in range(len(sequences)):
        aligned[i, :len(sequences[i])] = sequences[i]
        mask[i, :len(sequences[i])] = 1

    return aligned, mask

def align_y(sequences, filler=0, max_len=0):
    item = sequences[0]
    dtype = item.dtype

    if not (max_len > 0): 
        max_len = np.max([len(s) for s in sequences])

    aligned = np.zeros((len(sequences), max_len)).astype(dtype)
    mask = np.zeros_like(aligned)
    aligned.fill(filler)

    for i in range(len(sequences)):
        aligned[i, 0:len(sequences[i])] = sequences[i]
        mask[i, 0:len(sequences[i])] = 1

    return aligned, mask

def reshape_x(x, shapes):
    ''' inplace reshaping '''
    for i in range(len(x)):
        x[i] = x[i].reshape(shapes[i])
    return x
    
def iter_batches(x, y, batch_size=32, x_shapes=None, shuffle=False):
    '''Iterate over dataset with reshaping of x.
    'x_shapes' is useful when iterate over H5 datasets.

    TODO: shuffle

    '''
    n_batches = len(x) // batch_size

    if shuffle:

        indices = np.arange(len(x)).astype(int)
        np.random.shuffle(indices)

        for i in range(n_batches):
            
            b_indices =  list(
                sorted(indices[i*batch_size:(i+1)*batch_size]))

            bx = x[b_indices]
            by = y[b_indices]
            bx_shapes = x_shapes[b_indices]

            if x_shapes:
                bx = reshape_x(bx, bx_shapes)

            yield bx, by

    else:
        for i in range(n_batches):
            
            bx = x[i*batch_size:(i+1)*batch_size]
            if x_shapes:
                bx = reshape_x(bx, x_shapes[i*batch_size:(i+1)*batch_size])

            by = y[i*batch_size:(i+1)*batch_size]

            yield bx, by



if __name__ == '__main__':
    print batched_wer( ['cat', 'dog'], ['caat', 'dg'])


