import theano
from theano import tensor as T 
from deepml.layers import TimeDistributedDense, GRU, AttentionARSG, Conv2D, max_pool_2d, Dense
from deepml.activations import relu, tanh, softmax

class AttentionARSGy(object):

    def __init__(self, n_in, n_out, n_inner, n_outer):

        self.n_in = n_in
        self.n_out = n_out
        self.n_inner = n_inner      # hidden states for inner loop
        self.n_outer = n_outer      # hidden states for outer loop

        self.gru_outer = GRU(n_in + n_out, n_outer)
        #self.gru_outer = GRU(n_in, n_outer)
        
        self.d_x = Dense(n_in, n_inner)
        self.d_h = Dense(n_outer, n_inner)
        self.d_alpha = Dense(1, n_inner)
        self.d_a = Dense(n_inner, 1)

        #self.d_ya = Dense(n_out, n_inner)
        #self.d_yy = Dense(n_out, n_out)
        self.d_hy = Dense(n_outer, n_out)

        self.params = self.gru_outer.params +  self.d_alpha.params + \
                self.d_x.params + self.d_h.params + self.d_a.params + \
                self.d_hy.params
    
    def apply(self, x, n_cycles):

        def inner_loop(y_tm1, h, alpha, x):
            # TODO: convolve alpha

            alpha = alpha[:, :, None]
            #alpha = alpha[:, None, :, None]
            #al = self.conv.apply(alpha, border_mode='full')[:,:,7:-7,0]

            # a: bs, seq_len, n_hid
            a = tanh(self.d_x.apply(x) + self.d_h.apply(h).dimshuffle(0,'x',1) + self.d_alpha.apply(alpha))

            a =  self.d_a.apply( a ).flatten(2)   # squeeze
            alpha = softmax(a) 

            glimpse = T.sum(x * alpha.dimshuffle(0,1,'x'), axis=1)

            glimpse_y = T.concatenate([glimpse, y_tm1], axis=1)

            # new state
            h = self.gru_outer.apply_one_step(glimpse_y, h)
            #h = self.gru_outer.apply_one_step(glimpse, h)

            y_hat = softmax( self.d_hy.apply(h) ) 

            return y_hat, h, alpha

        #h0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_outer)
        #alpha0 = T.alloc(np.cast[floatX](0.), x.shape[0], x.shape[1])
        #y0 = T.alloc(np.cast[floatX](0.), x.shape[0], self.n_out)
        alpha0 = T.zeros((x.shape[0], x.shape[1]))
        y0 = T.zeros((x.shape[0], self.n_out))
        h0 = T.ones((x.shape[0], self.n_outer))

        (Y,H,A), _ = theano.scan(
            inner_loop,
            outputs_info=[y0,h0,alpha0],
            n_steps=n_cycles,
            non_sequences=[x]
        )

        # H: n_cycles, batch_size, n_outer
        # A: n_cycles, batch_size, x_len

        return H.dimshuffle(1,0,2), A.dimshuffle(1,0,2) , Y.dimshuffle(1,0,2)

def create_model(n_in, n_out, n_enc, n_hid, n_cyc):

    x = T.ftensor3('x')     # x <batch_size, sequence_len, n_in>
    y = T.imatrix('y')      # y <batch_size, sequence_len>

    # Layers

    c0 = Conv2D(16, 1, 3, 3)
    c1 = Conv2D(32, 16, 3, 3)
    #c1 = Conv2D(32, 32, 3, 3)
    #c1 = Conv2D(32, 1, 3, 3)
    #g0 = GRU(n_in, n_enc)
    #g1 = GRU(32*12, n_enc)
    #g2 = GRU(32*12, n_enc)
    g1 = GRU(32*7, n_enc)
    g2 = GRU(32*7, n_enc)

    d2 = TimeDistributedDense(n_enc*2, n_enc)

    #att = AttentionARSGy(n_enc, n_hid, n_cyc)
    att = AttentionARSGy(n_enc, n_out, n_hid, n_cyc)
    #do = Dense(n_cyc, n_out)
    #do = TimeDistributedDense(n_cyc, n_out)

    params = [
        c0.params,
        c1.params,
        #g0.params,
        g1.params,
        g2.params,
        att.params,
        #do.params,
        #d_0.params,
        #d_1.params,
        d2.params,
    ]

    # Logic
    x0 = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))

    xc = relu(c0.apply(x0))
    xc = max_pool_2d(xc, (2,2)) 

    xc = relu(c1.apply(xc))
    #xc = max_pool_2d(xc, (2,2)) 

    #xc = relu(c1.apply(xc))

    x1 = xc.dimshuffle(0,2,1,3)
    x1 = x1.reshape((x1.shape[0], x1.shape[1], -1))

    #x0 = g0.apply(x0)
    #x1 = x0[:, ::skip_rate[0]]
    #x1 = d_0.apply(x1)
    x2_f = g1.apply(x1) #, truncate_gradient=30)
    x2_b = g2.apply(x1[:,::-1]) #, truncate_gradient=30)
    x2 = T.concatenate([x2_f, x2_b[:,::-1]], axis=2)

    #x2 = x2[:, ::skip_rate[0]]
    #x2 = d_1.apply(x2)
    x3 = d2.apply(x2)

    xe = x3

    Y = []
    A = []

    # extract glimplse

    H, alphas, out = att.apply(xe, y.shape[1]) 
    # H: batch_size, y_len, n_hid
    # alphas: batch_size, x_len
    o_shp = out.shape
    o = T.reshape(out, (-1, o_shp[2]))

    loss = T.nnet.categorical_crossentropy(o, y.flatten()).mean()

    params = [p for pp in params for p in pp]


    return [x, y], out, loss, params, alphas


