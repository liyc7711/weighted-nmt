'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import pdb
import numpy
import copy

import os
import warnings
import sys
import time
import subprocess
import re
from collections import OrderedDict

from stream import TextIterator, get_sentences

profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# convert ids to tokens
def _idxs2words(idxs, ivocab, verbose=False, eos_token_id=0, unk_token_id=1):
    length = len(idxs)
    if(verbose):
	    try:
	        length = idxs.index(eos_token_id) + 1
	    except ValueError:
	        print 'idx2words error'

    return " ".join([ivocab.get(idx, unk_token_id) for idx in idxs[:length]])


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    print ('Loading model: '+path)
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]
        print("Loading {} shape {}".format( kk, str(vv.shape)))

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            #pdb.set_trace()
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    Wr = norm_weight(nin, dim)
    Wu = norm_weight(nin, dim)  
    params[_p(prefix, 'Wr')] = Wr
    params[_p(prefix, 'Wu')] = Wu

    params[_p(prefix, 'br')]  = numpy.zeros((dim,)).astype('float32')
    params[_p(prefix, 'bu')]  = numpy.zeros((dim,)).astype('float32')
    # recurrent transformation weights for gates
    Ur = ortho_weight(dim)
    Uu = ortho_weight(dim)
    params[_p(prefix, 'Ur')] = Ur
    params[_p(prefix, 'Uu')] = Uu
    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux
    # for wnmt
    params[_p(prefix, 'W_g_x')] = norm_weight(nin, dim)
    params[_p(prefix, 'W_g_h')] = ortho_weight(dim)
    params[_p(prefix, 'b_g')] = numpy.zeros((dim,)).astype('float32')
    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)


    # state_below is the input word embeddings
    state_below_u = tensor.dot(state_below, tparams[_p(prefix, 'Wu')]) + tparams[_p(prefix, 'bu')]
    state_below_r = tensor.dot(state_below, tparams[_p(prefix, 'Wr')]) + tparams[_p(prefix, 'br')]
    state_below_g = tensor.dot(state_below, tparams[_p(prefix, 'W_g_x')]) + tparams[_p(prefix, 'b_g')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')])  + tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_u, x_r, xx_, x_g, h_, Uu, Ur, Ux, W_g_h):
        
        gi = tensor.nnet.sigmoid(tensor.dot(h_, W_g_h) + x_g)
        # reset and update gates
        r = tensor.nnet.sigmoid(tensor.dot(h_, Ur) * gi + x_r * (1. - gi))
        u = tensor.nnet.sigmoid(tensor.dot(h_, Uu) * gi + x_u * (1. - gi))
        # compute the hidden state proposal
        cand_h = tensor.tanh(tensor.dot(h_, Ux) * r * gi + xx_ * (1. - gi))

        # leaky integrate and obtain next hidden state
        h = u * h_ * gi + (1. - u) * cand_h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_ * gi

        return h

    # prepare scan arguments
    seqs = [mask, state_below_u, state_below_r, state_belowx, state_below_g]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'Uu')],
                   tparams[_p(prefix, 'Ur')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'W_g_h')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    params[_p(prefix, 'Wr')] = norm_weight(nin, dim)
    params[_p(prefix, 'Wu')] = norm_weight(nin, dim)
    params[_p(prefix, 'br')] = numpy.zeros((dim,)).astype('float32')
    params[_p(prefix, 'bu')] = numpy.zeros((dim,)).astype('float32')

    params[_p(prefix, 'Ur')] = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Uu')] = ortho_weight(dim_nonlin)
    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    params[_p(prefix, 'U_nl_r')] = ortho_weight(dim_nonlin)
    params[_p(prefix, 'U_nl_u')] = ortho_weight(dim_nonlin)
    params[_p(prefix, 'b_nl_r')] = numpy.zeros((dim_nonlin,)).astype('float32')
    params[_p(prefix, 'b_nl_u')] = numpy.zeros((dim_nonlin,)).astype('float32')
    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    params[_p(prefix, 'Wc_r')] = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wc_u')] = norm_weight(dimctx, dim)
    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att
    # for wnmt, 
    params[_p(prefix, 'W_g1_x')] = norm_weight(nin, dim)
    params[_p(prefix, 'W_g1_h')] = ortho_weight(dim_nonlin)
    params[_p(prefix, 'b_g1')] = numpy.zeros((dim_nonlin,)).astype('float32')
    params[_p(prefix, 'W_g2_h')] = ortho_weight(dim_nonlin)
    params[_p(prefix, 'W_g2_ctx')] = norm_weight(dimctx, dim)
    params[_p(prefix, 'b_g2')] = numpy.zeros((dim_nonlin,)).astype('float32')
    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + tparams[_p(prefix, 'b_att')]


    # projected x
    state_belowx  = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    state_below_r = tensor.dot(state_below, tparams[_p(prefix, 'Wr')]) + tparams[_p(prefix, 'br')]
    state_below_u = tensor.dot(state_below, tparams[_p(prefix, 'Wu')]) + tparams[_p(prefix, 'bu')]
    # for wnmt
    state_below_g = tensor.dot(state_below, tparams[_p(prefix, 'W_g1_x')]) + tparams[_p(prefix, 'b_g1')]

    def _step_slice(m_, x_r, x_u, xx_, x_g, h_, ctx_, alpha_, pctx_, cc_,
                    Ur, Uu, Wc_r, Wc_u, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl_r, U_nl_u, Ux_nl, b_nl_r, b_nl_u, bx_nl, 
                    W_g1_h, W_g2_h, W_g2_ctx, b_g2):

    	g1 = tensor.nnet.sigmoid(tensor.dot(h_, W_g1_h) + x_g)
        r1 = tensor.nnet.sigmoid(tensor.dot(h_, Ur) * g1 + x_r * (1. - g1))
        u1 = tensor.nnet.sigmoid(tensor.dot(h_, Uu) * g1 + x_u * (1. - g1))
        cand_x1 = tensor.tanh(tensor.dot(h_, Ux) * r1 * g1 + xx_ * (1. - g1))

        h1 = u1 * h_ * g1 + (1. - u1) * cand_x1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_ * g1

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]

        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att) + c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        # stablized alpha softmax
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        #alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        g2 = tensor.nnet.sigmoid(tensor.dot(h1, W_g2_h) + tensor.dot(ctx_, W_g2_ctx) + b_g2)
        r2 = tensor.nnet.sigmoid(tensor.dot(h1, U_nl_r) * g2 + tensor.dot(ctx_, Wc_r) * (1. - g2) + b_nl_r)
        u2 = tensor.nnet.sigmoid(tensor.dot(h1, U_nl_u) * g2 + tensor.dot(ctx_, Wc_u) * (1. - g2) + b_nl_u)
        cand_x2 = tensor.tanh((tensor.dot(h1, Ux_nl) + bx_nl ) * r2 * g2 + tensor.dot(ctx_, Wcx)) * (1. - g2)

        h2 = u2 * h1 * g2 + (1. - u2) * cand_x2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1 * g2

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_r, state_below_u, state_belowx, state_below_g]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'Ur')],
                   tparams[_p(prefix, 'Uu')],
                   tparams[_p(prefix, 'Wc_r')],
                   tparams[_p(prefix, 'Wc_u')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl_r')],
                   tparams[_p(prefix, 'U_nl_u')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl_r')],
                   tparams[_p(prefix, 'b_nl_u')],
                   tparams[_p(prefix, 'bx_nl')],
                   tparams[_p(prefix, 'W_g1_h')],
                   tparams[_p(prefix, 'W_g2_h')],
                   tparams[_p(prefix, 'W_g2_ctx')],
                   tparams[_p(prefix, 'b_g2')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    ctxdim = 2 * options['dim']

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim'])
    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim)
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    # weighted readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm_mem',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev_mem',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx_mem',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params['W_m_m'] = ortho_weight(options['dim_word'])
    params['W_m_r'] = ortho_weight(options['dim_word'])
    params['W_m_p'] = ortho_weight(options['dim_word'])
    params['b_m_b'] = numpy.zeros((options['dim_word'],)).astype('float32')
    
    return params


def deep_readout(tparams, options, proj_h, emb, ctxs):
    
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    
    mem_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm_mem', activ='linear')
    mem_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev_mem', activ='linear')
    mem_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx_mem', activ='linear')
    
    mem_read = logit_lstm + logit_prev + logit_ctx
    
    re_cons = False
    if(mem_read.ndim == 2):
        mem_read = mem_read.reshape((1, mem_read.shape[0], mem_read.shape[1]))
        mem_lstm = mem_lstm.reshape((1, mem_lstm.shape[0], mem_lstm.shape[1]))
        mem_prev = mem_prev.reshape((1, mem_prev.shape[0], mem_prev.shape[1]))
        mem_ctx  = mem_ctx.reshape((1,  mem_ctx.shape[0],  mem_ctx.shape[1]))
        re_cons  = True

    mem = tensor.zeros((3, mem_read.shape[0], mem_read.shape[1], mem_read.shape[2]))
    mem = tensor.set_subtensor(mem[0:1], mem_lstm)
    mem = tensor.set_subtensor(mem[1:2], mem_prev)
    mem = tensor.set_subtensor(mem[2:3], mem_ctx)

    p_from_mem  = tensor.dot(mem, tparams['W_m_m'])
    p_from_read = tensor.dot(mem_read, tparams['W_m_r'])
    p = tensor.tanh(p_from_mem + p_from_read + tparams['b_m_b'])
    alpha = tensor.dot(p, tparams['W_m_p'])
    alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
    normalizer = alpha.sum(axis=0, keepdims=True)
    probs = alpha / normalizer

    if(re_cons):
        probs = probs.reshape((probs.shape[0], probs.shape[2], probs.shape[3]))

    p0 = probs[0]
    p1 = probs[1]
    p2 = probs[2]

    readout = tensor.tanh(logit_lstm * p0 + logit_prev * p1 + logit_ctx * p2)

    return readout


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)
    # word embedding for backward rnn (source)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit = deep_readout(tparams, options, proj_h, emb, ctxs)

    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost


# build a sampler
def build_sampler(tparams, options, trng, use_noise):
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder')
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r')

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    logit = deep_readout(tparams, options, next_state, emb, ctxs)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=10, maxlen=200,
               stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score -= numpy.log(next_p[0, nw])
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            pdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)

def gen_trans(test_src, test_ref, out_file, dict_src, idict_trg, \
                    tparams, f_init, f_next, model_options, trng, \
                    k=10,
                    maxlen=200,
                    stochastic=False,
                    argmax=False,
                    normalize=False,
                    bleu_script='./data/mteval-v11b.pl',
                    res_to_sgm='./data/plain2sgm'):
    
    print("Begin decoding ...")
    fout = open(out_file, 'w')
    start_time = time.time()
    i = 0
    src = get_sentences(test_src, dict_src)
    
    for sent in src:
        i += 1
        sample, score = gen_sample( tparams=tparams, f_init=f_init, f_next=f_next,
                                    x=numpy.array(sent).reshape([len(sent), 1]),
                                    options=model_options, trng=trng, k=k, maxlen=maxlen,
                                    stochastic=stochastic, argmax=argmax)
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        res = _idxs2words(sample[sidx][:-1], idict_trg)

        if(res.strip() == ''):
        	res = 'UNK'

        fout.write(res + '\n')
        if i % 100 == 0:
            print("Translated {} lines of valid/test set ...".format(i))
        sys.stdout.flush()

    fout.close()
    
    print('Decoding took %.4f minutes'%(float(time.time() - start_time) / 60.))
    
    try:
        print("Evaluate ...")  
        cmd_res_to_sgm = ['python ', res_to_sgm + '.py', out_file, test_src+'.sgm', out_file + '.sgm']      
        cmd_bleu_cmd = ['perl', bleu_script, \
                        '-r',   test_ref+'.sgm', \
                        '-s',   test_src+'.sgm', \
                        '-t',   out_file+'.sgm', \
                        '>',    out_file+'.eval']
        
        print('Covert result to sgm')
        subprocess.check_call(" ".join(cmd_res_to_sgm), shell=True)
        print('Compute bleu score')
        subprocess.check_call(" ".join(cmd_bleu_cmd), shell=True)

        fin = open(out_file+'.eval', 'rU')
        out = re.search('BLEU score = [-.0-9]+', fin.readlines()[7])
        fin.close()

        bleu_score = float(out.group()[13:])

        return bleu_score
    except:
        print('Evaluating error, return null')
        return 0

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8):

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    updates = []

    t_prev = theano.shared(numpy.float32(0.))
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2**t) / (1. - beta1**t)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=100000,  # source vocabulary size
          n_words=100000,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=2000,
          validFreq_fine=1000,
          val_burn_in=30000,
          val_burn_in_fine=60000, 
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=['./data/train_source', './data/train_target'],
          valid_datasets=['./data/valid_src', './data/valid_ref', './data/out_file'],
          dictionaries=['./data/vocab_source', './data/vocab_target'],
          use_dropout=False,
          reload_=False,
          overwrite=False):

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options'
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'
    train = TextIterator(datasets[0], datasets[1],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng, use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    
    if decay_c > 0.:
    	# apply L2 regularization on weights
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
        # apply L1 regularization on weights
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += abs(vv).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size
    sys.stdout.flush()
    

    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask)

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                pdb.set_trace()
                continue
                #return 1., 1., 1.

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print('Epoch %d  Updates %d Cost %.4f Time %.4f'%(eidx, uidx, cost, ud))
            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(3, x.shape[1])):
                    stochastic = True
                    sample, score = gen_sample(tparams, f_init, f_next,
                                               x[:, jj][:, None],
                                               model_options, trng=trng, k=1,
                                               maxlen=maxlen,
                                               stochastic=stochastic,
                                               argmax=False)
                    
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    #pdb.set_trace()
                    print("Source: {}".format(_idxs2words(idxs=x[:, jj].tolist(), ivocab=worddicts_r[0], verbose=True)))
                    print("Truth: {}".format(_idxs2words(idxs=y[:, jj].tolist(), ivocab=worddicts_r[1])))
                    print("Sample: {}".format(_idxs2words(idxs=ss, ivocab=worddicts_r[1])))
                    print
                sys.stdout.flush()

            if (uidx < val_burn_in):
            	continue

            # validate model on validation set and early stop if necessary
            if  (uidx <= val_burn_in_fine and numpy.mod(uidx, validFreq) == 0 ) or \
                 uidx >  val_burn_in_fine and numpy.mod(uidx, validFreq_fine) == 0 :
                
                use_noise.set_value(0.)
                valid_err = gen_trans(test_src=valid_datasets[0], test_ref=valid_datasets[1], out_file=valid_datasets[2], \
                                      dict_src=dictionaries[0], idict_trg=worddicts_r[1], \
                                      tparams=tparams, f_init=f_init, f_next=f_next, model_options=model_options, \
                                      trng=trng, k=10, stochastic=False)
                print('Valid  Epoch %d  Updates %d valid_bleu %.4f'%(eidx, uidx, valid_err))
                print ('timestamp {} {}'.format(int(uidx),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sys.stdout.flush()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    pdb.set_trace()
                #-----------------------------------
                if (uidx > val_burn_in):
	                # save with uidx
	                if not overwrite:
	                    print 'Saving the model at iteration {}...'.format(uidx),
	                    saveto_uidx = '{}.iter{}.npz'.format(
	                        os.path.splitext(saveto)[0], uidx)
	                    numpy.savez(saveto_uidx, history_errs=history_errs,
	                                uidx=uidx, **unzip(tparams))
	                    pkl.dump(model_options, open('%s.pkl' % saveto_uidx, 'wb'))
	                    print 'Done'

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = valid_sample(test_src=valid_datasets[0], test_ref=valid_datasets[1], out_file=valid_datasets[2], \
                             dict_src=dictionaries[0], idict_trg=worddicts_r[1], \
                             tparams=tparams, f_init=f_init, f_next=f_next, model_options=model_options, \
                             trng=trng, k=10, stochastic=False)
    print 'Valid_bleu ', valid_err
    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    print ('timestamp {} {}'.format('done',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    return valid_err


if __name__ == '__main__':
    pass
