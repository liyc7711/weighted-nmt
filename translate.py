'''
Translates a source file using a translation model.
'''
import argparse
import subprocess
import numpy
import time
import sys
import cPickle as pkl

from nmt import (build_sampler, load_params, init_params, init_tparams)
from nmt import _idxs2words, gen_trans
from stream import get_sentences

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import shared
import pdb


def sample(model, dictionary, dictionary_target, \
           source_file, ref_file, saveto, \
           k=10, normalize=False, \
           bleu_script='./data/mteval-v11b.pl', res_to_sgm='./data/plain2sgm'):

    # load model model_options
    with open(model+'.pkl', 'rb') as f:
        options = pkl.load(f)

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk

    val_start_time = time.time()

    trng = RandomStreams(1234)
    use_noise = shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)
    # word index
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    bleu_score = gen_trans(test_src=source_file, test_ref=ref_file, out_file=saveto, \
                           dict_src=dictionary, idict_trg=word_idict_trg, \
                           tparams=tparams, f_init=f_init, f_next=f_next, model_options=options, \
                           trng=trng, k=10, stochastic=False)

    print(model+' / '+source_file+' / '+'test bleu %.4f' %bleu_score)
    print ('timestamp {} {}'.format('done',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    sys.stdout.flush()
    
if __name__ == "__main__":
    '''

    '''
    print ('timestamp {} {}'.format('running',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    dict_src = '/home/jhli/ycli/data/hw/vocab/vocab_src.pkl'
    dict_trg = '/home/jhli/ycli/data/hw/vocab/vocab_trg.pkl'

    test_model = ['./train_model.iter122000.npz']

    test_file = [['/home/jhli/ycli/data/hw/test/nist02.cn',
                 '/home/jhli/ycli/data/hw/test/nist02.ref',
                 './data/nist02.out'],
                 ['/home/jhli/ycli/data/hw/test/nist03.cn',
                 '/home/jhli/ycli/data/hw/test/nist03.ref',
                 './data/nist03.out'],
                 ['/home/jhli/ycli/data/hw/test/nist04.cn',
                 '/home/jhli/ycli/data/hw/test/nist04.ref',
                 './data/nist04.out'],
                 ['/home/jhli/ycli/data/hw/test/nist05.cn',
                 '/home/jhli/ycli/data/hw/test/nist05.ref',
                 './data/nist05.out'],
                 ['/home/jhli/ycli/data/hw/test/nist06.cn',
                 '/home/jhli/ycli/data/hw/test/nist06.ref',
                 './data/nist06.out'],
                 ['/home/jhli/ycli/data/hw/test/nist08.cn',
                 '/home/jhli/ycli/data/hw/test/nist08.ref',
                 './data/nist08.out']]

    for model in test_model:
        for i in range(len(test_file)):
            #if(i !=0 ):
            #    continue
            sample(model, dict_src, dict_trg, \
                   test_file[i][0], test_file[i][1], test_file[i][2])

