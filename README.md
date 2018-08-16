# weighted-nmt
Source code for "Adaptive Weighting for Neural Machine Translation"

The source code is developed upon [dl4mt](https://github.com/nyu-dl/dl4mt-tutorial).

## Train
```
THEANO_FLAGS=device=gpu,floatX=float32 python train.py 
```
## Test
```
THEANO_FLAGS=device=gpu,floatX=float32 python translate.py 
```
## Reference
```
Yachao Li, Junhui Li and Min Zhang. Adaptive Weighting for Neural Machine Translation. Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018), 2018.
```
