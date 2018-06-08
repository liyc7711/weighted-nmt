#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu1,lib.cnmem=0.1,floatX=float32

#cd $PBS_O_WORKDIR
python translate.py


