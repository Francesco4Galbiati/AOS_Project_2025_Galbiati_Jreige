#!/usr/bin/env bash

exe() { echo -e "\t\$ $@" ; "$@" ; }

inputfile="cuspis_lava.cu"
libcuda="/usr/local/cuda/lib64/"

exe nvcc -c $inputfile -o tmp.o -I./cuspis -g -G
exe g++ tmp.o -o out -L$libcuda -ldl -lrt -pthread -I./cuspis -lcuda -lcudart -g
exe rm tmp.o