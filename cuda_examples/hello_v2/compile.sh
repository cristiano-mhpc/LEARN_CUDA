#!/bin/bash

nvcc -arch=sm_80 -o cuda_hello cuda_hello.cu

./cuda_hello 2 5


