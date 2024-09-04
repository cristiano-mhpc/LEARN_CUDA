#!/bin/bash

nvcc -arch=sm_89 trapezoid.cu -o trapezoid

#execute w th_per_blk = 32

./trapezoid 650 21 32 -5.0 5.0

