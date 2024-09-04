#!/bin/bash

nvcc -arch=sm_80 trapezoid.cu -o trapezoid

./trapezoid 32 1 64 -5.0 5.0

