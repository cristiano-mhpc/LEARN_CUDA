#!/bin/bash

nvcc -arch=sm_80 trapezoid.cu -o trapezoid

./trapezoid 3000 3 1024 -5.0 5.0

