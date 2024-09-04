#!/bin/bash

nvcc -arch=sm_89 trapezoid.cu -o trapezoid

./trapezoid 20000 20 1024 -5.0 5.0

