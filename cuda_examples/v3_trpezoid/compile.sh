#!/bin/bash

nvcc -arch=sm_89 trapezoid.cu -o trapezoid

./trapezoid 650 21 32 -5.0 5.0

