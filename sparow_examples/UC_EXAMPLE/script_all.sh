#!/bin/bash

echo "Starting HF at $(date)"
mpiexec -np 4 python HF_PH_parallel.py

echo "Starting LF at $(date)"
mpiexec -np 4 python LF_PH_parallel.py

echo "Starting MF at $(date)"
mpiexec -np 4 python MF_PH_random_parallel.py

echo "Finished all runs at $(date)"
