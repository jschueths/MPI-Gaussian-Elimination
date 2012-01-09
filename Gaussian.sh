#!/bin/bash

## Specify a jobname
#PBS -N GAUSSIAN-800-16

## Set the wall time HH:MM:SS (jobs using <=15 minutes are given priority)
#PBS -l walltime=00:15:00

## Set The number of nodes (jobs using <=16 nodes with <=2 ppn are given priority)
#PBS -l nodes=16

## Allocate some memory at each processor
##PBS -l mem=1gb

## Combine standard output and error files
#PBS -j oe

## Set the submission queue
#PBS -q @nic-cluster.srv.mst.edu

## Prerun that writes useful job info into the error file
/share/apps/job_data_prerun.py

mpirun -n 16 /nethome/users/jpm2t4/gaussian/gaussian /nethome/users/jpm2t4/gaussian/matrix.800.txt
