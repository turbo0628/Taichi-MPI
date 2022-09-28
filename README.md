# [WIP] Taichi-MPI demos

The Taichi MPI demos scaling with MPI4Py.

## Environment preparation

Anaconda or miniconda is recommended for Python environment management. [Official installation guide](https://docs.conda.io/en/latest/miniconda.html) for miniconda.

* Create the environment and install MPI4Py.
```
conda create -y -n mpi -c conda-forge python=3.8 mpi4py 
```
* Then activate the environment.
```
conda activate mpi
```
* Install Taichi and other dependencies.
```
pip install -U -r requirements.txt
```

## Reproduce on a single machine

There are 2 demos in this repository: 00-03 are Poisson solver with 1D and 2D parallelization and 04-06 are LBM solver with 1D parallelization. The LBM code is refactered from the [LBM_Taichi](https://github.com/hietwll/LBM_Taichi) repository.

To run:
```
mpirun -np <num_workers> code.py
```

All these demos have specific worker numbers, so I just list the run commands below for convenience.

```
python 00-base-poisson.py
mpirun -n 8 python 01-base-poisson-mpi-band.py
mpirun -n 16 python 02-base-poisson-mpi-halo.py
mpirun -n 16 python 02-base-poisson-mpi-halo.py
mpirun -n 16 python 03-base-poisson-mpi-halo-async.py
python 04-lbm-solver.py
mpirun -np 4 python 05-lbm-solver-mpi.py
mpirun -np 4 python 06-lbm-solver-mpi-faster-border.py
```

If you cannot reproduce, file an issue in this repo, thanks!

## Work in progress

* I haven't tested any MPI4Py programs on supercomputing clusters, it might have problems with different envs. I need a cluster to test these programs.
* Open MPI cannot properly work with the LBM demo, please make sure to use MPICH at the moment. (The conda version install MPICH automatically)
* We are writing a tutorial to explain how to parallelize a stencil computation program written in Taichi with MPI4Py, stay tuned!
