import taichi as ti
import math
import matplotlib.cm as cm
from mpi4py import MPI
import numpy as np
import time

ti.init(arch=ti.cpu,default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)

N = 128 # Local field edge size
nw = 8 # num workers
N_glo = N * nw # frame edge length of the entire field

comm = MPI.COMM_WORLD
assert(nw == comm.size)

# Local fields with halo edges
x  = ti.field(dtype=ti.f64, shape=(N + 2, nw * N + 2))
xt = ti.field(dtype=ti.f64, shape=(N + 2, nw * N + 2))
b  = ti.field(dtype=ti.f64, shape=(N + 2, nw * N + 2))


def mpi_bc():
    x_arr = x.to_numpy()
    top = x_arr[1, :]
    bottom = x_arr[-2, :]

    # send top to upper worker
    if comm.rank > 0:
        comm.send(top, dest=comm.rank-1)
    if comm.rank < nw - 1:
        new_bottom = comm.recv(source=comm.rank+1)
        x_arr[-1, :] = new_bottom

    # send bottom to lower worker
    if comm.rank < nw - 1:
        comm.send(bottom, dest=comm.rank+1)
    if comm.rank > 0:
        new_top = comm.recv(source=comm.rank-1)
        x_arr[0, :] = new_top

    # Update Taichi field
    x.from_numpy(x_arr)

def mpi_gather_fields(x_np):
    gather_data = comm.gather(x.to_numpy(), root=0)
    if comm.rank == 0:
        for i, buf in enumerate(gather_data):
            x_np[i * N : (i+1) * N, :] = buf[1 : N + 1, 1 : nw * N + 1]

@ti.kernel
def init_fields(rank : ti.i64):
    for I in ti.grouped(x):
        x[I] = 0.0
        xt[I] = 0.0
    for i,j in b:
        xl = (i + rank * N) / (nw * N)
        yl = (j) / (nw * N)

        if i == 0:
            xl = 0.0
        if i == N + 1:
            xl = 0.0
        if j == 0:
            yl = 0.0
        if j == nw * N + 1:
            yl = 0.0

        b[i,j] = 0.05 * ti.sin(math.pi * xl) * ti.sin(math.pi * yl)

@ti.kernel
def enforce_bc():
    pass

@ti.kernel
def iter():
    for i,j in ti.ndrange((1, N+1), (1, nw * N + 1)):
        xt[i,j] = (b[i,j] + x[i+1,j] + x[i-1,j] + x[i,j+1] + x[i,j-1]) / 4.0
    for I in ti.grouped(x):
        x[I] = xt[I]

x_np = None
if comm.rank == 0:
    x_np = np.empty((nw * N, nw * N))
    gui = ti.GUI('Poisson Solver', (nw * N, nw * N))

init_fields(comm.rank)
i = 0
while True:
    st = time.time()
    enforce_bc()
    mpi_bc()
    iter()
    et = time.time()
    print(f"Pure compute FPS {1.0/(et - st)}", flush=True)
    # Inefficient image composition with gather
    mpi_gather_fields(x_np)
    if comm.rank == 0:
        if not gui.running:
            comm.Abort()
            break
        x_img = cm.jet(x_np)
        gui.set_image(x_img)
        gui.show()
