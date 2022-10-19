from re import I
import taichi as ti
import math
import matplotlib.cm as cm
from mpi4py import MPI
import numpy as np
import time

ti.init(arch=ti.cpu,default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)

N = 256 # Local field edge size
nw = 4
ntiles = nw
comm = MPI.COMM_WORLD
cart2d = comm.Create_cart(dims = [ntiles, ntiles], periods=[False, False])
coord = cart2d.Get_coords(comm.rank)
assert(ntiles * ntiles == comm.size)

# Local fields with halo edges
x  = ti.field(dtype=ti.f64, shape=(N + 2, N + 2))
xt = ti.field(dtype=ti.f64, shape=(N + 2, N + 2))
b  = ti.field(dtype=ti.f64, shape=(N + 2, N + 2))



def mpi_bc():
    x_arr = x.to_numpy()
    top = x_arr[1, :]
    bottom = x_arr[-2, :]
    left = x_arr[:, 1]
    right = x_arr[:, -2]
    rx = coord[0]
    ry = coord[1]
    # send top edge
    if rx > 0:
        comm.send(top, dest=cart2d.Get_cart_rank([rx - 1, ry]))
    # send bottom edge
    if rx < nw - 1:
        comm.send(bottom, dest=cart2d.Get_cart_rank([rx + 1, ry]))
    # send left edge
    if ry > 0:
        comm.send(left, dest=cart2d.Get_cart_rank([rx, ry - 1]))
    # send right edge
    if ry < nw - 1:
        comm.send(right, dest=cart2d.Get_cart_rank([rx, ry + 1]))

    # receive top edge
    if rx > 0:
        new_top = comm.recv(source=cart2d.Get_cart_rank([rx - 1, ry]))
        x_arr[0, :] = new_top
    # receive bottom edge
    if rx < nw - 1:
        new_bottom = comm.recv(source=cart2d.Get_cart_rank([rx + 1, ry]))
        x_arr[-1, :] = new_bottom
    # receive left edge
    if ry > 0:
        new_left = comm.recv(source=cart2d.Get_cart_rank([rx, ry - 1]))
        x_arr[:, 0] = new_left
    # receive right edge
    if ry < nw - 1:
        new_right = comm.recv(source=cart2d.Get_cart_rank([rx, ry + 1]))
        x_arr[:, -1] = new_right


    # Update Taichi field
    x.from_numpy(x_arr)

def mpi_gather_fields(x_np):
    gather_data = comm.gather(x.to_numpy(), root=0)
    if comm.rank == 0:
        for r, buf in enumerate(gather_data):
            coord = cart2d.Get_coords(r)
            rx = coord[0]
            ry = coord[1]
            x_np[rx * N : (rx+1) * N, ry * N : (ry+1) * N] = buf[1 : N + 1, 1 : N + 1]

@ti.kernel
def init_fields(rx : ti.i64, ry : ti.i64):
    for I in ti.grouped(x):
        x[I] = 0.0
        xt[I] = 0.0
    for i,j in b:
        xl = (i + rx * N) / (nw * N)
        yl = (j + ry * N) / (nw * N)

        if i == 0:
            xl = 0.0
        if i == N + 1:
            xl = 0.0
        if j == 0:
            yl = 0.0
        if j == N + 1:
            yl = 0.0

        b[i,j] = 0.05 * ti.sin(math.pi * xl) * ti.sin(math.pi * yl)

@ti.kernel
def enforce_bc():
    pass

@ti.kernel
def iter():
    # for i,j in ti.ndrange((1, N+1), (1, nw * N + 1)):
    for i,j in ti.ndrange((1, N+1), (1, N+ 1)):
        xt[i,j] = (b[i,j] + x[i+1,j] + x[i-1,j] + x[i,j+1] + x[i,j-1]) / 4.0
    for I in ti.grouped(x):
        x[I] = xt[I]

x_np = None
if comm.rank == 0:
    x_np = np.empty((nw * N, nw * N))
    gui = ti.GUI('Poisson Solver', (nw * N, nw * N))

init_fields(coord[0], coord[1])
i = 0
while True:
    st = time.time()
    enforce_bc()
    mpi_bc()
    iter()
    et = time.time()
    if comm.rank == 0:
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
