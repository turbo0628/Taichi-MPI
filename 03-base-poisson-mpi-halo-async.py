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

def mpi_bound_async_send(x_arr):
    top = x_arr[1, :]
    bottom = x_arr[-2, :]
    left = x_arr[:, 1]
    right = x_arr[:, -2]
    rx = coord[0]
    ry = coord[1]
    # send top edge
    if rx > 0:
        comm.isend(top, dest=cart2d.Get_cart_rank([rx - 1, ry]), tag=100)
    # send bottom edge
    if rx < nw - 1:
        comm.isend(bottom, dest=cart2d.Get_cart_rank([rx + 1, ry]), tag=101)
    # send left edge
    if ry > 0:
        comm.isend(left, dest=cart2d.Get_cart_rank([rx, ry - 1]), tag=110)
    # send right edge
    if ry < nw - 1:
        comm.isend(right, dest=cart2d.Get_cart_rank([rx, ry + 1]), tag=111)

    req_top = None
    req_bottom= None
    req_left = None
    req_right = None
    # receive top edge
    if rx > 0:
        req_top = comm.irecv(source=cart2d.Get_cart_rank([rx - 1, ry]), tag=101)
    # receive bottom edge
    if rx < nw - 1:
        req_bottom = comm.irecv(source=cart2d.Get_cart_rank([rx + 1, ry]), tag=100)
    # receive left edge
    if ry > 0:
        req_left = comm.irecv(source=cart2d.Get_cart_rank([rx, ry - 1]), tag=111)
    # receive right edge
    if ry < nw - 1:
        req_right = comm.irecv(source=cart2d.Get_cart_rank([rx, ry + 1]), tag=110)

    return req_top, req_bottom, req_left, req_right


def mpi_bound_update(x_arr, req_top, req_bottom, req_left, req_right):
    # Update Taichi field
    if req_top != None:
        new_top = req_top.wait()
        x_arr[0, :] = new_top
    if req_bottom != None:
        new_bottom = req_bottom.wait()
        x_arr[-1, :] = new_bottom
    if req_left != None:
        new_left = req_left.wait()
        x_arr[:, 0] = new_left
    if req_right != None:
        new_right = req_right.wait()
        x_arr[:, -1] = new_right
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
def iter_body():
    for i,j in ti.ndrange((2, N), (2, N)):
        xt[i,j] = (b[i,j] + x[i+1,j] + x[i-1,j] + x[i,j+1] + x[i,j-1]) / 4.0

@ti.kernel
def iter_bc():
    for i in ti.ndrange((1, N+1)):
        # top and bottom
        xt[1,i] = (b[1,i] + x[2,i] + x[0,i] + x[1,i+1] + x[1,i-1]) / 4.0
        xt[N,i] = (b[N,i] + x[N+1,i] + x[N-1,i] + x[N,i+1] + x[N,i-1]) / 4.0
        # left and right
        xt[i,1] = (b[i,1] + x[i+1,1] + x[i-1,1] + x[i,2] + x[i,0]) / 4.0
        xt[i,N] = (b[i,N] + x[i+1,N] + x[i-1,N] + x[i,N+1] + x[i,N-1]) / 4.0

@ti.kernel
def update_fields():
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
    x_arr = x.to_numpy()
    reqs = mpi_bound_async_send(x_arr)
    iter_body()
    mpi_bound_update(x_arr, *reqs)
    iter_bc()
    update_fields()
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
