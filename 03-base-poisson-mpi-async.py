import taichi as ti
import math
import matplotlib.cm as cm
from mpi4py import MPI
import numpy as np
import time

ti_data_type=ti.f64

ti.init(arch=ti.cpu,default_fp=ti_data_type, offline_cache=False)

N = 1024 # Local field edge size
nw = 4 # num workers
N_glo = N * nw # frame edge length of the entire field

comm = MPI.COMM_WORLD
assert(nw == comm.size)

# Local fields with halo edges
x  = ti.field(dtype=ti_data_type, shape=(nw * N + 2, N + 2))
xt = ti.field(dtype=ti_data_type, shape=(nw * N + 2, N + 2))
b  = ti.field(dtype=ti_data_type, shape=(nw * N + 2, N + 2))

shadow_left = np.zeros((nw * N + 2))
shadow_right = np.zeros((nw * N + 2))
edge_left = np.zeros((nw * N + 2))
edge_right = np.zeros((nw * N + 2))

@ti.kernel
def extract_edge(edge_left : ti.types.ndarray(), edge_right : ti.types.ndarray()):
    for i in ti.ndrange((0, nw * N + 2)):
        edge_left[i] = x[i, 1]
        edge_right[i] = x[i, N]

@ti.kernel
def fill_shadow(shadow_left : ti.types.ndarray(), shadow_right : ti.types.ndarray()):
    for i in ti.ndrange((0, nw * N + 2)):
        x[i, 0] = shadow_left[i]
        x[i, N+1] = shadow_right[i]


def mpi_transfer_edges():
    extract_edge(edge_left, edge_right)
    if comm.rank > 0:
        comm.Send(edge_left, dest=comm.rank-1)

    if comm.rank < nw - 1:
        comm.Recv(shadow_right, source=comm.rank+1)

    if comm.rank < nw - 1:
        comm.Send(edge_right, dest=comm.rank+1)

    if comm.rank > 0:
        comm.Recv(shadow_left, source=comm.rank-1)

    # # Update Taichi field
    fill_shadow(shadow_left, shadow_right)


def mpi_send_edges():
    req_left = None
    req_right = None
    extract_edge(edge_left, edge_right)
    # x_arr = x.to_numpy()
    # ti.sync()
    # edge_left = np.ascontiguousarray(x_arr[:, 1])
    if comm.rank > 0:
        comm.Isend(edge_left, dest=comm.rank-1, tag=10)
        req_left = comm.Irecv(shadow_left, source=comm.rank - 1, tag=11)

    # edge_right = np.ascontiguousarray(x_arr[:, -2])
    if comm.rank < nw - 1:
        comm.Isend(edge_right, dest=comm.rank+1, tag=11)
        req_right = comm.Irecv(shadow_right, source=comm.rank + 1, tag=10)

    return req_left, req_right

def mpi_recv_edges(req_left, req_right):
    x_arr = x.to_numpy()
    if comm.rank < nw - 1:
        # comm.Irecv(shadow_right, source=comm.rank+1, tag=10).Wait()
        req_right.Wait()
        x_arr[:, -1] = shadow_right

    if comm.rank > 0:
        # comm.Irecv(shadow_left, source=comm.rank - 1, tag=11).Wait()
        req_left.Wait()
        x_arr[:, 0] = shadow_left
    # # Update Taichi field
    # x.from_numpy(x_arr)
    fill_shadow(shadow_left, shadow_right)
    ti.sync()

def mpi_gather_fields(x_np):
    gather_data = comm.gather(x.to_numpy(), root=0)
    if comm.rank == 0:
        for i, buf in enumerate(gather_data):
            x_np[:, i * N : (i+1) * N] = buf[1 : nw * N + 1, 1 : N + 1]

@ti.kernel
def init_fields(rank : ti.i64):
    for I in ti.grouped(x):
        x[I] = 0.0
        xt[I] = 0.0
    for i,j in b:
        xl = (i - 1) / (nw * N)
        yl = (j - 1 + rank * N) / (nw * N)

        if i == 0:
            xl = 0.0
        if i == nw * N + 1:
            yl = 0.0
        if j == N + 1:
            xl = 0.0
        if j == 0:
            yl = 0.0

        b[i,j] = 0.05 * ti.sin(math.pi * xl) * ti.sin(math.pi * yl)


@ti.kernel
def substep():
    for i,j in ti.ndrange((1, nw * N + 1), (1, N+1)):
        xt[i,j] = (b[i,j] + x[i+1,j] + x[i-1,j] + x[i,j+1] + x[i,j-1]) / 4.0
    for I in ti.grouped(x):
        x[I] = xt[I]

@ti.kernel
def substep_bulk():
    for i,j in ti.ndrange((1, nw * N + 1), (2, N)):
        xt[i,j] = (b[i,j] + x[i+1,j] + x[i-1,j] + x[i,j+1] + x[i,j-1]) / 4.0

@ti.kernel
def substep_edge(shadow_left : ti.types.ndarray(), shadow_right : ti.types.ndarray()):
    for i in ti.ndrange((1, nw * N + 1)):
        xt[i,1] = (b[i,1] + x[i+1,1] + x[i-1,1] + x[i,2] + x[i, 0]) / 4.0
        # xt[i,1] = (b[i,1] + x[i+1,1] + x[i-1,1] + x[i,2] + shadow_left[i]) / 4.0
        xt[i,N] = (b[i,N] + x[i+1,N] + x[i-1,N] + x[i,N+1] + x[i,N-1]) / 4.0
        # xt[i,N] = (b[i,N] + x[i+1,N] + x[i-1,N] + shadow_right[i] + x[i,N-1]) / 4.0

    # for i in ti.ndrange((1, nw * N + 1)):
    
    for I in ti.grouped(x):
        x[I] = xt[I]

x_np = None
if comm.rank == 0:
    x_np = np.empty((nw * N, nw * N))
    gui = ti.GUI('Poisson Solver', (nw * N, nw * N))

show_gui = False
init_fields(comm.rank)
i = 0
while True:
    st = time.time()
    # x_arr = x.to_numpy()
    # x_arr = None
    mpi_transfer_edges()
    # rl, rr = mpi_send_edges()
    substep_bulk()
    # mpi_recv_edges(rl, rr)
    substep_edge(shadow_left, shadow_right)
    # substep()
    ti.sync()
    et = time.time()
    if show_gui:
        mpi_gather_fields(x_np)
    if comm.rank == 0:
        print(f"Pure compute FPS {1.0/(et - st)}", flush=True)
        if not gui.running:
            comm.Abort()
            break
        if show_gui:
            x_img = cm.jet(x_np)
            gui.set_image(x_img)
            gui.show()
