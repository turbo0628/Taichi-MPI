import taichi as ti
import math
import matplotlib.cm as cm
from mpi4py import MPI
import numpy as np
import time
import argparse


@ti.data_oriented
class PoissonSolver():
    def __init__(self, N = 1024, ti_arch=ti.cpu, ti_data_type=ti.f64):
        ti.init(arch=ti_arch, default_fp=ti_data_type, offline_cache=False, device_memory_GB=6, packed=True)
        self.comm = MPI.COMM_WORLD
        self.N = N
        self.dx = 1.0 / (self.N + 1)
        self.nw = self.comm.size # num workers
        self.n = N // self.nw # frame edge length of the local field
        assert(N % self.nw == 0)
        
        if ti_data_type == ti.f32:
            self.np_data_type = np.float32
        elif ti_data_type == ti.f64:
            self.np_data_type = np.float64
        else:
            raise RuntimeError(f"Illegal data type {ti_data_type}")

        # Local fields in each MPI worker
        self.x  = ti.field(dtype=ti_data_type, shape=(self.N + 2, self.n + 2))
        self.xt = ti.field(dtype=ti_data_type, shape=(self.N + 2, self.n + 2))
        self.b  = ti.field(dtype=ti_data_type, shape=(self.N + 2, self.n + 2))

        self.shadow_left = np.zeros((self.N + 2), dtype=self.np_data_type)
        self.shadow_right = np.zeros((self.N + 2), dtype=self.np_data_type)
        self.edge_left = np.zeros((N + 2), dtype=self.np_data_type)
        self.edge_right = np.zeros((N + 2), dtype=self.np_data_type)

    @ti.kernel
    def extract_edge(self, edge_left : ti.types.ndarray(), edge_right : ti.types.ndarray()):
        for i in ti.ndrange((0, self.N + 2)):
            edge_left[i] = self.x[i, 1]
            edge_right[i] = self.x[i, self.n]

    @ti.kernel
    def fill_shadow(self, shadow_left : ti.types.ndarray(), shadow_right : ti.types.ndarray()):
        for i in ti.ndrange((0, self.N + 2)):
            self.x[i, 0] = shadow_left[i]
            self.x[i, self.n+1] = shadow_right[i]

    def mpi_transfer_edges(self):
        comm = self.comm
        self.extract_edge(self.edge_left, self.edge_right)
        if comm.rank > 0:
            comm.Send(self.edge_left, dest=comm.rank-1)

        if comm.rank < self.nw - 1:
            comm.Recv(self.shadow_right, source=comm.rank+1)

        if comm.rank < self.nw - 1:
            comm.Send(self.edge_right, dest=comm.rank+1)

        if comm.rank > 0:
            comm.Recv(self.shadow_left, source=comm.rank-1)

        # Update Taichi field
        self.fill_shadow(self.shadow_left, self.shadow_right)

    def mpi_gather_fields(self, x_np):
        gather_data = self.comm.gather(self.x.to_numpy(), root=0)
        if self.comm.rank == 0:
            for i, buf in enumerate(gather_data):
                x_np[:, i * self.n : (i+1) * self.n] = buf[1 : self.N + 1, 1 : self.n + 1]

    @ti.kernel
    def init_fields(self, rank : ti.i64):
        for I in ti.grouped(self.x):
            self.x[I] = 0.0
            self.xt[I] = 0.0
        for i,j in self.b:
            xl = (i - 1) / self.N
            yl = (j - 1 + rank * self.n) / self.N

            if i == 0:
                xl = 0.0
            if i == self.N + 1:
                yl = 0.0
            if j == self.n + 1:
                xl = 0.0
            if j == 0:
                yl = 0.0

            self.b[i,j] = ti.sin(math.pi * xl) * ti.sin(math.pi * yl)


    @ti.kernel
    def substep(self):
        for i,j in ti.ndrange((1, self.N + 1), (1, self.n + 1)):
            self.xt[i,j] = (-self.b[i,j]*self.dx**2 + self.x[i+1,j] + self.x[i-1,j] + self.x[i,j+1] + self.x[i,j-1]) / 4.0
        for I in ti.grouped(self.x):
            self.x[I] = self.xt[I]

    def step(self):
        self.mpi_transfer_edges()
        self.substep()
        ti.sync()

def main(N, ti_arch, ti_data_type, show_gui, steps_interval):
    x_np = None
    comm = MPI.COMM_WORLD
    if show_gui and comm.rank == 0:
        x_np = np.empty((N, N))
        gui = ti.GUI('Poisson Solver', (N,N))

    solver = PoissonSolver(N, ti_arch, ti_data_type)
    solver.init_fields(comm.rank)
    st = time.time()
    while True:
        for i in range(steps_interval):
            solver.step()
        et = time.time()

        if comm.rank == 0 and show_gui and not gui.running:
            comm.Abort()

        if show_gui:
            solver.mpi_gather_fields(x_np)
            if comm.rank == 0:
                x_min = np.min(x_np)
                x_max = np.max(x_np)                
                x_img = cm.jet(abs(x_np / (x_max - x_min)))
                gui.set_image(x_img)
                gui.show()
        else:
            if comm.rank == 0:
                print(f"Pure compute FPS {steps_interval/(et - st)}", flush=True)
        st = time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Taichi + MPI Poisson solver demo')
    parser.add_argument('-n', dest='N', type=int, default=1024)
    parser.add_argument('--fp32',  action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    show_gui = True
    ti_arch = ti.gpu
    if args.cpu:
        ti_arch = ti.cpu
    ti_data_type = ti.f64
    steps_interval = 1
    if args.fp32:
        ti_data_type = ti.f32
    if args.benchmark:
        show_gui = False
        steps_interval = 50
    main(args.N, ti_arch, ti_data_type, show_gui, steps_interval)
