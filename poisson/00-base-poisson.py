import taichi as ti
import math
import time
import matplotlib.cm as cm
import numpy as np
import argparse

@ti.data_oriented
class PoissonSolver():
    def __init__(self, N = 1024, ti_data_type=ti.f64):
        ti.init(arch=ti.gpu,default_fp=ti_data_type, offline_cache=False, device_memory_GB=6, packed=True)
        self.N = N
        self.dx = 1.0 / (self.N + 1)
        self.x = ti.field(dtype=ti_data_type, shape=(N + 2, N + 2))
        self.xt = ti.field(dtype=ti_data_type, shape=(N + 2, N + 2))
        self.b = ti.field(dtype=ti_data_type, shape=(N + 2, N + 2))

    @ti.kernel
    def init(self):
        for I in ti.grouped(self.x):
            self.x[I] = 0.0
            self.xt[I] = 0.0
        for i,j in self.b:
            xl = i / self.N
            yl = j / self.N
            self.b[i,j] = ti.sin(math.pi * xl) * ti.sin(math.pi * yl)

    @ti.kernel
    def substep(self):
        for i,j in ti.ndrange((1, self.N+1), (1, self.N+1)):
            self.xt[i,j] = (-self.b[i,j]*self.dx**2 + self.x[i+1,j] + self.x[i-1,j] + self.x[i,j+1] + self.x[i,j-1]) / 4.0
        for I in ti.grouped(self.x):
            self.x[I] = self.xt[I]

    def step(self):
        self.substep()
        ti.sync()
    
def main(N = 1024, ti_data_type=ti.f64, show_gui=True, steps_interval=1):
    if show_gui:
        gui = ti.GUI('Poisson Solver', (N,N))
    solver = PoissonSolver(N, ti_data_type)
    solver.init()
    st = time.time()
    while True:
        for i in range(steps_interval):
            solver.step()
        et = time.time()
        if show_gui:
            x_np = solver.x.to_numpy()
            ratio = 2000.  # Adjust the field level for proper displaying
            x_img = cm.jet(abs(x_np[1:N+1, 1:N+1] * ratio))
            gui.set_image(x_img)
            gui.show()
        else:
            print(f"Pure compute FPS {steps_interval/(et - st)}", flush=True)
        st = time.time()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Taichi + MPI Poisson solver demo')
    parser.add_argument('-n', dest='N', type=int, default=1024)
    parser.add_argument('--fp32',  action='store_true')
    parser.add_argument('--benchmark', action='store_true')

    args = parser.parse_args()
    show_gui = True
    data_type = ti.f64
    steps_interval = 1
    if args.fp32:
        data_type = ti.f32
    if args.benchmark:
        show_gui = False
        steps_interval = 50
    main(args.N, data_type, show_gui, steps_interval)
