import taichi as ti
import math
import time
import matplotlib.cm as cm


N = 256
show_gui = True
steps = 50
if show_gui:
    steps = 1
ti_data_type=ti.f64

ti.init(arch=ti.gpu,default_fp=ti_data_type, offline_cache=False, device_memory_GB=6, packed=True)

x = ti.field(dtype=ti_data_type, shape=(N + 2, N + 2))
xt = ti.field(dtype=ti_data_type, shape=(N + 2, N + 2))
b = ti.field(dtype=ti_data_type, shape=(N + 2, N + 2))

@ti.kernel
def init():
    for I in ti.grouped(x):
        x[I] = 0.0
        xt[I] = 0.0
    for i,j in b:
        xl = i / N
        yl = j / N

        if i == 0:
            xl = 0.0
        if i == N + 1:
            xl = 0.0
        if j == 0:
            yl = 0.0
        if j == N + 1:
            yl = 0.0

        b[i,j] = 0.5 * ti.sin(math.pi * xl) * ti.sin(math.pi * yl)

@ti.kernel
def enforce_bc():
    pass

@ti.kernel
def substep():
    for i,j in ti.ndrange((1, N+1), (1, N+1)):
        xt[i,j] = (b[i,j] + x[i+1,j] + x[i-1,j] + x[i,j+1] + x[i,j-1]) / 4.0
    for I in ti.grouped(x):
        x[I] = xt[I]

@ti.kernel
def residual()->ti.f64:
    sum = 0.0
    for i,j in r:
        r[i,j] = (4*x[i,j] - x[i-1,j] - x[i+1,j] - x[i,j-1] - x[i,j+1] - b[i,j])
        sum += r[i,j] ** 2
    return ti.sqrt(sum)

if show_gui:
    gui = ti.GUI('Poisson Solver', (N,N))
init()

st = time.time()
i = 0
while True:
    enforce_bc()
    substep()
    ti.sync()
    if i % steps == 0:
        et = time.time()
        if show_gui:
            x_np = x.to_numpy()
            x_img = cm.jet(x_np[1:N+1, 1:N+1])
            gui.set_image(x_img)
            gui.show()
        else:
            print(f"Pure compute FPS {steps/(et - st)}", flush=True)
        i = 0
        st = time.time()    
    i += 1
