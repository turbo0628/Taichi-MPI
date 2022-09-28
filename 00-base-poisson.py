import taichi as ti
import math
import time
import matplotlib.cm as cm

ti.init(arch=ti.cpu,default_fp=ti.f64, offline_cache=False)

N = 1024

x = ti.field(dtype=ti.f64, shape=(N + 2, N + 2))
xt = ti.field(dtype=ti.f64, shape=(N + 2, N + 2))
b = ti.field(dtype=ti.f64, shape=(N + 2, N + 2))
r = ti.field(dtype=ti.f64, shape=(N + 2, N + 2))

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

        b[i,j] = 0.05 * ti.sin(math.pi * xl) * ti.sin(math.pi * yl)

@ti.kernel
def enforce_bc():
    pass

@ti.kernel
def iter():
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

gui = ti.GUI('Poisson Solver', (N,N))
init()

while gui.running:
    st = time.time()
    enforce_bc()
    iter()
    et = time.time()
    print(f"Pure compute FPS {1.0/(et - st)}", flush=True)
    # print(f"B after iter {i}\n", b)
    # print(f"X after iter {i}\n", x)
    # r = residual()
    # print(f'Residual = {r:4.2f}')
    x_np = x.to_numpy()
    x_img = cm.jet(x_np[1:N+1, 1:N+1])
    gui.set_image(x_img)
    gui.show()
