# Fluid solver based on lattice boltzmann method using taichi language
# About taichi : https://github.com/taichi-dev/taichi
# Author : Wang (hietwll@gmail.com)

import taichi as ti
import numpy as np
import time
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from mpi4py import MPI

ti.init(arch=ti.cpu)


@ti.data_oriented
class lbm_solver:
    def __init__(self,
                 nx, # domain size
                 ny,
                 niu, # viscosity of fluid
                 bc_type, # [left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
                 bc_value, # if bc_type = 0, we need to specify the velocity in bc_value
                 cy = 0, # whether to place a cylindrical obstacle
                 cy_para = [0.0, 0.0, 0.0], # location and radius of the cylinder
                 steps = 60000): # total steps to run
        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau
        self.rho = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.mask = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.f_cur = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f_upd = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.bc_type = ti.field(dtype=ti.i32, shape=4)
        self.bc_value = ti.field(dtype=ti.f32, shape=(4, 2))
        self.cy = cy
        self.cy_para = ti.field(dtype=ti.f32, shape=3)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))
        self.cy_para.from_numpy(np.array(cy_para, dtype=np.float32))
        self.steps = steps

        self.w = ti.Matrix([ 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dt=ti.f32)

        self.e = ti.Matrix([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]], dt=ti.i32)


    @ti.func # compute equilibrium distribution function
    def f_eq_coord(self, i, j, k:ti.template()):
        eu = ti.cast(self.e[k, 0], ti.f32) * self.vel[i, j][0] + ti.cast(self.e[k, 1],
            ti.f32) * self.vel[i, j][1]
        uv = self.vel[i, j][0]**2.0 + self.vel[i, j][1]**2.0
        return self.w[k] * self.rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * uv)

    @ti.func
    def f_eq(self, vel, rho, k:ti.template()):
        eu = self.e[k, 0] * vel[0] + self.e[k, 1] * vel[1]
        uv = vel[0]**2.0 + vel[1]**2.0
        return self.w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * uv)

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            self.rho[i, j] = 1.0
            self.mask[i, j] = 0.0
            for k in ti.static(range(9)):
                vel = self.vel[i, j]
                rho = 1.0
                self.f_upd[i, j][k] = self.f_eq(vel, rho, k)
                self.f_cur[i, j][k] = self.f_upd[i, j][k]
            # Should make the entire cylindar reside inside a single worker.
            if(self.cy==1):
                if ((ti.cast(i, ti.f32) - self.cy_para[0])**2.0 + (ti.cast(j, ti.f32)
                    - self.cy_para[1])**2.0 <= self.cy_para[2]**2.0):
                    self.mask[i, j] = 1.0

    @ti.kernel
    def collide_and_stream(self): # lbm core equation
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                # The nine points around
                p_i = i - self.e[k, 0]
                p_j = j - self.e[k, 1]

                f_cur = self.f_cur[p_i, p_j]
                vel = self.vel[p_i, p_j]
                rho = self.rho[p_i, p_j]

                self.f_upd[i,j][k] = (1.0-self.inv_tau)*f_cur[k] + \
                                        self.f_eq(vel, rho, k) * self.inv_tau


    @ti.kernel
    def update_macro_var(self): # Update rho u v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            vel_x = 0.0
            vel_y = 0.0
            rho = 0.0
            for k in ti.static(range(9)):
                self.f_cur[i, j][k] = self.f_upd[i, j][k]
                rho += self.f_upd[i, j][k]
                vel_x += self.e[k, 0] * self.f_upd[i, j][k]
                vel_y += self.e[k, 1] * self.f_upd[i, j][k]
            vel_x /= rho
            vel_y /= rho
            self.rho[i, j] = rho
            self.vel[i, j][0] = vel_x
            self.vel[i, j][1] = vel_y

    @ti.kernel
    def apply_bc(self): # impose boundary conditions
        # left and right
        for j in ti.ndrange(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(True, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(True, 2, self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in ti.ndrange(self.nx):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(True, 1, i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(True, 3, i, 0, i, 1)

        # cylindrical obstacle
        # Note: for cuda backend, putting 'if statement' inside loops can be much faster!
        for i, j in ti.ndrange(self.nx, self.ny):
            if (self.cy == 1 and self.mask[i, j] == 1):
                self.vel[i, j][0] = 0.0  # velocity is zero at solid boundary
                self.vel[i, j][1] = 0.0
                inb = 0
                jnb = 0
                if (ti.cast(i,ti.f32) >= self.cy_para[0]):
                    inb = i + 1
                else:
                    inb = i - 1
                if (ti.cast(j,ti.f32) >= self.cy_para[1]):
                    jnb = j + 1
                else:
                    jnb = j - 1
                self.apply_bc_core(False, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, bc_i, bc_j, nb_i, nb_j):
        if (outer):  # handle outer boundary
            if (self.bc_type[dr] == 0):
                self.vel[bc_i, bc_j][0] = self.bc_value[dr, 0]
                self.vel[bc_i, bc_j][1] = self.bc_value[dr, 1]
            elif (self.bc_type[dr] == 1):
                self.vel[bc_i, bc_j][0] = self.vel[nb_i, nb_j][0]
                self.vel[bc_i, bc_j][1] = self.vel[nb_i, nb_j][1]
        self.rho[bc_i, bc_j] = self.rho[nb_i, nb_j]
        for k in ti.static(range(9)):
            vel_bc = self.vel[bc_i, bc_j]
            vel_nb = self.vel[nb_i, nb_j]
            rho_bc = self.rho[bc_i, bc_j]
            rho_nb = self.rho[nb_i, nb_j]
            d = self.f_eq(vel_bc, rho_bc, k) - self.f_eq(vel_nb, rho_nb, k)
            self.f_cur[bc_i,bc_j][k] = d + self.f_cur[nb_i,nb_j][k]

    def solve(self):
        gui = ti.GUI('lbm solver', (self.nx, 2 * self.ny))
        self.init()
        st = time.perf_counter()
        for i in range(self.steps):
            self.collide_and_stream()
            self.update_macro_var()
            self.apply_bc()
            display_steps = 50
            if (i % display_steps == 0):
                # visualize every 50 steps
                ##  code fragment displaying vorticity is contributed by woclass
                et = time.perf_counter()
                t = et - st
                st = time.perf_counter()
                print(f"Performance {display_steps * 1.0 / t} steps/s")
                vel = self.vel.to_numpy()
                ugrad = np.gradient(vel[:, :, 0])
                vgrad = np.gradient(vel[:, :, 1])
                vor = ugrad[1] - vgrad[0]
                vel_mag = (vel[:, :, 0]**2.0+vel[:, :, 1]**2.0)**0.5
                ## color map
                colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0),
                    (0.176, 0.976, 0.529), (0, 1, 1)]
                my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'my_cmap', colors)
                vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(
                    vmin=-0.02, vmax=0.02),cmap=my_cmap).to_rgba(vor)
                vel_img = cm.plasma(vel_mag / 0.15)
                img = np.concatenate((vor_img, vel_img), axis=1)
                gui.set_image(img)
                gui.show()
                print('Step: {:}'.format(i))
                # ti.imwrite((img[:,:,0:3]*255).astype(np.uint8), 'fig/karman_'+str(i).zfill(6)+'.png')

    def pass_to_py(self):
        return self.vel.to_numpy()[:,:,0]

if __name__ == '__main__':
    lbm = lbm_solver(nx=801, ny=201, niu=0.01, bc_type=[0, 0, 1, 0],
                     bc_value=[[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                     cy=1, cy_para=[160.0, 100.0, 20.0])
    lbm.solve()
