# Fluid solver based on lattice boltzmann method using taichi language
# About taichi : https://github.com/taichi-dev/taichi
# Author : Wang (hietwll@gmail.com)

import taichi as ti
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time

from mpi4py import MPI

ti.init(arch=ti.cpu, cpu_max_num_threads=2)


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

        ###############################
        # MPI section
        ###############################
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_worker_x = 4
        self.mpi_worker_y = 1
        assert(self.mpi_comm.size == self.mpi_worker_x * self.mpi_worker_y)
        # 4 * 1 tiling, using MPI Cartesian coordinates.
        self.mpi_cart = self.mpi_comm.Create_cart(dims = [self.mpi_worker_x, self.mpi_worker_y], periods=[False, False])
        self.mpi_rank = self.mpi_comm.rank
        self.mpi_rank_x = self.mpi_cart.Get_coords(self.mpi_rank)[0]
        self.mpi_rank_y = self.mpi_cart.Get_coords(self.mpi_rank)[1]

        self.local_nx = nx // self.mpi_worker_x + 2 # 2 for left and right bounds
        # self.local_nx = nx # 2 for left and right bounds
        self.local_ny = ny
        ###############################

        self.rho = ti.field(dtype=ti.f32, shape=(self.local_nx, self.local_ny))
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(self.local_nx, self.local_ny))
        self.mask = ti.field(dtype=ti.f32, shape=(self.local_nx, self.local_ny))
        self.f_cur = ti.Vector.field(9, dtype=ti.f32, shape=(self.local_nx, self.local_ny))
        self.f_upd = ti.Vector.field(9, dtype=ti.f32, shape=(self.local_nx, self.local_ny))

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
        ###############################
        # Border buffer preallocation
        ###############################
        self.f_l = np.zeros((self.local_nx, 9), dtype=np.float32)
        self.f_r = np.zeros((self.local_nx, 9), dtype=np.float32)
        self.vel_l = np.zeros((self.local_nx, 2), dtype=np.float32)
        self.vel_r = np.zeros((self.local_nx, 2), dtype=np.float32)
        self.rho_l = np.zeros((self.local_nx), dtype=np.float32)
        self.rho_r = np.zeros((self.local_nx), dtype=np.float32)

    def mpi_commit_border_transfer(self):
        self.get_borders(self.f_l, self.rho_l, self.vel_l, self.f_r, self.rho_r, self.vel_r)

        rx = self.mpi_rank_x 
        ry = self.mpi_rank_y

        # Send left border
        if rx > 0:
            self.mpi_comm.isend(self.f_l, dest=self.mpi_cart.Get_cart_rank([rx - 1, ry]), tag=10)
            self.mpi_comm.isend(self.rho_l, dest=self.mpi_cart.Get_cart_rank([rx - 1, ry]), tag=20)
            self.mpi_comm.isend(self.vel_l, dest=self.mpi_cart.Get_cart_rank([rx - 1, ry]), tag=30)

        # Send right border
        if rx < self.mpi_worker_x - 1:
            self.mpi_comm.isend(self.f_r, dest=self.mpi_cart.Get_cart_rank([rx + 1, ry]), tag=11)
            self.mpi_comm.isend(self.rho_r, dest=self.mpi_cart.Get_cart_rank([rx + 1, ry]), tag=21)
            self.mpi_comm.isend(self.vel_r, dest=self.mpi_cart.Get_cart_rank([rx + 1, ry]), tag=31)
            
        reqs = [None] * 6
        # Receive left edge which is the right edge of worker on the left
        if rx > 0:
            reqs[0] = self.mpi_comm.irecv(source=self.mpi_cart.Get_cart_rank([rx - 1, ry]), tag=11)
            reqs[1] = self.mpi_comm.irecv(source=self.mpi_cart.Get_cart_rank([rx - 1, ry]), tag=21)
            reqs[2] = self.mpi_comm.irecv(source=self.mpi_cart.Get_cart_rank([rx - 1, ry]), tag=31)

        # Receive right edge which is the left edge of worker on the right
        if rx < self.mpi_worker_x - 1:
            reqs[3] = self.mpi_comm.irecv(source=self.mpi_cart.Get_cart_rank([rx + 1, ry]), tag=10)
            reqs[4] = self.mpi_comm.irecv(source=self.mpi_cart.Get_cart_rank([rx + 1, ry]), tag=20)
            reqs[5] = self.mpi_comm.irecv(source=self.mpi_cart.Get_cart_rank([rx + 1, ry]), tag=30)
        return reqs

    @ti.kernel
    def set_borders(self, 
                        f_cur_l:ti.types.ndarray(), 
                        rho_l:ti.types.ndarray(), 
                        vel_l:ti.types.ndarray(),
                        f_cur_r:ti.types.ndarray(), 
                        rho_r:ti.types.ndarray(), 
                        vel_r:ti.types.ndarray(),
                        ):
        for j in range(self.local_ny):
            if self.mpi_rank_x > 0:
                for k in ti.static(range(9)):
                    self.f_cur[0, j][k] = f_cur_l[j, k]
                for k in ti.static(range(2)):
                    self.vel[0, j][k] = vel_l[j, k]
                self.rho[0, j] = rho_l[j]

            if self.mpi_rank_x < self.mpi_worker_x - 1:
                for k in ti.static(range(9)):
                    self.f_cur[self.local_nx - 1, j][k] = f_cur_r[j, k]
                for k in ti.static(range(2)):
                    self.vel[self.local_nx - 1, j][k] = vel_r[j, k]
                self.rho[self.local_nx - 1, j] = rho_r[j]

    @ti.kernel
    def get_borders(self, 
                        f_cur_l:ti.types.ndarray(), 
                        rho_l:ti.types.ndarray(), 
                        vel_l:ti.types.ndarray(),
                        f_cur_r:ti.types.ndarray(), 
                        rho_r:ti.types.ndarray(), 
                        vel_r:ti.types.ndarray(),
                        ):
        for j in range(self.local_ny):
            if self.mpi_rank_x > 0:
                for k in ti.static(range(9)):
                    f_cur_l[j, k] = self.f_cur[1, j][k]
                for k in ti.static(range(2)):
                    vel_l[j, k] = self.vel[1, j][k]
                rho_l[j] = self.rho[1, j]

            if self.mpi_rank_x < self.mpi_worker_x - 1:
                for k in ti.static(range(9)):
                    f_cur_r[j, k] = self.f_cur[self.local_nx - 2, j][k]
                for k in ti.static(range(2)):
                    vel_r[j, k] = self.vel[self.local_nx - 2, j][k]
                rho_r[j] = self.rho[self.local_nx - 2, j]

    def mpi_wait_border_transfer(self, reqs):
        for i, req in enumerate(reqs):
            if i == 0 and req != None:
                self.f_l = req.wait()
            elif i == 1 and req != None:
                self.rho_l = req.wait()
            elif i == 2 and req != None:
                self.vel_l = req.wait()
            elif i == 3 and req != None:
                self.f_r = req.wait()
            elif i == 4 and req != None:
                self.rho_r = req.wait()
            elif i == 5 and req != None:
                self.vel_r = req.wait()
        self.set_borders(self.f_l, self.rho_l, self.vel_l, self.f_r, self.rho_r, self.vel_r)

    def mpi_gather_vel(self):
        return self.mpi_comm.gather(self.vel.to_numpy(), root=0)

    def fill_vel_arr(self, gather_data):
        vel_np = np.ndarray([self.nx + 2, self.ny, 2], dtype=np.float32)
        if self.mpi_rank == 0:
            for r, buf in enumerate(gather_data):
                stride_x = self.local_nx - 2
                vel_np[r * stride_x : (r+1) * stride_x + 2, :, :] = buf[0: self.local_nx, :]
        return vel_np

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
                real_i = self.mpi_rank_x * (self.local_nx - 2) + i
                real_j = j
                if ((ti.cast(real_i, ti.f32) - self.cy_para[0])**2.0 + (ti.cast(real_j, ti.f32)
                    - self.cy_para[1])**2.0 <= self.cy_para[2]**2.0):
                    self.mask[i, j] = 1.0

    @ti.kernel
    def collide_and_stream_bulk(self): # lbm core equation
        for i, j in ti.ndrange((1, self.local_nx - 1), (1, self.local_ny - 1)):
            for k in ti.static(range(9)):
                # The nine points around
                p_i = i - self.e[k, 0]
                p_j = j - self.e[k, 1]

                f_cur = self.f_cur[p_i, p_j]
                vel = self.vel[p_i, p_j]
                rho = self.rho[p_i, p_j]

                self.f_upd[i,j][k] = (1.0 - self.inv_tau)*f_cur[k] + \
                                        self.f_eq(vel, rho, k) * self.inv_tau

    @ti.kernel
    def collide_and_stream_border(self): # lbm core equation
        for j in ti.ndrange((1, self.local_ny - 1)):
            for t in ti.static(range(2)): # left and right borders
                i = 0
                run_flg = False
                if t == 0 and self.mpi_rank_x < self.mpi_worker_x - 1:
                    run_flg = True
                elif t == 1 and self.mpi_rank_x > 0:
                    run_flg = True
                    i = self.local_nx - 1
                    
                if run_flg:
                    for k in ti.static(range(9)):
                        # The nine points around
                        p_i = i - self.e[k, 0]
                        p_j = j - self.e[k, 1]

                        f_cur = self.f_cur[p_i, p_j]
                        vel = self.vel[p_i, p_j]
                        rho = self.rho[p_i, p_j]

                        self.f_upd[i,j][k] = (1.0 - self.inv_tau)*f_cur[k] + \
                                                self.f_eq(vel, rho, k) * self.inv_tau


    @ti.kernel
    def update_macro_var(self): # Update rho u v
        for i, j in ti.ndrange((1, self.local_nx - 1), (1, self.local_ny - 1)):
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
        for j in ti.ndrange(1, self.local_ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            if self.mpi_rank_x == 0:
                self.apply_bc_core(True, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            if self.mpi_rank_x == self.mpi_worker_x - 1:
                self.apply_bc_core(True, 2, self.local_nx - 1, j, self.local_nx - 2, j)

        # top and bottom
        for i in ti.ndrange(self.local_nx - 1):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(True, 1, i, self.local_ny - 1, i, self.local_ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(True, 3, i, 0, i, 1)

        # cylindrical obstacle
        # Note: for cuda backend, putting 'if statement' inside loops can be much faster!
        for i, j in ti.ndrange(self.local_nx, self.local_ny):
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
        if self.mpi_rank == 0:
            gui = ti.GUI('lbm solver', (self.nx + 2, 2 * self.ny))
        self.init()
        st = time.perf_counter()
        for i in range(self.steps):
            reqs = self.mpi_commit_border_transfer()
            self.collide_and_stream_bulk()
            self.mpi_wait_border_transfer(reqs)
            self.collide_and_stream_border()
            self.update_macro_var()
            self.apply_bc()
            display_steps = 50
            if i % display_steps == 0:
                et = time.perf_counter()
                t = et - st
                vel_arr = self.mpi_gather_vel()
                if self.mpi_rank == 0:
                    print(f"Performance {1.0 / t * display_steps} steps/s", flush=True)
                    if not gui.running:
                        self.mpi_comm.Abort()
                        break
                    vel = self.fill_vel_arr(vel_arr) 
                    ##  code fragment displaying vorticity is contributed by woclass
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
                st = time.perf_counter()

    def pass_to_py(self):
        return self.vel.to_numpy()[:,:,0]

if __name__ == '__main__':
    lbm = lbm_solver(nx=800, ny=200, niu=0.01, bc_type=[0, 0, 1, 0],
                     bc_value=[[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                     cy=1, cy_para=[160.0, 100.0, 20.0])
    lbm.solve()
