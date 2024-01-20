import matplotlib.pyplot as plt
import numpy as np
import math

from cubic_spline_planner import Spline2D

from rrt_star import RRTStar
from MPC_Quadrotor import MPC, HexaCopter

def main():
    
    # Define MPC
    mpc = MPC()

    # initial state
    x0 = mpc.opti.parameter(mpc.n_x)
    x0_val = np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # initial time
    t0_val = 0.0
    mpc.init(x0_val, t0_val)

    # f(x, u)
    f = HexaCopter(module='numpy').dynamics

    # simulation condition
    T_sim = 1.0
    sampling_time = 0.05
    N_sim = math.ceil(T_sim / sampling_time)
    ts_sim = np.array([i*sampling_time for i in range(N_sim + 1)])

    xs_sim = []
    us_sim = []
    x = np.zeros(mpc.n_x)
    u = np.zeros(mpc.n_u)


    # ====Search Path with RRT====
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
        (6, 12, 1),
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    rrt_star = RRTStar(
        start=[0, 0],
        goal=[6, 10],
        rand_area=[-2, 15],
        obstacle_list=obstacle_list,
        expand_dis=5,
        robot_radius=0.5)
    path = rrt_star.planning(animation=False)
    path_np = np.array(path)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

    print("Spline 2D test")
    
    x_wp = path_np[:,0].tolist()
    y_wp = path_np[:,1].tolist()

    ds = 0.1  # [m] distance of each intepolated points

    sp = Spline2D(x_wp, y_wp)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, = [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)

    # simulation
    for t in ts_sim:
        if t.is_integer():
            print('t = ', t)

        u = mpc.solve(x, t)

        xs_sim.append(x)
        us_sim.append(u)

        x_next = x + f(x, u) * sampling_time
        x = x_next

    xs_sim = np.array(xs_sim)
    #print(xs_sim.shape)
    plt.plot(xs_sim[:,0],xs_sim[:,1])
    plt.show()
    plt.pause(1000)
        
    # plt.subplots(1)
    # plt.plot(x, y, "xb", label="input")
    # plt.plot(rx, ry, "-r", label="spline")
    # plt.grid(True)
    # plt.axis("equal")
    # plt.xlabel("x[m]")
    # plt.ylabel("y[m]")
    # plt.legend()

    # plt.show()

  
if __name__ == '__main__':
    main()
