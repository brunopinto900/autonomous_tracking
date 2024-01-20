import matplotlib.pyplot as plt
import numpy as np
import math

from MPC_Quadrotor import MPC, HexaCopter

def Circle_waypoints(n,Tmax = 2*np.pi):
    t = np.linspace(0,Tmax, n)
    x = 1+0.5*np.cos(t)
    y = 1+0.5*np.sin(t)
    z = 1+0*t
    return np.stack((x, y, z), axis=-1)

def main():
    
    # Define MPC
    mpc = MPC()

    # initial state
    x0 = mpc.opti.parameter(mpc.n_x)
    x0_val = np.array([5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # initial time
    t0_val = 0.0
    mpc.init(x0_val, t0_val)

    # f(x, u)
    f = HexaCopter(module='numpy').dynamics

    # simulation condition
    T_sim = 6
    sampling_time = 0.05
    N_sim = math.ceil(T_sim / sampling_time)
    ts_sim = np.array([i*sampling_time for i in range(N_sim + 1)])

    xs_sim = []
    us_sim = []
    x = x0_val
    u = np.zeros(mpc.n_u)

    # simulation
    for t in ts_sim:
        #if t.is_integer():
            #print('t = ', t)

        u = mpc.solve(x, t)

        xs_sim.append(x)
        us_sim.append(u)

        x_next = x + f(x, u) * sampling_time
        x = x_next

    xs_sim = np.array(xs_sim)
    plt.plot(xs_sim[:,0], xs_sim[:,1])
    #plt.plot( ts_sim[:] - np.pi, 'g') 
    #plt.plot( np.arctan2( np.sin( ts_sim[:] ), np.cos( ts_sim[:] ) ) - np.pi , 'r')
    
    arrow_resol = 0.5 # every half of second
    arrow_time = int( 0.5 / sampling_time ) # 0.5 / 0.05 -> every 10 time instants
    nArrows = int( (xs_sim.shape[0] - 1 ) / arrow_time ) # 200 / 10 = 20

    S = 0
    for i in range( nArrows ):
        yaw = ts_sim[S] - np.pi
        yaw_opt = xs_sim[S,8] - np.pi
        plt.arrow(xs_sim[S,0], xs_sim[S,1], np.cos(yaw), np.sin(yaw), width = 0.05, ec ='green')
        plt.arrow(xs_sim[S,0], xs_sim[S,1], np.cos(yaw_opt), np.sin(yaw_opt), width = 0.05, ec ='red')
        S += arrow_time    

    #plt.plot( np.arctan2( np.sin( ts_sim[:] ), np.cos( ts_sim[:] ) ) - np.pi , 'g')
    #plt.plot( np.arctan2( np.sin( xs_sim[:, 8] ), np.cos( xs_sim[:, 8] ) ) - np.pi , '-r')
    plt.plot(0,0, '-ro',markersize=10) 
    plt.show()
    plt.pause(1000)
  
if __name__ == '__main__':
    main()
