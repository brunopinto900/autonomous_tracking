import matplotlib.pyplot as plt
import numpy as np
import math

from MPC_Quadrotor_ESDF import MPC, HexaCopter

def getReferenceTrajectory(t):
    # state reference
    x_ref = [0] * 12
    # position reference
    x_ref[0] = 5*np.cos(t)
    x_ref[1] = 5*np.sin(t)
    # velocity reference
    x_ref[3] = -5*np.sin(t)
    x_ref[4] = 5*np.cos(t)
    x_ref[5] = 2.0
    # yaw reference
    #yaw = t #- np.pi # always pointing to the center of the circle
    #x_ref[8] = np.arctan2( np.sin(yaw), np.cos(yaw) ) # wrap angle around [-pi,pi]
    return x_ref
    
def main():
    #ox = [1]  # obstacle x position list [m]
    #oy = [4.6]  # obstacle y position list [m]
    
    ox = []
    oy = []
    
    # Define MPC
    mpc = MPC(  getReferenceTrajectory )
    
    # initial state
    x0 = mpc.opti.parameter(mpc.n_x)
    x0_val = np.array([5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # initial time
    t0_val = 0.0
    
    #mpc.setRefTraj( getReferenceTrajectory )
    mpc.init(x0_val, t0_val)
    

    # f(x, u)
    f = HexaCopter(module='numpy').dynamics

    # simulation condition
    T_sim = 7
    sampling_time = 0.05
    N_sim = math.ceil(T_sim / sampling_time)
    ts_sim = np.array([i*sampling_time for i in range(N_sim + 1)])

    xs_sim = []
    us_sim = []
    x = x0_val
    u = np.zeros(mpc.n_u)
    trajx = []
    trajy = []

    # simulation
    for t in ts_sim:
        xref = getReferenceTrajectory(t)
        trajx.append( xref[0] )
        trajy.append( xref[1] )
        
        mpc.setRefTraj( getReferenceTrajectory )
        mpc.set_obstacles(ox,oy)
        u = mpc.solve(x, t)

        xs_sim.append(x)
        us_sim.append(u[0])

        x_next = x + f(x, u) * sampling_time
        x = x_next

    xs_sim = np.array(xs_sim)
    plt.plot(trajx, trajy, 'b')
    plt.plot(xs_sim[:,0], xs_sim[:,1], 'g--')
    
    #plt.plot( ts_sim[:] - np.pi, 'g') 
    #plt.plot( np.arctan2( np.sin( ts_sim[:] ), np.cos( ts_sim[:] ) ) - np.pi , 'r')
    
    arrow_resol = 0.5 # every half of second
    arrow_time = int( 0.5 / sampling_time ) # 0.5 / 0.05 -> every 10 time instants
    nArrows = int( (xs_sim.shape[0] - 1 ) / arrow_time ) # 200 / 10 = 20

#    S = 0
#    for i in range( nArrows ):
#        yaw = ts_sim[S] - np.pi
#        yaw_opt = xs_sim[S,8] - np.pi
#        plt.arrow(xs_sim[S,0], xs_sim[S,1], np.cos(yaw), np.sin(yaw), width = 0.05, ec ='green')
#        plt.arrow(xs_sim[S,0], xs_sim[S,1], np.cos(yaw_opt), np.sin(yaw_opt), width = 0.05, ec ='red')
#        S += arrow_time    

    #plt.plot( np.arctan2( np.sin( ts_sim[:] ), np.cos( ts_sim[:] ) ) - np.pi , 'g')
    #plt.plot( np.arctan2( np.sin( xs_sim[:, 8] ), np.cos( xs_sim[:, 8] ) ) - np.pi , '-r')
    plt.plot(0,0, '-ro',markersize=10) 
    # plot obstacles
    #print("plotting obstacles")
    #obstacle = plt.Circle((ox[0], oy[0]), 0.5, color='k')
    #print("plotting obstacles")
    #plt.gca().add_patch(obstacle)
    plt.show()
    plt.pause(1000)
  
if __name__ == '__main__':
    main()
