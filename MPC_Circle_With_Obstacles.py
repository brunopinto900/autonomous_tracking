import matplotlib.pyplot as plt
import numpy as np
import math
from MPC_Quadrotor_ESDF import MPC, HexaCopter

# TODO
# solve issue with large reference

def plot_obstacles(obsX, obsY):
    for i, _ in enumerate(obsX):
        obstacle = plt.Circle((obsX[i], obsY[i]), 0.5, color='r')
        plt.gca().add_patch(obstacle)

#def plot_obstacles(obsX, obsY, ind):
#    if(ind != -1):
#        if(ind == 0):
#            obstacle1 = plt.Circle((obsX[0], obsY[0]), 0.5, color='r')
#        else:
#            obstacle1 = plt.Circle((obsX[0], obsY[0]), 0.5, color='k')
#        
#        if(ind == 1):
#            obstacle2 = plt.Circle((obsX[1], obsY[1]), 0.5, color='r')
#        else:
#            obstacle2 = plt.Circle((obsX[1], obsY[1]), 0.5, color='k')
#            
#        plt.gca().add_patch(obstacle1)
#        plt.gca().add_patch(obstacle2)

FOV_D = 6
def updateObstacleMap(x,y, obsX, obsY):
    obvsObsX = []
    obvsObsY = []
    for i, _ in enumerate(obsX):
        d = np.hypot(x - obsX[i], y - obsY[i])
        if(d < FOV_D):
            obvsObsX.append(obsX[i])
            obvsObsY.append(obsY[i])

    return obvsObsX, obvsObsY

def getNearestObstacle(x,y, obsX, obsY):
    # search nearest obstacle
    if( len(obsX) == 0): # no obstacles
        return [], [], -1
        
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(obsX):
        d = np.hypot(x - obsX[i], y - obsY[i])
        if dmin >= d:
            dmin = d
            minid = i
            
    return [ obsX[minid] ], [ obsY[minid] ], minid
            

def getReferenceTrajectory(t):
    # state reference
    x_ref = [0] * 12
    # position reference
    x_ref[0] = 5*np.cos(t)
    x_ref[1] = 5*np.sin(t)
    x_ref[2] = 5
    # velocity reference
    x_ref[3] = -2*np.sin(t)
    x_ref[4] = 2*np.cos(t)
    x_ref[5] = 2.0

    return x_ref
    
def main():
    obsX = [0, 2, 0]  # obstacle x position list [m]
    obsY = [-5.0, -5, 5]  # obstacle y position list [m]
    
    #ox = []
    #oy = []
    x0_val = np.array([5, 0, 5, -1, 1, 2, 0, 0, 0, 0, 0, 0])
    # initial time
    t0_val = 0.0
    
    # Define MPC
    mpc = MPC(  getReferenceTrajectory, x0_val, t0_val )
    
    # initial state
    x0 = mpc.opti.parameter(mpc.n_x)
    
    mpc.setInitialGuess(x0_val, t0_val)
    
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
    trajx = []
    trajy = []

    # simulation
    for t in ts_sim:
        xref = getReferenceTrajectory(t)
        trajx.append( xref[0] )
        trajy.append( xref[1] )
        
        mpc.setRefTraj( getReferenceTrajectory )
        ObvsObsX,ObvsObsY = updateObstacleMap(x[0],x[1], obsX, obsY)
        ox, oy, ind = getNearestObstacle(x[0],x[1],ObvsObsX,ObvsObsY)
        mpc.set_obstacles(ox, oy)
        u = mpc.solve(x, t)

        xs_sim.append(x)
        us_sim.append(u[0])

        x_next = x + f(x, u) * sampling_time
        x = x_next
       
        plot_obstacles(ObvsObsX,ObvsObsY)
        
        xs_plot = np.array(xs_sim)
        plt.plot(trajx, trajy, 'b')
        plt.plot(xs_plot[:,0], xs_plot[:,1], 'g--')
        plt.plot(0,0, '-ro',markersize=10) 
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        plt.pause(0.01)
        plt.xlim( (-6,6))
        plt.ylim( (-6,6))
    # plot obstacles
    plt.pause(1000)
  
if __name__ == '__main__':
    main()
