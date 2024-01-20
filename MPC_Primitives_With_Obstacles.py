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
    x_ref[3] = -1*np.sin(t)
    x_ref[4] = 1*np.cos(t)
    x_ref[5] = 1.0

    return x_ref
    
def main():
    obsX = [0, 2, 0]  # obstacle x position list [m]
    obsY = [-5.0, -5, 5]  # obstacle y position list [m]
    
    # Define MPC
    # initial state & time
    x0_val = np.array([5, 0, 5, -1, 1, 1, 0, 0, 0, 0, 0, 0])
    t0_val = 0.0
    mpc = MPC(  getReferenceTrajectory, x0_val, t0_val)
    
    
    mpc.setInitialGuess(x0_val, t0_val)
    
    # f(x, u)
    f = HexaCopter(module='numpy').dynamics
    # Initial conditions
    x = x0_val
    u = np.zeros(mpc.n_u)

    # simulation condition
    T_sim = 7
    sampling_time = 0.05
    N_sim = math.ceil(T_sim / sampling_time)
    ts_sim = np.array([i*sampling_time for i in range(N_sim + 1)])

    # For plotting
    plot_x = []
    plot_y = []
    us_sim = []
    trajx = []
    trajy = []
    
    plt.plot(0,0, '-ro',markersize=10) 
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim( (-6,6))
    plt.ylim( (-6,6))
        
    # simulation
    for t in ts_sim:
        xref = getReferenceTrajectory(t)
        trajx.append( xref[0] )
        trajy.append( xref[1] )
        
        # Update obstacle map
        ObvsObsX,ObvsObsY = updateObstacleMap(x[0],x[1], obsX, obsY)
        ox, oy, ind = getNearestObstacle(x[0],x[1],ObvsObsX,ObvsObsY)
        mpc.set_obstacles(ox, oy)
        
        # Calculate reference trajectory
        mpc.setRefTraj( getReferenceTrajectory )
        
        # Solve
        u = mpc.solve(x, t)

        # For plot
        plot_x.append(x[0])
        plot_y.append(x[1])
        us_sim.append(u[0])

        plot_obstacles(ObvsObsX,ObvsObsY)
        
        plt.plot(trajx, trajy, 'b')
        plt.plot(plot_x, plot_y, 'g--')
      
        plt.show()
        plt.pause(0.001)
        
        # Update state (system dynamics)
        x_next = x + f(x, u) * sampling_time
        x = x_next
        
    plt.pause(1000)
  
if __name__ == '__main__':
    main()
