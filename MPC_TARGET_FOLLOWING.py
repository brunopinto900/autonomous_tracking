import matplotlib.pyplot as plt
from matplotlib.pyplot import Rectangle
from matplotlib.patches import Wedge
import numpy as np
import math
from MPC_Quadrotor_ESDF import MPC, HexaCopter
import sys
sys.path.append('./TrajGen')
from trajGen import trajGenerator
import quadrocoptertrajectory as quadtraj
from scipy.interpolate import CubicSpline
import matplotlib
import os
plt.ion()
matplotlib.use("TkAgg")

# TODO
# clean code, organize into classes
# solve large reference value
# fix scale issues

# Global parameters
FOV_D = 6.0
FOV_theta = 40
radius = 0.5
DIST_2_TARGET = FOV_D/2
TRG_HORIZ = 10

fig = plt.figure()

pause = False
def onclick(event):
    global pause
    pause = not pause
    print(pause)
    
fig.canvas.mpl_connect('button_press_event', onclick)

class environment:
    def __init__(self):
#        self.obstacles = np.array([[4.0, 2.0],
#                            [4.0, 1.5],
#                            [4.0, 1.0],
#                            [4.0, 0.5],
#                            [5.0, 5.0],
#                            [5.0, 4.5],
#                            [5.0, 4.0],
#                            [5.0, 3.5],
#                            [5.0, 3.0],
#                            [8.0, 9.0],
#                            [7.5, 9.0],
#                            [7.0, 9.0],
#                            [6.5, 9.0],
#                            [6.0, 9.0],
#                            [5.5, 9.0],
#                            [5.0, 9.0],
#                            [8.0, 10.0],
#                            [9.0, 11.0]
#                            ])
        self.obstacles = np.array( [ [-2,5],[-1.5,5], [-1,5],[-0.5,5], [0,5],[0.5,5],  [1,5], [1.5,5],[2,5]] )
        self.bounds = 15

def plot_refYaw(x,y,yaw):
    yaw = math.degrees(yaw)
    cone = Wedge((x, y), 4, yaw - FOV_theta/8, yaw + FOV_theta/8, ec="none",edgecolor='orange',facecolor='orange')
    plt.gcf().gca().add_artist(cone)

def plot_cone(x,y,yaw):
    yaw = math.degrees(yaw)
    cone = Wedge((x, y), 4, yaw - FOV_theta, yaw + FOV_theta, ec="none",edgecolor='purple',facecolor='purple')
    plt.gcf().gca().add_artist(cone)

def plot_quadrotor(x,y,yaw):
    angle = math.radians(45)
    arm_length = 0.2
    pos_plus = arm_length*math.cos(angle)
    circle = plt.Circle((x+pos_plus, y+pos_plus),0.1,edgecolor='k',facecolor='b',zorder=2)
    plt.gcf().gca().add_artist(circle)
    circle = plt.Circle((x+pos_plus, y-pos_plus),0.1,edgecolor='k',facecolor='b',zorder=2)
    plt.gcf().gca().add_artist(circle)
    circle = plt.Circle((x-pos_plus, y+pos_plus),0.1,edgecolor='k',facecolor='b',zorder=2)
    plt.gcf().gca().add_artist(circle)
    circle = plt.Circle((x-pos_plus, y-pos_plus),0.1,edgecolor='k',facecolor='b',zorder=2)
    plt.gcf().gca().add_artist(circle)
    
def plotobs(x,y,obstacles):
    '''plot all obstacles'''
    l = radius
    w = radius
    color_ = 'k'
    for obs in obstacles:
        dx = x - obs[0]
        dy = y - obs[1]
        r = np.hypot(dx, dy)
        if(r <= FOV_D):
            color_ = 'b'
        else:
            color_ = 'gray'
        
        xloc = obs[0]-l/2
        yloc = obs[1]-w/2
        box_plt = Rectangle((xloc,yloc),l,w,linewidth=1.5,facecolor=color_,zorder = 2)
        plt.gcf().gca().add_artist(box_plt)
        
def plot_obstacles(obstacles):
    for i, _ in enumerate(obstacles):
        obstacle = plt.Circle((obstacles[i][0], obstacles[i][1]), radius, color='r')
        plt.gca().add_patch(obstacle)


def updateObstacleMap(x,y, obstacles):
    observed_obstacles = []

    for i, _ in enumerate(obstacles):
        d = np.hypot(x - obstacles[i,0], y - obstacles[i,1])
        if(d < FOV_D):
            observed_obstacles.append(obstacles[i])
            
    return observed_obstacles

def getNearestObstacle(x,y, obstacles):
    # search nearest obstacle
    if( len(obstacles) == 0): # no obstacles
        return [], -1
        
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(obstacles):
        d = np.hypot(x - obstacles[i][0], y - obstacles[i][1])
        if dmin >= d:
            dmin = d
            minid = i
            
    return [ obstacles[minid] ], minid
            
def getReferenceTrajectoryTarget(targetTraj, yaw, t):
    TrajType = 3 # 1 CIRCLE, 2 SNAP, 3 RAPID
    if(TrajType == 1):
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
        x_ref[8] = np.arctan2(2*np.cos(t) , -2*np.sin(t))
    
    if(TrajType == 2):    
        # state reference
        x_ref = [0] * 12
        # position reference
        x_ref[0] = targetTraj.get_des_state(t).pos[0]
        x_ref[1] = targetTraj.get_des_state(t).pos[1]
        x_ref[2] = 5
        # velocity reference
        x_ref[3] = targetTraj.get_des_state(t).vel[0]
        x_ref[4] = targetTraj.get_des_state(t).vel[1]
        x_ref[5] = 2.0
        x_ref[8] = yaw
        
    if(TrajType == 3):
        # state reference
        x_ref = [0] * 12
        # position reference
        x_ref[0] = targetTraj.get_position(t)[0]
        x_ref[1] = targetTraj.get_position(t)[1]
        x_ref[2] = 5
        # velocity reference
        x_ref[3] = targetTraj.get_velocity(t)[0]
        x_ref[4] = targetTraj.get_velocity(t)[1]
        x_ref[5] = 2.0
        x_ref[8] = yaw
        
    return x_ref
    
    
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
    #x_ref[8] = np.arctan2(2*np.cos(t) , -2*np.sin(t))

    return x_ref

def getSpline(waypoints):
    N = waypoints.shape[0]
    t = np.linspace(0, 7, num=N)
    splineX = CubicSpline(t, waypoints[:,0])
    splineY = CubicSpline(t, waypoints[:,1])
    return splineX,splineY

def getTargetTraj(waypoints):
    polynomial = trajGenerator(waypoints, max_vel = 5)
    return polynomial
    
def generate_single_motion_primitive(pos0,vel0,acc0,posf,velf, t):
    
    traj = quadtraj.RapidTrajectory(pos0, vel0, acc0)
    traj.set_goal_position(posf)
    traj.set_goal_velocity(velf)
    # Run the algorithm, and generate the trajectory.
    Tf = t+1.0
    traj.generate(Tf)
    return traj

def getRefTrajRapid(traj, t):
    position = traj.get_position(t)
    
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
    x_ref[8] = np.arctan2(2*np.cos(t) , -2*np.sin(t))
    
    return position

def getYaw(drone_x, drone_y, target_x, target_y):
     deltaX = target_x - drone_x
     deltaY = target_y - drone_y
     theta = np.arctan2( deltaY, deltaX)
     return theta
 
def getGoalPosition(drone_x, drone_y, target_x, target_y, const_dist):
    deltaX = target_x - drone_x
    deltaY = target_y - drone_y
    deltaD = np.hypot(deltaX, deltaY ) - const_dist
    theta = np.arctan2( deltaY, deltaX)
    droneCONST_x = drone_x + np.cos(theta)*deltaD
    droneCONST_y = drone_y + np.sin(theta)*deltaD
    
    return droneCONST_x, droneCONST_y

def main():
    # Load environment
    env = environment()
    all_obstacles = env.obstacles
    height = 5
    yaw = np.deg2rad(180)
    
    # Load target trajectory
    offline_path = np.load('target_waypoints.npy')
    #targetTraj = getTargetTraj(offline_path)
    splineX, splineY = getSpline(offline_path)
    Tf = 7 #targetTraj.TS[-1]
    target_x = []
    target_y = []
   
    # Define MPC
    x0_val = np.array([0, -3, height, -1, 1, 2, 0, 0, 0, 0, 0, 0])
    t0_val = 0.0
    target_x.append( splineX(t0_val) )
    target_y.append( splineY(t0_val) )
    
    pos0 = np.array([x0_val[0], x0_val[1], x0_val[2] ])
    vel0 = np.array([x0_val[3], x0_val[4], x0_val[5] ])
    velf = vel0
    acc0 = np.array([0, 0, 0 ])
    posfX, posfY = getGoalPosition(x0_val[0], x0_val[1], target_x[-1],target_y[-1], DIST_2_TARGET)
    posf = np.array([posfX,posfY, height ]) # TARGET CURRENT LOCATION SHIFTED WITH RELATIVE DIST
    rapid_traj = generate_single_motion_primitive(pos0,vel0,acc0,posf,velf, t0_val)
    mpc = MPC(  rapid_traj, getReferenceTrajectoryTarget, x0_val, t0_val )
    mpc.setInitialGuess(x0_val, t0_val) # initial optimization
    
    # simulation condition
    T_sim = Tf #6
    sampling_time = 0.05
    N_sim = math.ceil(T_sim / sampling_time)
    ts_sim = np.array([i*sampling_time for i in range(N_sim + 1)])

    xs_sim = []
    us_sim = []
    x = x0_val
    u = np.zeros(mpc.n_u)
    refTrajx = []
    refTrajy = []
    rapid_posX = []
    rapid_posY = []

    ######## Simulation ######## 
    # dynamics
    f = HexaCopter(module='numpy').dynamics
    t = t0_val
    idx = 0
    while(t < Tf):
        
    ######### PLOTTING #########
        plt.cla()
        plt.gca().set_facecolor('xkcd:pale green')
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim( (-4,5))
        plt.ylim( (-6,8))
        
        if not pause:
            pos0 = np.array([x[0], x[1], x[2] ])
            vel0 = np.array([x[3], x[4], x[5] ])
            velf = vel0
            acc0 = np.array([0, 0, 0 ])
            yaw = getYaw(x[0], x[1], target_x[-1],target_y[-1])
            tau = t+TRG_HORIZ*sampling_time
            # Select best goal to track target
            if(tau > Tf):
                target_xGoal = splineX(Tf)
                target_yGoal = splineY(Tf)
            else:
                target_xGoal = splineX(tau) # target_x[-1]
                target_yGoal = splineY(tau) # target_y[-1]
            
            posfX, posfY = getGoalPosition(x[0], x[1],target_xGoal,target_yGoal, DIST_2_TARGET)
            posf = np.array([posfX,posfY, height ]) # TARGET CURRENT LOCATION SHIFTED WITH RELATIVE DIST
            
            rapid_traj = generate_single_motion_primitive(pos0,vel0,acc0,posf,velf, t)
            rapid_posX.append( getRefTrajRapid(rapid_traj, t)[0] )
            rapid_posY.append( getRefTrajRapid(rapid_traj, t)[1] )
            
            xref = getReferenceTrajectoryTarget(rapid_traj, yaw, t)
            refTrajx.append( xref[0] )
            refTrajy.append( xref[1] )
            
            target_x.append( splineX(t) )
            target_y.append( splineY(t) )
            
            observed_obstacles = updateObstacleMap(x[0],x[1], all_obstacles)
            nearest_obstacle, ind = getNearestObstacle(x[0],x[1],observed_obstacles)
            mpc.set_obstacles(nearest_obstacle)
            
            mpc.setTraj( rapid_traj )
            mpc.setTrajFunction( getReferenceTrajectoryTarget )
            mpc.setYaw(yaw)        
                
            u = mpc.solve(x, t)
    
            xs_sim.append(x)
            us_sim.append(u[0])
            
            x_next = x + f(x, u) * sampling_time
            x = x_next
            
            t += sampling_time
            idx = idx + 1
        ######## Simulation ########
            
            
            
            xs_plot = np.array(xs_sim)
            # PLOT TRAJECTORIES
            plt.plot(refTrajx, refTrajy, 'g') # REFERENCE TRAJ (MOTION PRIMITIVE)
            plt.plot(xs_plot[:,0], xs_plot[:,1], 'b--') # DRONE TRAJECTORY
            plt.plot(target_x,target_y, 'r') # TARGET'S TRAJECTORY
            plt.plot(target_x[-1],target_y[-1], '-ro',markersize=10) # TARGET LAST
            plt.plot(posfX, posfY, '-go',markersize=15) # GOAL LAST (Motion primitive)
            plt.legend(['reference','quadrotor', 'target', 'target_LOC', 'goal_LOC'])
            
            # PLOT ENVIRONMENT
            plot_quadrotor(x[0], x[1], x[8])
            plot_cone(x[0],x[1],x[8])
            plot_refYaw(x[0],x[1],yaw)
            plotobs(x[0],x[1],all_obstacles)
           
            #plt.waitforbuttonpress()
            plt.savefig(os.getcwd()+"/images/"+"image"+str(idx)+'.png')
            plt.pause(0.01)
        
        ######### PLOTTING #########
    plt.show()
    plt.pause(1000)
  
if __name__ == '__main__':
    main()
