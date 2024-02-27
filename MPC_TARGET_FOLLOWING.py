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
from math import fabs, pi

# TODO
# clean code, organize into classes
# solve large reference value
# fix scale issues

# Global parameters
FOV_D = 6.0
FOV_theta = 40
radius = 0.5
DIST_2_TARGET = FOV_D/2
DS = DIST_2_TARGET
TRG_HORIZ = 5

fig = plt.figure()

def distPoint2Vector(p1, p2, p3):
    return np.abs(np.linalg.norm(np.cross(p2-p1, p1-p3)))/np.linalg.norm(p2-p1) 

def getGoalPoint(Wc, T, obstacles):
    
    nObs = len(obstacles)
    if(nObs > 0):
        lambda_ = 0.5 / nObs
    else:
        lambda_ = 0.5
    
    lambda_ = 2.0 # 2.0
    dm = 0.02*0 #math.sin(FOV_theta)*(DS/2) # 0.02
    theta_discr = np.linspace(-math.pi,math.pi, int(math.pi*10) )
    costs = []
    discrete_wp = []
    
    for theta in theta_discr:
        cost = 0
        candidate_wp = np.array( [DS*np.cos(theta) + T[0], DS*np.sin(theta) + T[1] ] )
        discrete_wp.append(candidate_wp)
        
        l1 = np.hypot( candidate_wp[0] - Wc[0], candidate_wp[1] - Wc[1] )
        cost += l1
        
        midx = (T[0] + candidate_wp[0] ) / 2
        midy = (T[1] + candidate_wp[1] ) / 2
    
        for ob in obstacles:
            p1 = np.array( [T[0], T[1] ] ) # point from bearing vector
            p2 = np.array( [candidate_wp[0], candidate_wp[1] ] ) # point from bearing vector
            p3 = np.array([ob[0], ob[1] ] ) # candidate goalpoint
            
            l2 = distPoint2Vector(p1, p2, p3)
            #l2 = np.hypot(  midx - ob[0] , midy - ob[1] )
            #dis2toObs = np.hypot(  p2[0] - ob[0] , p2[1] - ob[1] )
            cost +=  lambda_* math.exp( -(l2+dm) )
        
        costs.append(cost)
    
    costs_arr = np.array(costs)
    idx = np.argmin(costs_arr)
    
    Wg = discrete_wp[idx]
    
    return Wg

def sign(x):
    if x < 0:
        return -1
    if x == 0:
        return 0
    else:
        return 1
    
def truncated_remainder(dividend, divisor):
    divided_number = dividend / divisor
    divided_number = \
        -int(-divided_number) if divided_number < 0 else int(divided_number)

    remainder = dividend - divisor * divided_number

    return remainder

def transform_to_pipi(input_angle):
    p1 = truncated_remainder(input_angle + np.sign(input_angle) * pi, 2 * pi)
    p2 = (np.sign(np.sign(input_angle)
                  + 2 * (np.sign(fabs((truncated_remainder(input_angle + pi, 2 * pi))
                                      / (2 * pi))) - 1))) * pi
    output_angle = p1 - p2

    return output_angle

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
        self.obstacles = np.array( [ [-2,5],[-1.5,5], [-1,5],[-0.5,5], [0,5],
                                    [0.5,5],  [1,5], [1.5,5],[2,5], [6,5] ] ) 
                                    # [2.5, 1.0],[3.0, 1.0], [3.5, 1.0]  ] )
        self.bounds = 15

def plot_tangent(x,y,yaw):
    yaw = math.degrees(yaw)
    cone = Wedge((x, y), 4, yaw - FOV_theta/16, yaw + FOV_theta/16, ec="none",edgecolor='black',facecolor='black')
    plt.gcf().gca().add_artist(cone)
    
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
    x_ref[5] = 0.0
    x_ref[8] = np.arctan2(2*np.cos(t) , -2*np.sin(t))
    
    return position

def getYaw(x1, y1, x2, y2):
     deltaX = x2 - x1
     deltaY = y2 - y1
     theta = np.arctan2( deltaY, deltaX)
     return theta

def getGoalPosition(obstacle, drone_x, drone_y, target_x, target_y, const_dist, 
                    target_lastX, target_lastY, previousTan):
    
    USE_TAN = True
    deltaX = target_x - drone_x
    deltaY = target_y - drone_y
    theta = np.arctan2( deltaY, deltaX)
    deltaD = np.hypot(deltaX, deltaY )
    droneCONST_x = drone_x + np.cos(theta)*(deltaD - const_dist)
    droneCONST_y = drone_y + np.sin(theta)*(deltaD - const_dist)
        
#    if(len(obstacle) > 0):
#        # Detect collision LOS with obstacles
#        discret_dist = np.linspace(0, deltaD, int(deltaD*10))
#        ob_x = obstacle[0][0]
#        ob_y = obstacle[0][1]  
#        for d in discret_dist:
#            diffX = ob_x - droneCONST_x + np.cos(theta)*d
#            diffY = ob_y - droneCONST_y + np.sin(theta)*d
#            plt.plot(ob_x, ob_y, '-ko',markersize=15)
#            # check collision
#            if( np.hypot(diffX, diffY) < 3.0):
#                print("LOS BLOCKED")
#                USE_TAN = True
#                break
        
    # Calculate tangent
    deltaXTan = target_x - target_lastX
    deltaYTan = target_y - target_lastY
    tan = np.arctan2( deltaYTan, deltaXTan)
    
    alpha = 0.9
    cosa = alpha * np.cos(previousTan) + (1 - alpha) * np.cos(tan)
    sina = alpha * np.sin(previousTan) + (1 - alpha) * np.sin(tan)
    tanFiltered = np.arctan2(sina, cosa) # Calculate tangent
 
    #if( abs(tanFiltered) < np.deg2rad(45) ):
     #   USE_TAN = False
        
    if(USE_TAN):
        droneTAN_x = target_x - np.cos(tanFiltered)*const_dist
        if(droneTAN_x < target_x):
            droneTAN_x = target_x + np.cos(tanFiltered)*const_dist
            
        droneTAN_y = target_y - np.sin(tanFiltered)*const_dist
    
    droneCONST_x = droneCONST_x + (droneTAN_x - droneCONST_x)/2
    droneCONST_y = droneCONST_y + (droneTAN_y - droneCONST_y)/2
        
    return droneCONST_x, droneCONST_y, tanFiltered

def computeDesiredAcceleration(drone_x, drone_y, target_x, target_y, const_dist):
    
    deltaX = target_x - drone_x # - const_dist
    deltaY = target_y - drone_y #- const_dist
    
    deltaD = np.hypot(deltaX, deltaY ) - const_dist
    theta = np.arctan2( deltaX, deltaY)
    
    kp = 0.1
    return kp*np.array([deltaD*np.cos(theta), deltaD*np.sin(theta), 0])

WEIGHTED_AVG = False
def main():
    # Load environment
    env = environment()
    all_obstacles = env.obstacles
    height = 5
    yaw = np.deg2rad(180)
    
    # Load target trajectory
    offline_path = np.load('target_waypoints_easy.npy') # target_waypoints.npy
    #targetTraj = getTargetTraj(offline_path)
    splineX, splineY = getSpline(offline_path)
    Tf = 7 #targetTraj.TS[-1]
    target_x = []
    target_y = []
   
    # Define MPC
    x0_val = np.array([0.6, -1.5, height, -0.1, 0.5, 2, 0, 0, 0, 0, 0, 0])
    t0_val = 0.0
    target_x.append( splineX(t0_val) )
    target_y.append( splineY(t0_val) )
    
    pos0 = np.array([x0_val[0], x0_val[1], x0_val[2] ])
    vel0 = np.array([x0_val[3], x0_val[4], x0_val[5] ])
    velf = vel0
    acc0 = np.array([0, 0, 0 ])
    posfX, posfY, tangent = getGoalPosition([], x0_val[0], x0_val[1], target_x[-1],target_y[-1], DIST_2_TARGET, x0_val[0], x0_val[1],1.42)
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
    
    goalPointsX = []
    goalPointsY = []
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
        plt.xlim( (-10,10))
        plt.ylim( (-10,10))
        
        if not pause:
            observed_obstacles = updateObstacleMap(x[0],x[1], all_obstacles)
            nearest_obstacle, ind = getNearestObstacle(x[0],x[1],observed_obstacles)
            
            pos0 = np.array([x[0], x[1], x[2] ])
            vel0 = np.array([x[3], x[4], x[5] ])
            velf = vel0 + computeDesiredAcceleration(x[0], x[1], 
                                                     target_x[-1],target_y[-1], DIST_2_TARGET)*t
            acc0 = np.array([0, 0, 0 ])
            yaw = getYaw(x[0], x[1], target_x[-1],target_y[-1])
            tau = t+TRG_HORIZ*sampling_time
            # Select best goal to track target
            if(tau > Tf):
                target_xPredicted = splineX(Tf)
                target_yPredicted = splineY(Tf)
            else:
                target_xPredicted = splineX(tau) # target_x[-1]
                target_yPredicted = splineY(tau) # target_y[-1]
            
            if(len(target_x) <= 1): # first time, initialize with drone's initial state
                nearest_obstacle_target, ind = getNearestObstacle(target_xPredicted,
                                                              target_yPredicted,
                                                              observed_obstacles)
                posfX, posfY, tangent = getGoalPosition(nearest_obstacle_target, x[0], x[1],
                                                        target_xPredicted,target_yPredicted,
                                                        DIST_2_TARGET, x[0], x[1], 1.42)
            else:
#                nearest_obstacle_target, ind = getNearestObstacle(target_xPredicted,
#                                                              target_yPredicted,
#                                                              observed_obstacles)
#                posfX, posfY, tangent = getGoalPosition(nearest_obstacle_target, x[0], x[1],
#                                                        target_xPredicted,target_yPredicted,
#                                                        DIST_2_TARGET,
#                                               target_x[-1], target_y[-1], tangent)
                
                #T = np.array( [target_x[-1],target_y[-1]] )
                T = np.array( [target_xPredicted, target_yPredicted] )
                Wc = np.array( [x[0], x[1]] )
                nearest_obstacle_target, ind = getNearestObstacle(x[0], x[1],observed_obstacles)
                
                Wg = getGoalPoint(Wc, T, observed_obstacles)
                posfX = posfX + 0.9* ( Wg[0] - posfX)
                posfY = posfY + 0.9* ( Wg[1] - posfY)
                
                goalPointsX.append(Wg[0])
                goalPointsY.append(Wg[1])
            
            if(WEIGHTED_AVG):
                w = 0.1
                weightedX = (1-w)*splineX(t) + w*target_xPredicted # weighted average
                weightedY = (1-w)*splineY(t) + w*target_yPredicted # weighted average
                deltaX = weightedX - splineX(t)
                deltaY = weightedY - splineY(t)
                posfX = weightedX #x[0] + 1.5*sign(deltaX) #x[0] + 1.5*sign(deltaX)
                posfY = weightedY #x[1] + 1.5*sign(deltaY) #x[1] + 1.5*sign(deltaY)
            
            
            posf = np.array([posfX,posfY, height ]) # TARGET CURRENT LOCATION SHIFTED WITH RELATIVE DIST
            
            rapid_traj = generate_single_motion_primitive(pos0,vel0,acc0,posf,velf, t)
            rapid_posX.append( getRefTrajRapid(rapid_traj, t)[0] )
            rapid_posY.append( getRefTrajRapid(rapid_traj, t)[1] )
            
            xref = getReferenceTrajectoryTarget(rapid_traj, yaw, t)
            refTrajx.append( xref[0] )
            refTrajy.append( xref[1] )
            
            target_x.append( splineX(t) )
            target_y.append( splineY(t) )
            
            
            
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
            
            plt.plot(xs_plot[:,0], xs_plot[:,1], 'b--') # DRONE TRAJECTORY
            plt.plot(target_x,target_y, 'r') # TARGET'S TRAJECTORY
            plt.plot(target_x[-1],target_y[-1], '-ro',markersize=10) # TARGET LAST
            #plt.plot(goalPointsX, goalPointsY, 'g--')
            #plt.plot(posfX, posfY, '-go',markersize=15) # GOAL LAST (Motion primitive)
            plt.plot(refTrajx, refTrajy, 'g') # REFERENCE TRAJ (MOTION PRIMITIVE)
            plt.legend(['quadrotor', 'target', 'target_LOC', 'goalpoint'])
            
            # PLOT ENVIRONMENT
            plot_quadrotor(x[0], x[1], x[8])
            plot_cone(x[0],x[1],x[8])
            plot_refYaw(x[0],x[1],yaw)
            plotobs(x[0],x[1],all_obstacles)
           
            # PLOT TANGENT
            #print("YAW: " + str(yaw))
            #print("TANGENT: " + str(tangent))
            #plot_tangent(target_x[-1],target_y[-1],tangent)
            
            #plt.waitforbuttonpress()
            plt.savefig(os.getcwd()+"/images/"+"image"+str(idx)+'.png')
            plt.pause(0.001)
        
        ######### PLOTTING #########
    plt.show()
    plt.pause(1000)
  
if __name__ == '__main__':
    main()