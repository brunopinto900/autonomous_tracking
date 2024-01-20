import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Rectangle
from scipy.spatial import cKDTree
import heapq
import math
import quadrocoptertrajectory as quadtraj

show_animation = True
RADIUS = 0.5

# NODE STATE ENUM
IN_CLOSE_SET = 1
IN_OPEN_SET = 2
NOT_EXPAND = 3

REACH_HORIZON = 1
REACH_END = 2
NO_PATH = 3
NEAR_END = 4

# TODO
# verify parameters/config (maybe make parameters depending on others?)
# verify time propagation
#convert Path to Traj or setup a new reference trajectory method for MPC
obstacles = np.array( [ [-2,5],[-1.5,5], [-1,5],[-0.5,5], [0,5],[0.5,5],  [1,5], [1.5,5],[2,5]] )

def infinity_norm(vector):
    return np.linalg.norm(vector, np.inf)
    #return vector.flat[np.abs(vector).argmax()]
  
class Planner():
    def __init__(self): # params
        self.path_nodes = []
        self.DEBUG_X = []
        self.DEBUG_Y = []
        self.qeue = []
        self.init_search = True
        self.traj_shot_ = None
        self.t_shot_ = 0
        self.is_shot_succ_ = False
        self.has_path = False
        self.obstacle_kd_tree = None
        self.expanded_nodes_ = NodeHash()
        self.time_origin = 0
        self.start_vel = 0
        self.start_acc = 0
        self.path = []
        self.w_time_ = -1.0
        self.max_vel = 0
        self.max_acc_ = 0
        self.tie_breaker_ = 1.0 + 1.0 / 10000
        self.lambda_heu_ = 0.0
        self.dynamic = True
        self.resolution_ = 0.1
        self.time_resolution = 1
        self.inv_time_resolution_ = 1.0 / self.time_resolution
        self.horizon_ = 7.0
        
        self.init_max_tau_ = 0
        self.max_tau_ = 0
        self.num_iters = 0
    
    def replan(self, start_pos,start_vel, start_acc,
               end_pos, end_vel):
        
        self.reset()
        status = self.search(start_pos, start_vel, start_acc, end_pos, end_vel)
        if(status == NO_PATH):
            print(NO_PATH)
        
        path = self.getKinoTraj(0.01)
        
        return path
        
    def reset(self):
        self.obstacle_kd_tree = cKDTree()
        self.init_search = True
        self.qeue = []
        self.path_nodes = []
        self.expanded_nodes_ = []
        self.is_shot_succ_ = False
        self.num_iters = 0
        
    def config(self, time_start, time_aggressivness, max_vel, max_acc, resolution,
               lambda_heu_, time_resolution, horizon,
               init_max_tau, max_tau):
        
        self.time_origin = time_start
        self.w_time_ = time_aggressivness
        self.max_vel = max_vel
        self.max_acc_ = max_acc
        self.resolution = resolution
        self.lambda_heu_ = lambda_heu_
        self.time_resolution = time_resolution
        self.inv_time_resolution_ = 1.0 / time_resolution
        self.horizon_ = horizon
        self.init_max_tau_ = init_max_tau
        self.max_tau_ = max_tau
    
    # Based on Search-based Motion Planning for Quadrotors using
    # Linear Quadratic Minimum Time Control
    def calc_distance_heuristic(self, start_pos, start_vel, goal_pos, goal_vel):
        heuristic_cost = 0
        optimal_time = 0
        
        dp = goal_pos - start_pos
        v0 = start_vel
        v1 = goal_vel
        
        c1 = -36 * dp.dot(dp)
        c2 = 24 * (v0+v1).dot(dp)
        c3 = -4 * (v0.dot(v0) + v0.dot(v1) + v1.dot(v1) )
        c4 = 0
        c5 = self.w_time_
        
        v_max = self.max_vel
        t_bar = infinity_norm( dp ) / v_max
        t = t_bar
        c = -c1/(3*t*t*t) - c2/(2*t*t) - c3/t + self.w_time_*t
        
        # ts is a numpy array (roots 4th order polynomial)
        ts = abs( np.real( self.quartic(c5,c4,c3,c2,c1) ) )
        
        v_max = self.max_vel
        t_bar = infinity_norm( dp ) / v_max
        ts = np.append(ts, t_bar)
        
        cost = 100000000
        t_d = t_bar
        
        for t in ts:
            if(t < t_bar):
                continue
            c = -c1/(3*t*t*t) - c2/(2*t*t) - c3/t + self.w_time_*t
            if( c < cost):
                cost = c
                t_d = t
            
        optimal_time = t_d
        heuristic_cost =  1.0 * (1 + self.tie_breaker_) * cost
        return heuristic_cost, optimal_time
    
    def quartic(self, a, b, c, d, e):
        coeffs = [a, b, c, d, e]
        return np.roots(coeffs)
        
    def setParam(self, xy_resolution):
        self.xy_resolution = xy_resolution
        
    def posToIndex(self, pos):
        origin_ = np.array([-10,-10,5])
        return np.floor(  (1.0/self.resolution_) * (pos - origin_) ).astype(int)
  
    def timeToIndex(self, time):
        dt = time - self.time_origin_
        return math.floor( dt * self.inv_time_resolution_ )
    
    def search(self, start_pos, start_vel, start_acc, goal_pos, goal_vel,time_start):
        
        print("Searching")
        self.start_pos = start_pos
        self.start_vel = start_vel
        self.start_acc = start_acc
                                               
        cur_node = Node( pos = start_pos, vel = start_vel, pos_index = self.posToIndex(start_pos), coords = [start_pos]  ) 
        goal_node = Node(pos = goal_pos, vel = goal_vel, pos_index = self.posToIndex(goal_pos), coords = [goal_pos] )
        
        
        hc, optimal_time = self.calc_distance_heuristic(start_pos, start_vel, goal_pos, 
                                goal_vel)
        
        cur_node.f_score = self.lambda_heu_ * hc
        cur_node.node_state = IN_OPEN_SET
        
        heapq.heappush(self.qeue, (cur_node.f_score, cur_node) )
        
        # Dynamic allocation
        self.time_origin_ = time_start
        cur_node.time = time_start
        cur_node.time_idx = self.timeToIndex(time_start)
        self.expanded_nodes_.insert(cur_node.pos_index, cur_node.time_idx, cur_node)
        
        while( len(self.qeue) > 0 ): # while the qeue is not empy
            
            if(self.num_iters == 1000):
                print("Niteration exceeded")
                return NO_PATH
        
            self.num_iters = self.num_iters + 1
            cur_node = self.qeue[0][1] # [1] -> retrieves node
           
            search_state = self.check_for_termination(cur_node, start_acc, start_pos, goal_node)
            if(search_state != NO_PATH):
                print("Search terminated" + " with :" + str(search_state))
                return search_state
                
            heapq.heappop(self.qeue)
            cur_node.node_state = IN_CLOSE_SET
            cur_pos = cur_node.pos
            cur_vel = cur_node.vel
            
            inputs_, durations = self.generate_motion_primitives(start_acc)
            
            self.propagate_nodes(cur_node, cur_pos, cur_vel, goal_pos, goal_vel, durations, inputs_, durations) #optimal_time
     
   
        return NO_PATH
    
    def in_close_set(self,pos, time):
        pos_id = self.posToIndex(pos)
        time_id = self.timeToIndex(time)
        return self.expanded_nodes_.find(pos_id, time_id)
        
    def state_transit(self, cur_pos, cur_vel, u, tau):
        next_pos = cur_pos + cur_vel*tau + 0.5 * (tau*tau) * u
        next_vel = cur_vel + tau*u
        return next_pos, next_vel
        
    def propagate_nodes(self, cur_node, cur_pos, cur_vel, goal_pos, goal_vel, time_to_goal, inputs, durations):
        
        tmp_expand_nodes = []
       
        for i in np.arange(0,len(inputs),1):
            for j in np.arange(0,len(durations),1):
                um = inputs[i]
                tau = durations[j]
                
                pro_pos, pro_vel = self.state_transit(cur_pos, cur_vel, um, tau)
                
                #print(pro_pos)
                #print("POSITION")
                
                pro_pos_id = self.posToIndex(pro_pos)
                pro_t = cur_node.time + tau
                
                pro_node = self.in_close_set(pro_pos, pro_t) 
                if( not pro_node):
                    if(self.init_search):
                        continue
                
                same_voxel = np.linalg.norm( pro_pos_id - cur_node.pos_index) == 0 
                if( same_voxel ):
                    if(self.init_search):
                        continue
                    
                feasible = self.check_feasibility(pro_pos,pro_vel, um, tau)
                if( not feasible):
                    if(self.init_search):
                        continue
            
                tmp_g_score = (np.linalg.norm(um.dot(um)) + self.w_time_) * tau + cur_node.g_score
             
                hc, opt_time = self.calc_distance_heuristic(pro_pos,pro_vel, goal_pos, goal_vel)
                tmp_f_score = tmp_g_score+self.lambda_heu_*hc
             
                # Compare nodes expanded from the same parent
                prune = False;
                for expanded_node in tmp_expand_nodes:
                    if( np.linalg.norm(pro_pos_id - expanded_node.pos_index) == 0):
                        prune = True
                        if(tmp_f_score < expanded_node.f_score):
                            expanded_node.f_score = tmp_f_score
                            expanded_node.g_score = tmp_g_score
                            expanded_node.pos = pro_pos
                            expanded_node.vel = pro_vel
                            expanded_node.input = um
                            expanded_node.duration = tau
                            expanded_node.time = cur_node.time + tau
                            
                    break
                
                if(not prune):
                    if( not pro_node ): # doesn«t exist yet
                        pro_node = Node()
                        pro_node.pos=pro_pos
                        pro_node.vel=pro_vel 
                        pro_node.f_score = tmp_f_score
                        pro_node.g_score = tmp_g_score
                        pro_node.pos_index = pro_pos_id
                        pro_node.input_ = um
                        pro_node.duration = tau
                        pro_node.parent= cur_node
                        pro_node.node_state = IN_OPEN_SET
                        pro_node.time = cur_node.time + tau
                        pro_node.time_index = self.timeToIndex(cur_node.time + tau)
                
                        
                
                        if not show_animation:
                            plt.cla()
                            plt.gca().set_facecolor('xkcd:pale green')
                            # for stopping simulation with the esc key.
                            plt.gcf().canvas.mpl_connect(
                                'key_release_event',
                                lambda event: [exit(0) if event.key == 'escape' else None])
                            plt.gca().set_aspect('equal', adjustable='box')
                            plt.xlim( (-4,5))
                            plt.ylim( (-6,8))
                            
                            plt.plot(cur_node.pos[0],cur_node.pos[1], '-ro',markersize=10) # TARGET LAST
                            plt.plot(pro_node.pos[0],pro_node.pos[1], '-bo',markersize=10) # TARGET LAST
                            plotobs(obstacles)
                            plt.show()
                            plt.pause(0.001)
                            #print(kino_planner.DEBUG_X)
                            #plt.plot(kino_planner.DEBUG_X, kino_planner.DEBUG_Y, 'k')
                        
                        heapq.heappush(self.qeue,(pro_node.f_score,pro_node) )
                        #self.DEBUG_X.append( pro_node.pos[0] )
                        #self.DEBUG_Y.append( pro_node.pos[1] )
                        
                        self.expanded_nodes_.insert(pro_node.pos_index, pro_node.time_index, pro_node)
                        tmp_expand_nodes.append(pro_node)
                    
                    elif(pro_node.node_state == IN_OPEN_SET):
                        if(tmp_g_score < pro_node.g_score):
                            pro_node.pos=pro_pos
                            pro_node.vel=pro_vel 
                            pro_node.f_score = tmp_f_score
                            pro_node.g_score = tmp_g_score
                            pro_node.input_ = um
                            pro_node.duration = tau
                            pro_node.parent=cur_node
                            pro_node.time = cur_node.time + tau
                            pro_node.time_index = self.timeToIndex(cur_node.time + tau)

    def generate_motion_primitives(self,start_acc):
        
        inputs = []
        durations = []
        res = 0.5
        time_res = self.max_tau_ / 4
#        time_res_init = self.max_tau_ / 4
#        
        if(self.init_search):
            self.init_search = False
#            inputs.append( start_acc )
#            t0 = time_res_init
             #ts = time_res_init
#            tf = self.init_max_tau_ + ts
#            
#            for tau in np.arange(t0 , tf, ts):
#                durations.append(tau)
#        else:
        t0 = time_res
        ts = time_res
        tf = self.max_tau_ + ts
        for tau in np.arange(t0 , tf, ts):
            durations.append(tau)
    
        acc0 = -self.max_acc_
        accts = res
        accf = self.max_acc_ + accts
        
        for ax in np.arange(acc0,accf,accts):
            for ay in np.arange(acc0,accf,accts):
                for az in np.arange(acc0,accf,accts):
                    inputs.append( np.array([ax,ay,az]))
        
        return inputs, durations
            
    def check_for_termination(self,cur_node, start_acc, start_pos, goal_node):
        reached_horizon = np.linalg.norm( cur_node.pos - start_pos ) >= self.horizon_
        near_end = ( np.linalg.norm( cur_node.pos - goal_node.pos ) ) <= 2.0 #self.resolution
        
        #hc, optimal_time = self.calc_distance_heuristic(cur_node.pos, cur_node.vel, 
                                                       #goal_node.pos,goal_node.vel) 
        
        #self.computeShotTraj(cur_node.pos, cur_node.vel, start_acc, 
                                #goal_node.pos,goal_node.vel, optimal_time) # optimal_time
                
        if (reached_horizon or near_end):
            terminate_node = cur_node
            self.retrievePath( terminate_node )
            
            print("COMPUTING SHOT TRAJ")
            hc, optimal_time = self.calc_distance_heuristic(cur_node.pos, cur_node.vel, 
                                                   goal_node.pos,goal_node.vel) 
    
            print(optimal_time)
            print(cur_node.pos)
            self.computeShotTraj(cur_node.pos, cur_node.vel, start_acc, 
                            goal_node.pos,goal_node.vel, optimal_time)
            
        if(self.is_shot_succ_):
            print("SHOT SUCCESSFUL REACH_END")
            return REACH_END
        
        if(reached_horizon):
            print(optimal_time)
            if(self.is_shot_succ_):
                print("REACH_END")
                return REACH_END
            else:
                print("REACH_HORIZON")
                return REACH_HORIZON
        
        if(near_end):
            if(self.is_shot_succ_):
                print("SHOT SUCCESSFUL REACH_END")
                return REACH_END
            elif(not cur_node.parent):
                print("NEAR END")
                return NEAR_END
            else:
                print("NO_PATH")
                return NO_PATH
        
        return NO_PATH
    
    def check_feasibility(self,cur_pos, cur_vel, u, time):
        # Check for maximum
        if( max( abs(cur_vel) ) > self.max_vel ):
            return False
            
        for tau in np.arange(time/10 , time + time/10, time/10):
            next_pos, next_vel = self.state_transit(cur_pos, cur_vel, u, tau)
            # check collision
            ids = self.obstacle_kd_tree.query_ball_point([next_pos[0], next_pos[1]], RADIUS)

            if ids:
                return False
            
        return True
            
    def check_feasibility_traj(self,traj, time):
        x_list = []
        y_list = []
        z_list = []
        for tau in np.arange(time/10 , time + time/10, time/10):
            pos = traj.get_position(tau)
            x_list.append(pos[0])
            y_list.append(pos[1])
            z_list.append(pos[2])
            vel = traj.get_velocity(tau)
            acc = traj.get_acceleration(tau)
        
            # Check for maximum
            if( (max( abs(vel) ) > self.max_vel) or (max( abs(acc) ) > self.max_acc_) ):
                return False
            
            # check collision
            return not self.check_collision(x_list, y_list, z_list)
    
    def computeShotTraj(self, start_pos, start_vel, start_acc, goal_pos, goal_vel, time_to_goal):
        # Based on RapidQuadcopter Motion Primitive
        traj = quadtraj.RapidTrajectory(start_pos, start_vel, start_acc)
        traj.set_goal_position(goal_pos)
        traj.set_goal_velocity(goal_vel)
        traj.generate(time_to_goal) # Run the algorithm, and generate the trajectory.
        
        traj_is_feasible = self.check_feasibility_traj(traj, time_to_goal)
        print("Is traj feasible :" + str(traj_is_feasible) )
        
        if(traj_is_feasible):
            self.traj_shot_ = traj
            self.t_shot_ = time_to_goal
            self.is_shot_succ_ = True
        else:
            self.traj_shot_ = None
            self.t_shot_ = -1
            self.is_shot_succ_ = False
            
    def retrievePath(self,end_node):
        cur_node = end_node
        self.path_nodes.append(cur_node)
        
        while(cur_node.parent):
            cur_node = cur_node.parent
            self.path_nodes.append(cur_node)
        
        return self.path_nodes.reverse()
        
        
    def setMap(self,ox, oy):
        self.ox = ox
        self.oy = oy
        self.obstacle_kd_tree = cKDTree(np.vstack((ox, oy)).T)
    
    def getKinoTraj(self, delta_t):
        pos_list = []
        vel_list= []
        xpos_list = []
        ypos_list = []
        
        node = self.path_nodes[-1]
        duration = node.duration
        t0 = duration
        ts = -delta_t
        tf = -1e-5 + ts
    
        if(self.is_shot_succ_):
            for t in np.arange(self.path_nodes[0].duration, self.t_shot_ + delta_t, delta_t):
                pos_list.append( self.traj_shot_.get_position(t) )
                vel_list.append( self.traj_shot_.get_velocity(t) )
                xpos_list.append(self.traj_shot_.get_position(t)[0])
                ypos_list.append(self.traj_shot_.get_position(t)[1])
                
            return pos_list, vel_list, xpos_list, ypos_list
            
        while(node.parent):
      
            ut = node.input
            pos0 = node.parent.pos
            vel0 = node.parent.vel
            t0 = node.parent.duration
            ts = -delta_t
            tf = -1e-5 + ts
            for t in np.arange(t0, tf, ts):
                next_pos, next_vel = self.state_transit(pos0,vel0,ut,t)
                pos_list.append(next_pos)
                xpos_list.append(next_pos[0])
                ypos_list.append(next_pos[1])
                vel_list.append(next_vel)
            
            node = node.parent
        
        return pos_list, vel_list, xpos_list, ypos_list
    
    def check_collision(self,x_list, y_list, z_list):
        collision = False
        for i_x, i_y, i_z in zip(x_list, y_list, z_list):
            cx = i_x
            cy = i_y
            ids = self.obstacle_kd_tree.query_ball_point([cx, cy], RADIUS)

            if ids:
                collision = True
                break

        return collision

class Node():
    def __init__(self, pos = np.array([0,0,0]), vel = np.array([0,0,0]),
                 pos_index = np.array([-1,-1,-1]),
                 coords = [],g_score = np.inf, f_score = np.inf,
                 parent=None, time = 0, time_index = 0,
                 input_ = np.array([0,0,0]), duration = 0):
        
        self.pos = pos
        self.vel = vel
        self.pos_index = pos_index
        self.coords = coords
        self.g_score = g_score
        self.f_score = f_score
        self.parent = parent
        self.time = time
        self.time_index = time_index
        self.input = input_
        self.duration = duration
    
    def __lt__(self, other):
        return self.f_score < other.f_score
    
    def __eq__(self, other):
        return self.f_score == other.f_score
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.parent

class NodeHash():
    def __init__(self):
        self.data = {}
    
    def insert(self, pos_idx, time_idx, node):
        idx_array = np.hstack( (pos_idx, time_idx) )
        self.data[ idx_array.tobytes() ] = node
    
    def find(self,pos_idx, time_idx):
        idx_array = np.hstack( (pos_idx, time_idx) )
        return self.data.get( idx_array.tobytes() )
        
    def reset(self):
        self.data.clear()

        
radius = 0.5
def plotobs(obstacles):
    '''plot all obstacles'''
    l = radius
    w = radius
    color_ = 'k'
    for obs in obstacles:
        xloc = obs[0]-l/2
        yloc = obs[1]-w/2
        box_plt = Rectangle((xloc,yloc),l,w,linewidth=1.5,facecolor=color_,zorder = 2)
        plt.gcf().gca().add_artist(box_plt)

def getObstaclesCoords(obstacles):
    ox = obstacles[:,0]
    oy = obstacles[:,1]
    return ox, oy       

def main():
    
    
    ox, oy = getObstaclesCoords( obstacles )
    
    # Initial conditions
    height = 5
    start_pos = np.array([0, -3, height])
    start_vel = np.array([1, 1, 2])
    start_acc = np.array([1,1,-9.81]) 
    goal_pos = np.array([2,6,height])
    goal_vel = np.array([1,1,2])
    time_start = 0
    
    # Instantiante planner
    kino_planner = Planner()
    time_aggressivness = 10
    max_vel = 4
    max_acc = 2
    resolution = 0.1
    lambda_heu = 100 #100.0
    time_resolution = 0.1
    horizon = 16
    max_tau = 0.8
    init_max_tau = 0.8
    
    kino_planner.config(time_start, time_aggressivness, max_vel, max_acc, resolution,
                        lambda_heu, time_resolution, horizon,
                        init_max_tau, max_tau)
    
    kino_planner.setMap(ox,oy)
    print("Executing Planner")
    planner_state = kino_planner.search(start_pos, start_vel, start_acc, goal_pos, goal_vel, time_start)
    if(planner_state != 3): # 3 means NO_PATH
        print("path_found")
        pos_list, vel_list, xpos_list, ypos_list = kino_planner.getKinoTraj(0.01)
    else:
        print("Path not found")
        
    if show_animation:
        plt.cla()
        plt.gca().set_facecolor('xkcd:pale green')
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim( (-16,16))
        plt.ylim( (-16,16))
        
        plt.plot(start_pos[0],start_pos[1], '-ro',markersize=10) # TARGET LAST
        plt.plot(goal_pos[0],goal_pos[1], '-bo',markersize=10) # TARGET LAST
        #print(kino_planner.DEBUG_X)
        plt.plot(kino_planner.DEBUG_X, kino_planner.DEBUG_Y, 'k')
        plt.plot(xpos_list, ypos_list,'b')
        
        plotobs(obstacles)
    
    plt.show()
    plt.pause(100)
    
    print("Done")
if __name__ == '__main__':
    main()