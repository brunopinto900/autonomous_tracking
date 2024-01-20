import numpy as np
#import aerosandbox.numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# hexacopter class
class HexaCopter:
    def __init__(self, module='numpy'):
        if module == 'casadi':
            self.sin = ca.sin
            self.cos = ca.cos
            self.pi = ca.pi
        elif module == 'numpy':
            self.sin = np.sin
            self.cos = np.cos
            self.pi = np.pi
        else:
            raise TypeError

        self.m = 1.44
        self.l = 0.23
        self.k = 1.6e-09
        self.Ixx = 0.0348
        self.Iyy = 0.0459
        self.Izz = 0.0977
        self.gamma = 0.01
        self.gc = 9.80665

        self.lbu = 0.144
        self.ubu = 6
        
    def dynamics(self, x, u):
        sin = self.sin
        cos = self.cos
        pi = self.pi

        m = self.m
        l = self.l
        k = self.k
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        gamma = self.gamma
        gc = self.gc         

        U1 = sum(u[i] for i in range(6))
        U2 = l*(-u[0]/2 - u[1] - u[2]/2 + u[3]/2 + u[4]+ u[5]/2)
        U3 = l*(-(np.sqrt(3)/2)*u[0] + (np.sqrt(3)/2)*u[2] + (np.sqrt(3)/2)*u[3] - (np.sqrt(3)/2)*u[5])
        U4 = k*(-u[0] + u[1] - u[2] + u[3] - u[4] + u[5]) - gamma * x[11]

        dx = [0] * 12
        dx[0] = x[6]
        dx[1] = x[7]
        dx[2] = x[8]
        dx[3] = x[9]
        dx[4] = x[10]
        dx[5] = x[11]
        dx[6] = (cos(x[5])*sin(x[4])*cos(x[3]) + sin(x[5])*sin(x[3]))*U1/m
        dx[7] = (sin(x[5])*sin(x[4])*cos(x[3]) - cos(x[5])*sin(x[3]))*U1/m
        dx[8] = -gc + (cos(x[3])*cos(x[4]))*U1/m
        dx[9] = ((Iyy-Izz)/Ixx)*x[10]*x[11] + U2/Ixx
        dx[10] = ((Izz-Ixx)/Iyy)*x[9]*x[11] + U3/Iyy
        dx[11] = ((Ixx-Iyy)/Izz)*x[9]*x[10] + U4/Izz

        return np.array(dx)

# cost calculating class
class CostFunction:
    def __init__(self, module='numpy'):
        if module == 'casadi':
            self.sin = ca.sin
            self.cos = ca.cos
            self.pi = ca.pi
        elif module == 'numpy':
            self.sin = np.sin
            self.cos = np.cos
            self.pi = np.pi
        else:
            raise TypeError
        pi = self.pi

        m = 1.44
        g_c = 9.80665

        self.n_x = 12
        self.n_u = 6
       
        self.u_ref = np.array([(m * g_c) / 6] * self.n_u)
        self.Q = np.array([1, 1, 1, 0.01, 0.01, 0, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001])
        self.Q[8] *= 10000
        self.R = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.Q_f = np.array([1, 1, 1, 0.01, 0.01, 0, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001])
        self.Q_f[8] *= 10000

        # initial obstacle field
        self.obstacle = []

    def set_obstacles(self, obstacle ):
        self.obstacle = obstacle
         
    def collision_cost(self, x, y):
        if( len(self.obstacle) == 0): # no obstacles
            return 0
        
        # calc repulsive potential
        dq = np.hypot(x - self.obstacle[0][0], y - self.obstacle[0][1])
        ETA = 100.0
        rr = 0.8
        
        return ca.exp(-( dq-rr) * ETA)

    def stage_cost(self, xref, x, u, t):
        l = 0.0

        for i in range(self.n_x):
            l += 0.5 * self.Q[i] * (x[i] - xref[i]) ** 2
        for i in range(self.n_u):
            l += 0.5 * self.R[i] * (u[i] - self.u_ref[i])**2
        
        l += self.collision_cost(x[0], x[1])
        return l
    
    def terminal_cost(self, xref, x, t):
        lf = 0.0
        for i in range(self.n_x):
            lf += 0.5 * self.Q_f[i] * (x[i] - xref[i]) ** 2
        
        lf += self.collision_cost(x[0], x[1])
        return lf



class MPC:
    def __init__(self, traj, traj_function, initial_X, initial_time):
        # opti interface
        opti = ca.Opti()
        self.hexacopter = HexaCopter()
        self.cost = CostFunction()
        self.traj = traj
        self.fn = traj_function
        self.yaw = 0
       
        # dimension of state and input
        n_x = 12
        n_u = 6
        self.n_x = n_x
        self.n_u = n_u
        
        # horizon length[s], total grids
        T = 1.0
        N = 20
        dt = T / N

        # time at each stage
        ts = opti.parameter(N + 1)
        t0 = initial_time
        opti.set_value(ts, [t0 + i * dt for i in range(N + 1)])

        # initial state
        x0 = opti.parameter(n_x)
        x0_val = initial_X
        opti.set_value(x0, x0_val)

        # decision variables (state, input) over the entire horizon.
        xs = opti.variable(n_x, N + 1)
        us = opti.variable(n_u, N)
        xs_guess = np.tile(x0_val, (N + 1, 1)).T
        us_guess = np.array([[1, 1, 1, 1, 1, 1] for i in range(N)]).T
        opti.set_initial(xs, xs_guess)
        opti.set_initial(us, us_guess)
        
        # cost
        J = 0.0
        for i in range(N):
            J += self.cost.stage_cost(self.getReferenceTraj( ts[i] ), xs[:, i], 
                                      us[:, i], ts[i]) * dt
        J += self.cost.terminal_cost(self.getReferenceTraj( ts[N] ) , xs[:, N], ts[N])

        opti.minimize(J)

        # state space equasion as equality constraints
        opti.subject_to(xs[:, 0] == x0)
        for i in range(N):
            f_array = self.hexacopter.dynamics(xs[:, i], us[:, i])
            f = ca.vertcat(*f_array)
            x1 = xs[:, i] + f * dt
            opti.subject_to(xs[:, i + 1] == x1)

        # bound for control input
        for i in range(N):
            opti.subject_to(opti.bounded(self.hexacopter.lbu, us[:, i], self.hexacopter.ubu))
           
        self.opti = opti
        
        self.dt = dt
        self.T = T
        self.N = N
        self.dt = dt
        self.J = J
        self.x0 = x0
        self.t0 = t0
        self.xs = xs
        self.us = us
        self.ts = ts
    
    
        
    def setHorizon(self,T):
        self.T = T
        self.dt = self.T / self.N
    
    def setTraj(self,traj):
        self.traj = traj
        
    def setTrajFunction(self,fn):
        self.fn = fn
    
    def setYaw(self, yaw):
        self.yaw = yaw
        
    def getReferenceTraj(self,t):
        return self.fn(self.traj,self.yaw, t) 
        
    def set_obstacles(self, obstacles ):
        self.cost.set_obstacles(obstacles)
    
    def update_cost_function(self):
        # cost
        J = 0.0
        for i in range(self.N):
            J += self.cost.stage_cost(self.getReferenceTraj( self.ts[i] ), self.xs[:, i], 
                                      self.us[:, i], self.ts[i]) * self.dt
        J += self.cost.terminal_cost(self.getReferenceTraj( self.ts[self.N] ) , self.xs[:, self.N], self.ts[self.N])
        
        self.opti.minimize(J)

    def setInitialGuess(self, x0_val=None, t0_val=None):
        if x0_val is None:
            x0_val = np.zeros(self.n_x)
        if t0_val is None:
            t0_val = 0.0

        self.opti.set_value(self.x0, x0_val)
        self.opti.set_value(self.ts, [t0_val + i * self.dt for i in range(self.N + 1)])

        xs_guess = np.tile(x0_val, (self.N + 1, 1)).T
        us_guess = np.array([[1, 1, 1, 1, 1, 1] for i in range(self.N)]).T
        self.opti.set_initial(self.xs, xs_guess)
        self.opti.set_initial(self.us, us_guess)        

        # use IPOPT as NLP solver
        init_solver_option = {'print_time': False, 'calc_lam_x': True, 'calc_lam_p': True, 'ipopt': {'mu_min': 0.1, 'max_iter': 1000, 'warm_start_init_point': 'yes', 'print_level':0, 'print_timing_statistics':'no'}}
        self.opti.solver('ipopt', init_solver_option)

        # to get initial strict solution
        sol = self.opti.solve()

        # solver option for MPC
        self.solver_option = init_solver_option
        self.solver_option['ipopt']['max_iter'] = 5
        self.opti.solver('ipopt', self.solver_option)
        
        # store primal and dual variables for warm-start
        self.xs_opt = sol.value(self.xs)
        self.us_opt = sol.value(self.us)
        self.lam_gs_opt = sol.value(self.opti.lam_g)

    def solve(self, x0_val, t0_val):
        # update current state
        self.opti.set_value(self.x0, x0_val)

        # update time sequence
        self.opti.set_value(self.ts, [t0_val + i * self.dt for i in range(self.N + 1)])

        self.update_cost_function()
        
        # warm start
        self.opti.set_initial(self.xs, self.xs_opt)
        self.opti.set_initial(self.us, self.us_opt)
        self.opti.set_initial(self.opti.lam_g, self.lam_gs_opt)

        try:
            sol = self.opti.solve()
        except:
            # print('fail')
            pass
        
        # store primal and dual variables for warm-start
        self.xs_opt = self.opti.debug.value(self.xs)
        self.us_opt = self.opti.debug.value(self.us)
        self.lam_gs_opt = self.opti.debug.value(self.opti.lam_g)

        # In MPC we use initial optimal input as actual input
        u0 = self.us_opt[:, 0]
        if not isinstance(u0, np.ndarray):
            u0 = np.array([u0])

        return u0