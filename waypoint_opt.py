#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 06:55:50 2024

@author: bruno
"""
import numpy as np
import matplotlib.pyplot as plt
import math

DS = 3.0 # 5 meters constant tracking
FOV_theta = 40/2
from matplotlib.patches import Wedge

def plot_cone(x,y,yaw):
    yaw = math.degrees(yaw)
    cone = Wedge((x, y), DS, yaw - FOV_theta, yaw + FOV_theta, ec="none",edgecolor='purple',facecolor='purple')
    plt.gcf().gca().add_artist(cone)

def plot_tangent(x,y,yaw):
    yaw = math.degrees(yaw)
    cone = Wedge((x, y), DS, yaw - FOV_theta/16, yaw + FOV_theta/16, ec="none",edgecolor='black',facecolor='black')
    plt.gcf().gca().add_artist(cone)
    
def distPoint2Vector(p1, p2, p3):
    return np.abs(np.linalg.norm(np.cross(p2-p1, p1-p3)))/np.linalg.norm(p2-p1) 

    
def getGoalPoint(Wc, T, obstacles):
    nObs = obstacles.shape[0]
    lambda_ = 0.5 / nObs
    dm = 0.02 #math.sin(FOV_theta)*(DS/2)
    theta_discr = np.linspace( 0,-math.pi/2, int(math.pi*10) )
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
            
            #l2 = distPoint2Vector(p1, p2, p3)
            l2 = np.hypot(  midx - ob[0] , midy - ob[1] )
            cost +=  lambda_*math.exp( -(l2+dm) ) 
        
        costs.append(cost)
    
    costs_arr = np.array(costs)
    idx = np.argmin(costs_arr)
    
    Wg = discrete_wp[idx]
    
    return Wg



current_wp = np.array( [1,1] )
target_point = np.array( [2.5,6.5] )
initial_guess = np.array( [2.0, 3.2] )

obstacles = np.array( [ [1.0,5.0],[4.0,5.2],[1.6,2.0] ] ) # 

plt.cla()
plt.gca().set_facecolor('xkcd:pale green')
# for stopping simulation with the esc key.
plt.gcf().canvas.mpl_connect(
    'key_release_event',
    lambda event: [exit(0) if event.key == 'escape' else None])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim( (-1,10))
plt.ylim( (-1,10))

drone = plt.Circle((current_wp[0], current_wp[1]),0.2,edgecolor='k',facecolor='r',zorder=2)
target = plt.Circle((target_point[0], target_point[1]),0.2,edgecolor='r',facecolor='r',zorder=2)
#next_wp = plt.Circle((initial_guess[0], initial_guess[1]),0.2,edgecolor='y',facecolor='y',zorder=2)
obstacle_plt1 = plt.Circle((obstacles[0][0], obstacles[0][1]),0.5,edgecolor='k',facecolor='k',zorder=2)
obstacle_plt2 = plt.Circle((obstacles[1][0], obstacles[1][1]),0.3,edgecolor='k',facecolor='k',zorder=2)
obstacle_plt3 = plt.Circle((obstacles[2][0], obstacles[2][1]),0.3,edgecolor='k',facecolor='k',zorder=2)
#obstacle_plt4 = plt.Circle((obstacles[3][0], obstacles[3][1]),0.3,edgecolor='k',facecolor='k',zorder=2)
const_dist = plt.Circle((target_point[0], target_point[1]),DS,edgecolor='k',facecolor='None',zorder=2)

target_point2 = np.array( [4.0,8.0] )

wg = getGoalPoint(current_wp, target_point, obstacles)
wg_plot = plt.Circle((wg[0], wg[1]),0.2,edgecolor='b',facecolor='b',zorder=2)


plt.gcf().gca().add_artist(drone)
plt.gcf().gca().add_artist(target)
#plt.gcf().gca().add_artist(next_wp)
plt.gcf().gca().add_artist(obstacle_plt1)
plt.gcf().gca().add_artist(obstacle_plt2)
plt.gcf().gca().add_artist(obstacle_plt3)
#plt.gcf().gca().add_artist(obstacle_plt4)
plt.gcf().gca().add_artist(wg_plot)
plt.gcf().gca().add_artist(const_dist)

deltaXTan = target_point[0] - wg[0]
deltaYTan = target_point[1] - wg[1]
direction = np.arctan2( deltaYTan, deltaXTan)
    
plot_cone(wg[0],wg[1],direction)
plot_tangent(wg[0],wg[1],direction)

plt.show()
plt.pause(1000)
