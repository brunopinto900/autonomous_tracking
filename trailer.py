#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:05:45 2024

@author: bruno
"""


import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.interpolate import CubicSpline

def getSpline(waypoints):
    N = waypoints.shape[0]
    t = np.linspace(0, 7, num=N)
    splineX = CubicSpline(t, waypoints[:,0])
    splineY = CubicSpline(t, waypoints[:,1])
    return splineX, splineY
    
# Load target trajectory
offline_path = np.load('target_waypoints.npy') # target_waypoints.npy
#targetTraj = getTargetTraj(offline_path)
splineX, splineY = getSpline(offline_path)
Tf = 7 #targetTraj.TS[-1]

target_x = []
target_y = []

trailer_x = []
trailer_y = []

drone_x = []
drone_y = []

t = 0
dt = 0.03
e1 = np.array([[1],[0]]).reshape(1,2)
e2 = np.array([ [0,1] ]).reshape(1,2)
d = 1.0 #1.0 # constant distance
a = math.pi/2
R = np.array( [ [np.cos(a), -np.sin(a)],[np.sin(a),np.cos(a) ] ] )
smallR = np.array( [ [0,0], [0,0 ] ] )

plt.cla()
plt.gca().set_facecolor('xkcd:pale green')
# for stopping simulation with the esc key.
plt.gcf().canvas.mpl_connect(
    'key_release_event',
    lambda event: [exit(0) if event.key == 'escape' else None])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim( (-10,10))
plt.ylim( (-10,10))

while(t < Tf):
    
    PLx = splineX(t) # target_x
    PLy = splineY(t)
    
    if(t == 0):
        VTx = 0
        VTy = 0
    else:
        VTx = ( PLx - splineX(t-dt) ) / dt
        VTy = ( PLy - splineY(t-dt) ) / dt
    R = np.array( [ [np.cos(a), -np.sin(a)],[np.sin(a),np.cos(a) ] ] )
    
    VT = np.array([VTx,VTy]).reshape(2,1)
    
    #WT = float( np.dot( e2 , np.matmul(np.transpose(R),VT) ) )
    WT = np.cos(a)*VTx
    a = a + WT*dt #math.atan2(PLy-PTy, PLx-PTx)   #a + WT*dt
    
    print(a)
    
    
    #R = np.array( [ [np.cos(a), -np.sin(a)],[np.sin(a),np.cos(a) ] ] )
    
    PL = np.array([PLx,PLy])
    
    
    #PT = PL - d*np.dot(R,np.transpose(e1))
    PTx = PLx - d*np.cos(a) #PT[0]
    PTy = PLy - d*np.sin(a)
    PT = np.array([PTx,PTy]).reshape(2,1)
    
    ri = np.array( [ [0],[0.8] ] )
    PD = PT - np.dot(R,ri)
    PDx = PD[0]
    PDy = PD[1]
    
  
    t+=dt
    
    
    # Plotting
    target_x.append( PLx )
    target_y.append( PLy )
    trailer_x.append( PTx )
    trailer_y.append( PTy )
    drone_x.append( PDx )
    drone_y.append( PDy )
    
    plt.plot(target_x,target_y, 'r')
    plt.plot(trailer_x, trailer_y, '--g')
    plt.plot(drone_x, drone_y, '--k')
    plt.pause(0.001)
    
    

plt.show()
plt.pause(1000)