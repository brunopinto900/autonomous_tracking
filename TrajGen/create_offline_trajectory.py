#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:10:39 2022

@author: bruno
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Rectangle
from mpl_point_clicker import clicker
#from scipy.interpolate import interp1d
from trajGen import trajGenerator

def on_move(event):
    # append event's data to lists
    x.append(event.xdata)
    y.append(event.ydata)
    coords.append((x, y))
    # update plot's data  
    points.set_data(x,y)
    # restore background
    #fig.canvas.restore_region(background)
    # redraw just the points
    ax.draw_artist(points)
    # fill in the axes rectangle
    fig.canvas.blit(ax.bbox)
    
def plotobstacles(ax,obs):
    '''plot all obstacles'''
    for box in obs:
        l = 0.5
        w = 0.5
        xloc = box[0]-l/2
        yloc = box[1]-w/2
        box_plt = Rectangle((xloc,yloc),l,w,linewidth=1.5,facecolor='k',zorder = 2)
        plt.gcf().gca().add_artist(box_plt)
        
def get_ob_list(ob):
    ob_list = np.empty( (ob.shape[0],4),dtype=np.float)
   
    for i in range(ob.shape[0]):
        ob_list[i][0] = ob[i][0] - 0.5
        ob_list[i][1] = ob[i][1] - 0.5
        ob_list[i][2] = ob[i][0] + 0.5
        ob_list[i][3] = ob[i][1] + 0.5
        
    return ob_list


ob = np.array( [ [-2,5],[-1.5,5], [-1,5],[-0.5,5], [0,5],[0.5,5],  [1,5], [1.5,5],[2,5]] )

#ob = np.array([[4.0, 2.0],
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
    
#ob_list = get_ob_list(ob)
        
fig, ax = plt.subplots()
plotobstacles(ax,ob)
ax.set_xlim(-15,15)
ax.set_ylim(-15,15)

x,y = [], []
# create empty plot
#points, = ax.plot([], [], 'o')
x_tg = np.array([0.0, 0.0])
goal = np.array([0.0, 7.0])
plt.plot(x_tg[0], x_tg[1], "Dr", markersize = 4)
plt.plot(goal[0], goal[1], "Xb",markersize = 8)

# cache the background
#background = fig.canvas.copy_from_bbox(ax.bbox)

#coords = []
klicker = clicker(ax, ["event"], markers=["x"])

fig.canvas.mpl_connect("button_press_event", on_move)
#print(x)

plt.show()
plt.pause(30)
traj_list = klicker.get_positions()['event']
traj_arr = np.array(traj_list)

x = traj_arr[:,0]
y = traj_arr[:,1]
waypoints = np.vstack((x,y)).T
np.save('target_waypoints.npy',waypoints)

#polynomial = trajGenerator(waypoints, max_vel = 2)
#
#
#t0 = polynomial.TS[0]
#tf = polynomial.TS[-1]
##
#t = t0
#dt = 0.1
##
#x_list = []
#y_list = []
##
#while(t < tf):
#    x_list.append( polynomial.get_des_state(t).pos[0] )
#    y_list.append( polynomial.get_des_state(t).pos[1] )
#    t+=dt
#
#fig1 = plt.figure(2)
#ax1 = fig1.gca()
#ax1.plot(x_list, y_list, 'g')
#plt.show()
#plt.pause(20)
#
##print( polynomial.get_des_state(t).pos )
#
##print(pos)
#
##xt = np.linspace(x[0],x[-1], num=x.shape[0], endpoint=True)
##yt = np.linspace(y[0],y[-1], num=y.shape[0], endpoint=True)
##
##xt_new = np.linspace(x[0],x[-1], num=220, endpoint=True)
##yt_new = np.linspace(y[0],y[-1], num=220, endpoint=True)
##
##fx = interp1d(xt, x, kind='cubic')
##fy = interp1d(yt, y, kind='cubic')
##
##x_interp = fx(xt_new)
##y_interp = fy(yt_new)
##
##traj = np.vstack((x_interp,y_interp))
##print(traj.shape)
