import matplotlib.pyplot as plt
import numpy as np

SCALE = 10
FOV_D = 6.0
CONST_DIST = FOV_D/2

target_x = 6
target_y = 4
drone_x = 1
drone_y = 2
d = np.hypot(target_x - drone_x, target_y - drone_y )

figure, axes = plt.subplots()
CONST_DIST_circle = plt.Circle(( target_x , target_y ), CONST_DIST, color = 'g', fill = False)
DIST_circle = plt.Circle(( target_x , target_y ), d, color = 'r', fill = False)
drone_circle = plt.Circle((drone_x , drone_y ), 0.5, color = 'b', fill = False)
target_circle = plt.Circle((target_x, target_y ), 0.5, color = 'b', fill = False)

deltaX = target_x - drone_x
deltaY = target_y - drone_y
deltaD = np.hypot(deltaX, deltaY ) - CONST_DIST
theta = np.arctan2( deltaY, deltaX)
droneCONST_x = drone_x + np.cos(theta)*deltaD
droneCONST_y = drone_y + np.sin(theta)*deltaD

droneCONST_circle = plt.Circle((droneCONST_x , droneCONST_y ), 0.5, color = 'k', fill = False)



axes.add_artist(CONST_DIST_circle)
axes.add_artist(drone_circle)
axes.add_artist(target_circle)
axes.add_artist(DIST_circle)
axes.add_artist(droneCONST_circle)
plt.title( 'Colored Circle' )
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim( (-15, 15))
plt.ylim( (-15, 15))
plt.show()
plt.pause(100)