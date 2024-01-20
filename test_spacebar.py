import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")

import pylab as plt
plt.ion()

fig = plt.figure()

pause = False

def onclick(event):
    global pause
    pause = not pause

fig.canvas.mpl_connect('button_press_event', onclick)

def drawGraph(matrix):
    fig.clear()
    G = nx.DiGraph(np.array(matrix))
    nx.draw_networkx(G)
    plt.draw()

A=[[0,1],[1,0]]
B=[[0,1],[0,0]]
x=1
while True:
    if not pause:
        if x==1:
            drawGraph(A)
            x=0
        else:
            drawGraph(B)
            x=1
    fig.canvas.get_tk_widget().update() # process events