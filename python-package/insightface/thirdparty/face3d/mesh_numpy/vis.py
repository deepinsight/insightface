from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D

def plot_mesh(vertices, triangles, subplot = [1,1,1], title = 'mesh', el = 90, az = -90, lwdt=.1, dist = 6, color = "grey"):
	'''
	plot the mesh 
	Args:
		vertices: [nver, 3]
		triangles: [ntri, 3]
	'''
	ax = plt.subplot(subplot[0], subplot[1], subplot[2], projection = '3d')
	ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles = triangles, lw = lwdt, color = color, alpha = 1)
	ax.axis("off")
	ax.view_init(elev = el, azim = az)
	ax.dist = dist
	plt.title(title)

### -------------- Todo: use vtk to visualize mesh? or visvis? or VisPy?
