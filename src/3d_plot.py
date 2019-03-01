'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D as mp3
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')
#random_search_results
# 300results
# Make data.

# data = np.loadtxt('150results',delimiter=',', dtype=np.str)
#
# data_sig_softm = []
#
# for item in data:#range(0, len(data)):
#     # print(item)
#     a = float(item[0][2:])
#     b = float(item[1][0:len(item[1])-1])
#     c = str(item[2][2:len(item[2])-1])
#     d = int(item[3])
#     e = int(item[4])
#     f = str(item[5][2:len(item[5])-1])
#     item = [a,b,c,d,e,f]
#     if c == 'selu' and f == 'softmax':
#         data_sig_softm.append(item)
#
# for item in data_sig_softm:
#     print(item)
# X = []
# Y = []
# Z = []
# for i in range(0,len(data_sig_softm)):
#     X.append(data_sig_softm[i][2])
#     Y.append(data_sig_softm[i][3])
#     Z.append(data_sig_softm[i][1])
#
#
# # data = plt.cm.jet(data[X, Y])
# ax = fig.gca(projection='3d')
# surf = ax.plot_trisurf(X, Y, Z,linewidth=0.2, antialiased=True, cmap = cm.OrRd)
# # ax.scatter(X, Y, Z, marker='s', c=data)
#
# ax.set_zlim(0.20, 1.00)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

##########################################################
#
# data = np.loadtxt('random_search_results',delimiter=',', dtype=np.str)
#
# data_sig_softm = []
#
# for item in data:#range(0, len(data)):
#     # print(item)
#     a = float(item[0][2:])
#     b = float(item[1][0:len(item[1])-1])
#     c = str(item[2][2:len(item[2])-1])
#     d = int(item[3])
#     e = int(item[4])
#     print(e)
#     f = str(item[5][2:len(item[5])-1])
#     item = [a,b,d,e]
#     # if c == 'selu' and f == 'softmax':
#     data_sig_softm.append(item)
#
# for item in data_sig_softm:
#     print(item)
#
# X = []
# Y = []
# Z = []
# for i in range(0,len(data_sig_softm)):
#     X.append(data_sig_softm[i][2])
#     Y.append(data_sig_softm[i][3])
#     Z.append(data_sig_softm[i][1])
#
#
# # data = plt.cm.jet(data[X, Y])
# ax = fig.gca(projection='3d')
# surf = ax.plot_trisurf(X, Y, Z,linewidth=0.2, antialiased=True, cmap = cm.OrRd)
# # ax.scatter(X, Y, Z, marker='s', c=data)
#
# ax.set_zlim(0.20, 1.00)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

##########################################################
data = np.loadtxt('70_t_res',delimiter=',', dtype=np.str)

data_sig_softm = []

for item in data:#range(0, len(data)):
    # print(item)
    a = float(item[0][2:])
    b = float(item[1][0:len(item[1])-1])
    c = str(item[2][2:len(item[2])-1])
    d = int(item[3])
    e = int(item[4])
    print(e)
    f = str(item[5][2:len(item[5])-1])
    g = float(item[6][:len(item)-1])
    item = [a,b,d,e,g]
    # if c == 'selu' and f == 'softmax':
    data_sig_softm.append(item)

for item in data_sig_softm:
    print(item)

X = []
Y = []
Z = []
for i in range(0,len(data_sig_softm)):
    X.append(data_sig_softm[i][2])
    Y.append(data_sig_softm[i][3])
    Z.append(data_sig_softm[i][1])


# data = plt.cm.jet(data[X, Y])
ax = fig.gca(projection='3d')
surf = ax.plot_trisurf(X, Y, Z,linewidth=0.2, antialiased=True, cmap = cm.OrRd)
# ax.scatter(X, Y, Z, marker='s', c=data)

ax.set_zlim(0.20, 1.00)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

##########################################################

# X = []
# Y = []
# Z = []
# for i in range(0,len(data_sig_softm)):
#     X.append(data_sig_softm[i][4])
#     Y.append(data_sig_softm[i][3])
#     Z.append(data_sig_softm[i][0])
#
#
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(X, Y, Z,linewidth=0.2, antialiased=True, cmap = cm.OrRd)
# plt.show()

##########################################################
# mp3.plot_trisurf(np.array(X), np.array(Y), np.array(Z))
#
#
# X, Y = np.meshgrid(X, Y)
# # Z = np.meshgrid(np.array(np.ones(len(data_sig_softm[1])), data_sig_softm[1]))

# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# #
# # # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# #
# # # Add a color bar which maps values to colors.
# # fig.colorbar(surf, shrink=0.5, aspect=5)
# #
