import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import random


#3d rotation
def rotate(X, theta, axis='x'):
  '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
  c, s = np.cos(theta), np.sin(theta)
  if axis == 'x': return np.dot(X, np.array([
    [1.,  0,  0],
    [0 ,  c, -s],
    [0 ,  s,  c]
  ]))
  elif axis == 'y': return np.dot(X, np.array([
    [c,  0,  -s],
    [0,  1,   0],
    [s,  0,   c]
  ]))
  elif axis == 'z': return np.dot(X, np.array([
    [c, -s,  0 ],
    [s,  c,  0 ],
    [0,  0,  1.],
  ]))

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

#3d plot
def _3D_Plotter(Input_np_array):#3d matlab plot 3d plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    sequence_containing_x_vals = list(Input_np_array[:,0])
    sequence_containing_y_vals = list(Input_np_array[:,1])
    sequence_containing_z_vals = list(Input_np_array[:,2])
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    plt.show()

_3D_points=[]
_3D_points.append((-150.0, -150.0, -125.0))
_3D_points.append((150.0, -150.0, -125.0))
_3D_points.append(( 0.0, 0.0, 0.0))
_3D_points.append((0.0, -330.0, -65.0))
_3D_points.append((-225.0, 170.0, -135.0))
_3D_points.append(( 225.0, 170.0, -135.0))

#convert to numpy array for matrix operations
_3D_points_np=np.array(_3D_points)


#NEXT - remember matrix multiplication rules
#need to convert to homogenous vecotrs (x,y,z,1), do translation, then convert back from homogenous (divinde by last element)


def Translate3D(xyz_coords,**kwargs):
  #need to add concatenate input matrix
  #Ones=np.ones(xyz_coords.shape[0])
  #array2 = np.append(xyz_coords, [Ones], axis = 1)
  HomogenousMatrix=np.ones([xyz_coords.shape[0],xyz_coords.shape[1]+1])
  HomogenousMatrix[:xyz_coords.shape[0],:xyz_coords.shape[1]]=xyz_coords[:,:]
  for key, value in kwargs.items():
        print ("%s == %s" %(key, value))
  #now have an extra column for each xyz point - homogenous projection
  #create translation matrix
  #3d translation matrix
  I = np.identity(4)
  #last row is translation
  I[3,0]=kwargs["x"]
  I[3,1]=kwargs["y"]  
  I[3,2]=kwargs["z"]
  pass


Translate3D(_3D_points_np,x=1,y=2,z=3)

while True:

  #_3D_Plotter(_3D_points_np)

  _3D_points_np=rotate(_3D_points_np,math.radians(1),axis="x")

  _3D_Plotter(_3D_points_np)