from mimetypes import init
from tkinter.font import names
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import random
import _3DVisLabLib
from dataclasses import dataclass,field
from numpy.linalg import eig
OutputPath=r"C:\Working\testOutput"

Result = input("Press enter to clear folder:" + str(OutputPath))
if Result=="":
  _3DVisLabLib.DeleteFiles_RecreateFolder(OutputPath)

#set order =True so we can compare
@dataclass(order=True,repr=False)
class _3D_data:
  #provide type hints
  sort_index: int =field(init=False)#use this for sorting
  name:str
  _3dPoints:np.array
  def __post_init__(self):#need this for sorting
    #self.sort_index=self._3D_points_np.size
    object.__setattr__(self,'sort_index',self._3dPoints.size)#if we want to freeze object

  @staticmethod
  def Translate3D_static(xyz_coords,**kwargs):
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
    TranslatedHomogenous= HomogenousMatrix@I
    TranslatedCartesian=TranslatedHomogenous[:,0:3]
    return TranslatedCartesian
  
  def Translate3D_classmethod(self,**kwargs):
    #need to add concatenate input matrix
    #Ones=np.ones(xyz_coords.shape[0])
    #array2 = np.append(xyz_coords, [Ones], axis = 1)
    HomogenousMatrix=np.ones([self._3dPoints.shape[0],self._3dPoints.shape[1]+1])
    HomogenousMatrix[:self._3dPoints.shape[0],:self._3dPoints.shape[1]]=self._3dPoints[:,:]
    #now have an extra column for each xyz point - homogenous projection
    #create translation matrix
    #3d translation matrix
    I = np.identity(4)
    #last row is translation
    I[3,0]=kwargs["x"]
    I[3,1]=kwargs["y"]  
    I[3,2]=kwargs["z"]
    TranslatedHomogenous= HomogenousMatrix@I
    TranslatedCartesian=TranslatedHomogenous[:,0:3]#watch out here as remember to divide by last number to convert back to cartesian

    self._3dPoints= TranslatedCartesian
  

#how to get vector of PCA
#   If you need a rotation matrix representing an orientation, we can choose the axis
#    in which the volume distribution of the object is highest (normalised first eigenvector -
#     that is the eigenvector associated with the largest eigenvalue) as the first column of the matrix.

# For the 2nd column of the matrix choose the 2nd eigenvector but you have to subtract from it
#  its projection onto the 1st eigenvector so that it is orthogonal to the first. To calculate its 
#  projection you can use the dot product - if the eigenvectors are already normalised you can just
#   use the dot product to calculate the length of the vector to subtract: so dot product the two vectors
#    and multiply the 1st vector by the dot product, then subtract the resulting vector from the 1st eigenvector.

# For the 3rd column there will only be one choice left - the cross product of the two calculated above.

  @staticmethod
  def GetPrincipleAxis(InputArray:np.array)->None:
    '''test get principle axis'''
    # calculate the mean of each column (each axis)
    M = np.mean(InputArray.T, axis=1)
    # center columns by subtracting column means
    C = InputArray - M
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)
    # eigendecomposition of covariance matrix
    values, vectors = eig(V)
    # project data
    #Once chosen, data can be projected into the subspace via matrix multiplication.
    #P = B^T . A
    #Where A is the original data that we wish to project, B^T
    # is the transpose of the chosen principal components and P is the projection of A.
    P = vectors.T.dot(C.T)
    return None

  def GetPCA_SetToMean(self):
    '''test get principle axis'''
    # calculate the mean of each column (each axis)
    M = np.mean(self._3dPoints.T, axis=1)
    # center columns by subtracting column means
    C = self._3dPoints - M
    #destructively set array to centre
    self._3dPoints=C
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)
    # eigendecomposition of covariance matrix
    values, vectors = eig(V)
    # project data
    #Once chosen, data can be projected into the subspace via matrix multiplication.
    #P = B^T . A
    #Where A is the original data that we wish to project, B^T
    # is the transpose of the chosen principal components and P is the projection of A.
    P = vectors.T.dot(C.T)
    return values, vectors
  

  @staticmethod
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

  def rotate_ClassMethod(self, theta, axis='x'):
    '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x': self._3dPoints= np.dot(self._3dPoints, np.array([
      [1.,  0,  0],
      [0 ,  c, -s],
      [0 ,  s,  c]
    ]))
    elif axis == 'y': self._3dPoints= np.dot(self._3dPoints, np.array([
      [c,  0,  -s],
      [0,  1,   0],
      [s,  0,   c]
    ]))
    elif axis == 'z': self._3dPoints= np.dot(self._3dPoints, np.array([
      [c, -s,  0 ],
      [s,  c,  0 ],
      [0,  0,  1.],
    ]))


# The camera matrix P is a 4x3 matrix of the form P = K[R t]:

# K is a 3x3 matrix containing the intrinsic parameters (principal point and focal length in pixels)
# [R t] is a 3x4 matrix obtained by concatenating R, a 3x3 matrix representing the rotation from the 
# camera frame to the world frame, and t, a 3-vector which represents the position of the origin of 
# the world in the camera frame.
# This means that the parameters you have, which seem to be the position of the camera in the world
#  frame, have to be inverted. The inverse of [R t] is [R' t'] where R' = inverse(R) = transpose(R) and t' = -inverse(R)t.

# You would first have to know how to compute the 3x3 camera rotation matrix from the three angles 
# you have, and there are many possible representations of a rotation matrix from three angles. The
#  most common are yaw/pitch/roll, and Euler angles with all possible rotation orders.

# The 3x3 intrinsics matrix K is [[f 0 cx][0 f cy][0 0 1]], where f = 26/0.00551 = 4719 and (cx,cy) 
# is the principal point, which you can take as the center of the image (4288/2,2848/2).

# Then to compute the homography (3x3 matrix) that goes from the plane at world height Z0 to your 
# image, you multiply P by (X,Y,Z0,1), which gives you an expression of the form Xv1 + Yv2 + v3 where
#  v1, v2, and v3 are 3-vectors. The matrix H=[v1 v2 v3] is the homography you are looking for. The 8
#   coefficients for PIL.Image.transform should be the first 8 coefficients from that matrix, divided by the 9th one.


#build camera matrix
#rotation matrix
#translation matrix

@dataclass(order=True,repr=False)
class CameraClass:
  sort_index: int =field(init=False)#use this for sorting
  name:str
  Focallength_mm:float
  Pixel_size_sensor_mm: float
  Image_Height:int
  Image_Width:int

  #these are variables (usually in __init__ as "self")
  TranslationMatrix_t=np.array([0, 0, 50])#a 3D translation vector describing the position of the camera center
  RotationMatrix_R=np.eye(3, k=0)#orientation of the cmaera
  IntrinsicMatrix_K=None# calibration matrix
  ExtrinsicMatrix=None
  _3D_worldCoordinates=None

  def __post_init__(self):#need this for sorting
    object.__setattr__(self,'sort_index',self.Focallength_mm)#if we want to freeze object
  def CalculateIntrinsicMatrix(self):
    '''Create instrinsic camera matrix, internal details'''
    focalX=self.Focallength_mm/self.Pixel_size_sensor_mm
    focalY=self.Focallength_mm/self.Pixel_size_sensor_mm
    cx=self.Image_Width/2
    cy=self.Image_Height/2

    CameraMatrix_Intrinsic=np.eye(3)
    CameraMatrix_Intrinsic[2,2]=1
    CameraMatrix_Intrinsic[0,0]=focalX
    CameraMatrix_Intrinsic[1,1]=focalY
    CameraMatrix_Intrinsic[0,2]=cx
    CameraMatrix_Intrinsic[1,2]=cy
    self.IntrinsicMatrix_K=CameraMatrix_Intrinsic
  def CalculateExtrinsicMatrix(self):
    '''create extrinsic properties, translation & rotation [R|T]'''
    #add column to  Rotation matrix -
    b = np.append(self.RotationMatrix_R, [[self.TranslationMatrix_t[0]],[self.TranslationMatrix_t[1]],[self.TranslationMatrix_t[2]]], axis = 1)
    #b = np.pad(self.RotationMatrix, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    #lame code to add extra column - definitely better ways of doing this
    #b[0,3]=self.TranslationVector[0]
    #b[1,3]=self.TranslationVector[1]
    #b[2,3]=self.TranslationVector[2]
    self.ExtrinsicMatrix=b

  def GetProjectedPoints(self,InputPoint:np.array)->np.array:
    '''Calculate projection matrix P=K [r|T]
    this should be done in another function after proof of principle
    Expecting input of 3D homogenous coordinates'''
    self.CalculateIntrinsicMatrix()
    self.CalculateExtrinsicMatrix()

    #convert 3d point to homogenous coordinates (just add a 1)
    Homogenous3DPt=np.append(InputPoint,([1]))
    
    scaledPixelCoords = np.matmul(np.matmul(self.IntrinsicMatrix_K,self.ExtrinsicMatrix),Homogenous3DPt)
    #scaledPixelCoords = np.matmul(np.matmul(self.ExtrinsicMatrix,Homogenous3DPt),self.IntrinsicMatrix_K)
    return scaledPixelCoords




  




Focallength_mm= 26 
Pixel_size_sensor_mm= 0.00551 
Image_Height=1000
Image_Width=500

focalX=Focallength_mm/Pixel_size_sensor_mm
focalY=Focallength_mm/Pixel_size_sensor_mm
cx=Image_Width/2
cy=Image_Height/2

CameraMatrix_Intrinsic=np.eye(3)
CameraMatrix_Intrinsic[2,2]=1
CameraMatrix_Intrinsic[0,0]=focalX
CameraMatrix_Intrinsic[1,1]=focalY
CameraMatrix_Intrinsic[0,2]=cx
CameraMatrix_Intrinsic[1,2]=cy


#TranslationVector=np.array([0, 0, 1])


#extrinsic camera properties, rotation and translation
#is this just an identity matrix?
#RotationMatrix=np.eye(3)
RotationMatrix=np.eye(3, k=0),
TranslationVector=np.array([0, 0, 0,1])
#-	First camera do A * [R|T]
#-	First position set ROTATION MATRIX IDENTITY and/as well TRANSLATION VECTOR [0 ,0,0,1]^t
#-	Second position use output of R/T from StereoCalibrate
#P = np.column_stack((np.matmul(CameraMatrix_Intrinsic,RotationMatrix), TranslationVector))
#add column to new Rotation matrix - then multiply by camera matrix
#b = np.pad(RotationMatrix, ((0, 0), (0, 1)), mode='constant', constant_values=0)
#lame code to add extra column - definitely better ways of doing this
#b[0,3]=TranslationVector[0]
#b[1,3]=TranslationVector[1]
#b[2,3]=TranslationVector[2]
#P=np.mat(CameraMatrix_Intrinsic)*np.mat(b)
#3d rotation


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
def _3D_Plotter(Input_np_array,Filepath,Drawpaths,CrossProducts,**kwargs):#3d matlab plot 3d plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    sequence_containing_x_vals = list(Input_np_array[:,0])
    sequence_containing_y_vals = list(Input_np_array[:,1])
    sequence_containing_z_vals = list(Input_np_array[:,2])
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
  
    if Drawpaths is True:
      x_start,y_start,z_start=0,0,0
      for Index,(x,y,z) in enumerate(zip(sequence_containing_x_vals,sequence_containing_y_vals,sequence_containing_z_vals)):
        if Index==0:
          x_start,y_start,z_start=x,y,z
          continue
        plt.plot([x_start,x],[y_start,y],[z_start,z],label='v')
        x_start,y_start,z_start=x,y,z
   
    if 'eigvectors' in kwargs and 'eigvalues' in kwargs:
      #user wants to draw eigendecomp stuff into render
      PrincipleVector=kwargs['eigvectors'][0] * 300#kwargs['eigvalues'][0]
       #* -kwargs['eigvalues'][0]
      plt.plot([0,PrincipleVector[0]],[0,PrincipleVector[1]],[0,PrincipleVector[2]],label='v')

    if CrossProducts is True:
      for Index,(x,y,z) in enumerate(zip(sequence_containing_x_vals,sequence_containing_y_vals,sequence_containing_z_vals)):
        if Index==0:
          x_start,y_start,z_start=x,y,z
          continue
        #get cross product of last and current vectors - remember right hand rule!
        crossProd=np.cross([x_start,y_start,z_start],[x,y,z])
        plt.plot([0,crossProd[0]],[0,crossProd[1]],[0,crossProd[2]],label='v')

    #plt.show()
    if Filepath is not None:
      plt.savefig(Filepath)

_3D_points=[]
_3D_points.append((-150.0, -150.0, -125.0))
_3D_points.append((150.0, -150.0, -125.0))
_3D_points.append(( 0.0, 0.0, 0.0))
_3D_points.append((0.0, -330.0, -65.0))
_3D_points.append((-225.0, 170.0, -135.0))
_3D_points.append(( 225.0, 170.0, -135.0))

#create random field of 3d points with a rough shape
_3D_points=[]
for Index in range(0,10):
  _3D_points.append((Index*random.random(),Index*random.random(),4*Index*random.random()))

#convert to numpy array for matrix operations
_3D_points_np=np.array(_3D_points)


#NEXT - remember matrix multiplication rules
#need to convert to homogenous vecotrs (x,y,z,1), do translation, then convert back from homogenous (divinde by last element)

#try dataclass object
_5pointFace=_3D_data("5pointface",_3D_points_np)
  

MyCamera=CameraClass("ForProjectionMatrix",26,0.00551,1000,500)

for _3dPoint in _5pointFace._3dPoints:
  print(_3dPoint)
  print(MyCamera.GetProjectedPoints(_3dPoint))




#Translate3D(_3D_points_np,x=1,y=1,z=1)

Counter=0
while True:

  _5pointFace.Translate3D_classmethod(x=0,y=0,z=0)
  _5pointFace.rotate_ClassMethod(math.radians(5),axis="z")
  #_5pointFace.rotate_ClassMethod(math.radians(5),axis="x")
  #_5pointFace.rotate_ClassMethod(math.radians(5),axis="y")
  
  Filepath=OutputPath + "\\00" + str(Counter) + ".jpg"
  
  Counter=Counter+1
  if Counter>100:
    break

  eigvalues, eigvectors=_5pointFace.GetPCA_SetToMean()
  _3D_Plotter(_5pointFace._3dPoints,Filepath,True,True)#,eigvalues=eigvalues,eigvectors=eigvectors)



#projection matrix 
#https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html