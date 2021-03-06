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
import random
import enum


def rotateAroundAxis(X, theta, axis='x'):
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


class UserOperationStrings(enum.Enum):
    Xplus="w"
    Xminus="s"
    Yplus="a"
    Yminus="d"
    Zplus="p"
    Zminus="l"

    FocalLengthPlus="["
    FocalLengthMinus="]"

    Cam_LockRotateX="x"
    Cam_LockRotateY="y"
    Cam_LockRotateZ="z"

    Obj_RotateX="b"
    Obj_RotateY="n"
    Obj_RotateZ="m"



    def MapKeyPressToString(self,keypress):
        if keypress in self.MapKeyPress_to_String:
            return self.MapKeyPress_to_String[keypress]
        else:
            return None
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
  
  def _MyPurpose(self):
    print("SUper class")
  
  def AddRandomMovement(self,Magnitude):

    for Index,Point in enumerate(self._3dPoints):
      self._3dPoints[Index]=self._3dPoints[Index]+round(random.random())


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
  



  def rotate_ClassMethod(self, theta, axis='x'):
    self._3dPoints=rotateAroundAxis(self._3dPoints, theta, axis)

    # '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
    # c, s = np.cos(theta), np.sin(theta)
    # if axis == 'x': self._3dPoints= np.dot(self._3dPoints, np.array([
    #   [1.,  0,  0],
    #   [0 ,  c, -s],
    #   [0 ,  s,  c]
    # ]))
    # elif axis == 'y': self._3dPoints= np.dot(self._3dPoints, np.array([
    #   [c,  0,  -s],
    #   [0,  1,   0],
    #   [s,  0,   c]
    # ]))
    # elif axis == 'z': self._3dPoints= np.dot(self._3dPoints, np.array([
    #   [c, -s,  0 ],
    #   [s,  c,  0 ],
    #   [0,  0,  1.],
    # ]))


#test inherited class

class _3D_data_Experiment(_3D_data):
  def _MyPurpose(self):
    #calls the superclass function, we can do whatever we like with that
    super()._MyPurpose()


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
  TranslationMatrix_t=np.array([0, 0, 90])#a 3D translation vector describing the position of the camera center
  RotationMatrix_R=np.eye(3, k=0)#orientation of the camera
  #experiment with rotation matrix input
  RotationMatrix_R[0,:]=np.array([1,0,0])
  RotationMatrix_R[1,:]=np.array([0,1, 0])
  RotationMatrix_R[2,:]=np.array([0,0, 1])
  IntrinsicMatrix_K=None# calibration matrix
  ExtrinsicMatrix=None
  _3D_worldCoordinates=None

  def __post_init__(self):#need this for sorting
    object.__setattr__(self,'sort_index',self.Focallength_mm)#if we want to freeze object
    
  @property
  def TranslationMatrix(self):
      return self.TranslationMatrix_t
  @TranslationMatrix.setter
  def TranslationMatrix(self, value):
        self.TranslationMatrix_t = value

  def Translation_Cam_XYZ(self,x,y,z):
    self.TranslationMatrix_t[0]=self.TranslationMatrix_t[0]+x
    self.TranslationMatrix_t[1]=self.TranslationMatrix_t[1]+y
    self.TranslationMatrix_t[2]=self.TranslationMatrix_t[2]+z


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

  def CalculateExtrinsicMatrix_MovingCam(self):
    '''we want the camera to move around the world rather than vice-versa, so formula  [R|t] can be modified to
    [Rc C]^-1.
    replace [R|t] with [R|-RC]??.  Inverse of a valid rotation matrix is just the transpose'''
    #add column to  Rotation matrix - note the tranpose which is an inverse if a rotation matrix
    #t=-RC
    Rinv=self.RotationMatrix_R.T
    C=self.TranslationMatrix_t
    Rinv_mul_C=np.matmul(Rinv,C)
    #calculate [R|-RC]
    b = np.append(self.RotationMatrix_R, [[Rinv_mul_C[0]],[Rinv_mul_C[1]],[Rinv_mul_C[2]]], axis = 1)

    self.ExtrinsicMatrix=b

  def CalculateExtrinsicMatrix(self):
    '''create extrinsic properties, translation & rotation [R|t]'''
    
    #add column to  Rotation matrix -
    b = np.append(self.RotationMatrix_R, [[self.TranslationMatrix_t[0]],[self.TranslationMatrix_t[1]],[self.TranslationMatrix_t[2]]], axis = 1)
    #b = np.pad(self.RotationMatrix, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    #lame code to add extra column - definitely better ways of doing this
    #b[0,3]=self.TranslationVector[0]
    #b[1,3]=self.TranslationVector[1]
    #b[2,3]=self.TranslationVector[2]
    self.ExtrinsicMatrix=b

  def RotateView_AroundLocation(self,XYZRotation_degrees:np.array,XYZRotateOrigin:np.array):
    '''camera will spin around area of keeping area in frame'''

    TestList=[Index for Index, x in enumerate(XYZRotation_degrees) if x>0]
    if len(TestList)>1:
      raise Exception("RotateView_AroundLocation multiple axis input not implemented yet")
    #maybe changing the up vector depending on input axis to rotate will
    #help with gimbal lock problem - this is a fudge TODO
    UpVectorDict={}
    UpVectorDict[0]=([ 0, 1, 0 ])
    UpVectorDict[1]=([ 0, 1, 0 ])
    UpVectorDict[2]=([ 0, 1,0 ])
    #multiply arbitrary up vector by current rotation advice
    #this might help problem we have
    #UpVector_inputAxis=self.RotationMatrix_R@UpVectorDict[TestList[0]]
    UpVector_inputAxis=UpVectorDict[TestList[0]]

    #https://computergraphics.stackexchange.com/questions/4668/correcting-my-look-at-matrix-so-that-it-works-on-non-camera-objects

    #  "Using a constant vector such as (0,1,0), which is the only recommendation I've ran 
    #  across is no good. You end up rotating the bank axis for no reason, and near singularity
    #   points (attitude +-90) the camera spins wildly.  My solution was using [currentRotationMatrix * (0,1,0)] 
    #   as the up vector. It makes sure only the heading and attitude are changed for the lookAt so the bank 
    #   is not altered, and it has stood up very well through testing. I highly suggest you mention it on the
    #    lookAt page."https://www.euclideanspace.com/maths/algebra/vectors/lookat/index.htm

    #try using [currentRotationMatrix * (0,1,0)] as up vector instead of just a static up vector
    #static up vector was causing issues with singularities (maybe - or something at 90 degrees)

    #TODO maybe try rodrigues
    #rot vec
    #rmat,jac=cv2.rodrigues(rot_vec)
    #angles=cv2.RDecompose3x3

    #translate rotate origin to zero

    #apply rotation(s)
    #TODO this is not nice
    self.TranslationMatrix_t=rotateAroundAxis(self.TranslationMatrix_t, math.radians(XYZRotation_degrees[0]), axis="x")
    self.TranslationMatrix_t=rotateAroundAxis(self.TranslationMatrix_t, math.radians(XYZRotation_degrees[1]), axis="y")
    self.TranslationMatrix_t=rotateAroundAxis(self.TranslationMatrix_t, math.radians(XYZRotation_degrees[2]), axis="z")
    self.TranslationMatrix_t=np.round(self.TranslationMatrix_t,1)
    #untranslate position back
    #camera direction will now be a unit vector of inverted position
   # self.RotationMatrix_R=self.TranslationMatrix_t*-1

    

    lookat=XYZRotateOrigin#np.array([0,0,0])
    vz = lookat - self.TranslationMatrix_t
    vz = vz / (np.linalg.norm(vz) + 1e-16)#normalise - works for vectors with mag=0
    vx = np.cross(UpVector_inputAxis, vz)#cross with up vector (?)
    vx = vx / (np.linalg.norm(vx) + 1e-16)#normalise - works for vectors with mag=0
    vy = np.cross(vz,vx)

    #4x4 matrix will support all affine transforms, including rotation and translation
    RotMat=np.zeros([4,4])
    RotMat[0:3,0]=vx
    RotMat[0:3,1]=vy
    RotMat[0:3,2]=vz
    #RotMat[0,0:3]=vx
    #RotMat[1,0:3]=vy
    #RotMat[2,0:3]=vz
    RotMat[3,3]=1
    #test if a rotation matrix (only need 3x3 section for rotation only)
    
    
    self.RotationMatrix_R=RotMat[0:3,0:3]
    
    #try using rotation vector- magnitude is the z rotation (?)
    #rrmat,jac=cv2.Rodrigues(vx)
    #self.RotationMatrix_R=rrmat

    # print(self.RotationMatrix_R)
    if _3DVisLabLib.isRotationMatrix(RotMat[0:3,0:3])==False:
      print(f"bad rotation matrix{self.RotationMatrix_R}")
      #raise Exception("RotateView_AroundLocation isRotationMatrix, rotation matrix not valid",self.RotationMatrix_R)
    
    else:
      pass
      #print(np.round(self.RotationMatrix_R,2))
      #print(np.degrees(_3DVisLabLib.rotationMatrixToEulerAngles(RotMat[0:3,0:3])))


  @staticmethod
  def euclid_dist(t1, t2):
    return np.sqrt(np.sum(np.square(t1-t2)))
  def GetDistanceFromCamera(self,_3DInputPoint:np.array)->float:
    '''get distance of current point from camera'''
    return self.euclid_dist(self.TranslationMatrix_t,_3DInputPoint)
    return np.linalg.norm(self.TranslationMatrix_t-_3DInputPoint)

  def BuildProjectionMatrix(self):
    '''Refresh projection matrix if any intrinsic or extrinsic camera details have changed'''
    self.CalculateIntrinsicMatrix()
    self.CalculateExtrinsicMatrix_MovingCam()

  def GetProjectedPoints(self,InputPoint:np.array)->np.array:
    '''Calculate projection matrix P=K [r|T] and return 2d points
    this should be done in another function after proof of principle
    Expecting input of 3D homogenous coordinates'''
    

    #convert 3d point to homogenous coordinates (just add a 1)
    Homogenous3DPt=np.append(InputPoint,([1]))
    
    scaledPixelCoords = np.matmul(np.matmul(self.IntrinsicMatrix_K,self.ExtrinsicMatrix),Homogenous3DPt)
    #print(scaledPixelCoords)#,end="\r")
    #do we test for clipping here?

    if scaledPixelCoords[-1]!=0:
      CartesianCoords=scaledPixelCoords/scaledPixelCoords[-1]#divide by last element to convert from homogenous to cartesian
    else:
      CartesianCoords=scaledPixelCoords
      
    
    return CartesianCoords

  def Get2DProjectedPoints(self,InputPoints:np.array):
    '''return 2D coordinates'''
    SuccessPnt=0
    _2dPoint_list=[]

    for _3dPoint in InputPoints:
      _2dPoint=self.GetProjectedPoints(_3dPoint)
      try:
        _2dPoint_list.append((int(_2dPoint[0:2][0]),int(_2dPoint[0:2][1])))
        SuccessPnt=SuccessPnt+1
      except IndexError as error:
        #for pixels out of range we can ignore
        pass
      except Exception as e:
        print("error plotting 2d projected points in Get2DProjectedImage")
        print(e)
    return np.array(_2dPoint_list)
    
  @staticmethod
  def mapFromTo(x,a,b,c,d):
    # x:input value; 
    # a,b:input range
    # c,d:output range
    # y:return value
        y=(x-a)/(b-a)*(d-c)+c
        return y

  def Is_3DPointBehindCamera(self,_3DInputPoint:np.array)->bool:
    '''return a boolean if the 3D point is clipped from view (behind camera)'''
    #need 3d camera location, normal describing camera plane, 3d input point

    #take product of camera normal CN and (plot point-camera location)
    #TODO define this axis in initialisation
    CameraForwardVector=self.RotationMatrix_R[0:3,2]
    DotProd=np.dot(CameraForwardVector, (self.TranslationMatrix_t-_3DInputPoint))
    #print(f"DotProd {DotProd} _3DInputPoint{_3DInputPoint} self.TranslationMatrix_t{self.TranslationMatrix_t}")#,end="\r")
    if DotProd <0:
      return False
    else:
      return True

  def Get2DProjectedImage(self,InputPoints:np.array):


    '''return opencv object of 3d input points on 2d plane, add dimming using distance from camera - fudge'''
    #TODO fudge code for distance from camera - is there a formal way to decide this
    Image=np.zeros([self.Image_Width,self.Image_Height,3],dtype="uint8")
    SuccessPnt=0
    _2dPoint_list=[]
    _2dPoint_Distance=[]
    for _3dPoint in InputPoints:
      #use *_ in case we want to add more output
      _2dPoint=self.GetProjectedPoints(_3dPoint)
      try:
        #get distance of point from camera object
        Dist_from_cam=np.linalg.norm(self.TranslationMatrix_t-_3dPoint)
        #print(f"{Dist_from_cam}")#,end="\r")
        #map arbitrary dimness distance
        ColElement=int(self.mapFromTo(Dist_from_cam,150,10,0,255))
        #set limits for maximum colour value
        ColElement=max(0,ColElement)
        ColElement=min(255,ColElement)
        ColElement=255
        Image[int(_2dPoint[0:2][0]),int(_2dPoint[0:2][1]),:]=ColElement
        _2dPoint_list.append((int(_2dPoint[0:2][0]),int(_2dPoint[0:2][1])))
        _2dPoint_Distance.append(ColElement)
        SuccessPnt=SuccessPnt+1
      except IndexError as error:
        #for pixels out of range we can ignore
        pass
      except Exception as e:
        print("error plotting 2d projected points in Get2DProjectedImage")
        print(e)
    #if len(InputPoints)>0:
    #  print(f"{int((SuccessPnt/len(InputPoints))*100)}% points projected to 2D")

    #draw lines in opencv
    line_thickness = 1
    for Index,(_2dPoint,_2dPointDistance) in enumerate(zip(_2dPoint_list,_2dPoint_Distance)):
      if Index>=len(_2dPoint_list)-1:
        break
      cv2.line(Image, (_2dPoint_list[Index][1],_2dPoint_list[Index][0]), (_2dPoint_list[Index+1][1],_2dPoint_list[Index+1][0]), (_2dPointDistance,_2dPointDistance,_2dPointDistance), thickness=line_thickness)
      pass
    return Image
# Focallength_mm= 26 
# Pixel_size_sensor_mm= 0.00551 
# Image_Height=1000
# Image_Width=500

# focalX=Focallength_mm/Pixel_size_sensor_mm
# focalY=Focallength_mm/Pixel_size_sensor_mm
# cx=Image_Width/2
# cy=Image_Height/2

# CameraMatrix_Intrinsic=np.eye(3)
# CameraMatrix_Intrinsic[2,2]=1
# CameraMatrix_Intrinsic[0,0]=focalX
# CameraMatrix_Intrinsic[1,1]=focalY
# CameraMatrix_Intrinsic[0,2]=cx
# CameraMatrix_Intrinsic[1,2]=cy

#TranslationVector=np.array([0, 0, 1])

#extrinsic camera properties, rotation and translation
#is this just an identity matrix?
#RotationMatrix=np.eye(3)
#RotationMatrix=np.eye(3, k=0),
#TranslationVector=np.array([0, 0, 0,1])
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
  
    #plt.xlim([-500,500])
    #plt.ylim([-500,500])
    OrbitRange=300
    ax.set_xlim(-OrbitRange,OrbitRange)
    ax.set_ylim(-OrbitRange,OrbitRange)
    ax.set_zlim(-OrbitRange,OrbitRange)

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
    else:
      plt.show()
      plt.close()
def main():
  OutputPath=r"C:\Working\testOutput"

  #Result = input("Press enter to clear folder:" + str(OutputPath))
  #if Result=="":
  #  _3DVisLabLib.DeleteFiles_RecreateFolder(OutputPath)


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

  def GenerateSquare():
    _side= np.arange(-10, 11, 1)
    _StaticAxis= np.full((len(_side)),10 )
    out_arr_top = np.column_stack((_side, _StaticAxis))
    _StaticAxis= np.full((len(_side)),-10 )
    out_arr_lower = np.column_stack((_side, _StaticAxis))
    _StaticAxis= np.full((len(_side)),10 )
    out_arr_left = np.column_stack((_StaticAxis, _side))
    _StaticAxis= np.full((len(_side)),-10 )
    out_arr_right = np.column_stack((_StaticAxis, _side))

    #must be better way of doing this
    OutputArray=[]
    for a,b,c,d in zip (out_arr_top,out_arr_lower,out_arr_left,out_arr_right):
      OutputArray.append(tuple(a))
      OutputArray.append(tuple(b))
      OutputArray.append(tuple(c))
      OutputArray.append(tuple(d))

    return OutputArray


  #draw cube
  _3D_points=[]
  #z axis
  #for Z_Index in range(-10,10,1):
  #create square
  _Square=GenerateSquare()
  for elem in _Square:
    _3D_points.append((elem[0],elem[1],-10))
  for elem in _Square:
    _3D_points.append((elem[0],elem[1],10))
  #generate vertical sides
  #have a go making this smarter
  for Elem in range (-10,10):
    randomint=random.randint(-2,2)
    _3D_points.append((-10+randomint,-10+randomint,Elem))
    _3D_points.append((10+randomint,-10+randomint,Elem))
    _3D_points.append((-10,10,Elem))
    _3D_points.append((10,10,Elem))
  





  #convert to numpy array for matrix operations
  _3D_points_np=np.array(_3D_points)

  #NEXT - remember matrix multiplication rules
  #need to convert to homogenous vecotrs (x,y,z,1), do translation, then convert back from homogenous (divinde by last element)

  #try dataclass object
  _5pointFace=_3D_data_Experiment("5pointface",_3D_points_np)
  _5pointFace._MyPurpose()

  MyCamera=CameraClass("ForProjectionMatrix",3,0.00551,1000,1000)



  #Translate3D(_3D_points_np,x=1,y=1,z=1)

  DebugDraw=False
  while True:
    DebugDraw=False
    #refresh projection matrix
    MyCamera.BuildProjectionMatrix()

    Image=MyCamera.Get2DProjectedImage(_5pointFace._3dPoints)
    UserRequest=_3DVisLabLib.ImageViewer_Quickv2_UserControl(Image,0,True,False)
    #move camera controls
    if UserRequest==UserOperationStrings.Xplus.value:
      MyCamera.Translation_Cam_XYZ(1,0,0)
    if UserRequest==UserOperationStrings.Xminus.value:
      MyCamera.Translation_Cam_XYZ(-1,0,0) 
    if UserRequest==UserOperationStrings.Yplus.value:
      MyCamera.Translation_Cam_XYZ(0,1,0)
    if UserRequest==UserOperationStrings.Yminus.value:
      MyCamera.Translation_Cam_XYZ(0,-1,0) 
    if UserRequest==UserOperationStrings.Zplus.value:
      MyCamera.Translation_Cam_XYZ(0,0,1)
    if UserRequest==UserOperationStrings.Zminus.value:
      MyCamera.Translation_Cam_XYZ(0,0,-1) 
    
    #rotate camera around position
    if UserRequest==UserOperationStrings.Cam_LockRotateX.value:
      #_5pointFace.rotate_ClassMethod(math.radians(5),axis="x")
      MyCamera.RotateView_AroundLocation([5,0,0],[0,0,0])
      DebugDraw=True
    if UserRequest==UserOperationStrings.Cam_LockRotateY.value:
      #_5pointFace.rotate_ClassMethod(math.radians(5),axis="y")
      MyCamera.RotateView_AroundLocation([0,5,0],[0,0,0])
      DebugDraw=True
    if UserRequest==UserOperationStrings.Cam_LockRotateZ.value:
      #_5pointFace.rotate_ClassMethod(math.radians(5),axis="z")
      MyCamera.RotateView_AroundLocation([0,0,5],[0,0,0])
      DebugDraw=True
    
    if UserRequest==UserOperationStrings.Obj_RotateX.value:
      _5pointFace.rotate_ClassMethod(math.radians(5),axis="x")
      
    if UserRequest==UserOperationStrings.Obj_RotateY.value:
      _5pointFace.rotate_ClassMethod(math.radians(5),axis="y")
      
    if UserRequest==UserOperationStrings.Obj_RotateZ.value:
      _5pointFace.rotate_ClassMethod(math.radians(5),axis="z")
      

    if DebugDraw==True:
      #as we revolve around the object we will draw into the 3d view to quickly check motion of camera
      #flip the transformation array otherwise if we see it straight on its hard to decipher
      OldShape=_5pointFace._3dPoints.shape
      TempOb=np.append(_5pointFace._3dPoints,np.round(MyCamera.TranslationMatrix/2)[::-1])
      TempOb=TempOb.reshape(OldShape[0]+1,OldShape[1])
      _5pointFace._3dPoints=TempOb
    #change camera focal length
    if UserRequest==UserOperationStrings.FocalLengthPlus.value:
      MyCamera.Focallength_mm=MyCamera.Focallength_mm+1
    if UserRequest==UserOperationStrings.FocalLengthMinus.value:
      MyCamera.Focallength_mm=MyCamera.Focallength_mm-1


    
  Counter=0
  while True:
    Filepath=OutputPath + "\\00" + str(Counter) + ".jpg"
    Filepath_2d=OutputPath + "\\00" + str(Counter) + "_2d.jpg"
    Filepath_2dProject=OutputPath + "\\_2DProject00" + str(Counter) + "_2d.jpg"


    #Projected_2dPoints=[]
    ##for _3dPoint in _5pointFace._3dPoints:
    #  Projected_2dPoints.append(MyCamera.GetProjectedPoints(_3dPoint))
    #Projected_2dPoints=np.array(Projected_2dPoints)


    Image=MyCamera.Get2DProjectedImage(_5pointFace._3dPoints)
    cv2.imwrite(Filepath_2dProject, Image)

    _5pointFace.Translate3D_classmethod(x=0,y=0,z=0)
    _5pointFace.rotate_ClassMethod(math.radians(5),axis="z")
    _5pointFace.rotate_ClassMethod(math.radians(5),axis="x")
    _5pointFace.rotate_ClassMethod(math.radians(5),axis="y")
    MyCamera.Translation_Cam_XYZ(0,0,int(-Counter/10))
    #MyCamera.Focallength_mm=MyCamera.Focallength_mm+Counter
    #_5pointFace.AddRandomMovement(19)
    
    Counter=Counter+1
    if Counter>100:
      break

    #eigvalues, eigvectors=_5pointFace.GetPCA_SetToMean()
    #_3D_Plotter(_5pointFace._3dPoints,Filepath,False,False)#,eigvalues=eigvalues,eigvectors=eigvectors)
    #_3D_Plotter(Projected_2dPoints,Filepath_2d,False,False)


  #projection matrix 
  #https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html

if __name__ == "__main__":
    main()
