from enum import auto
import numpy as np
from dataclasses import dataclass,field
import cv2
import time

from scipy import rand
import _3DVisLabLib
import random
import Practise_3dView
import copy

@dataclass(order=True,repr=False)
class TrailUnit:
    '''trail object - this is best done with a matrix - this will probably be very slow'''
    sort_index: float =field(init=False)#use this for sorting
    CountDown: int
    position_ND:np.array([],dtype=float)
    Colour: tuple

    def __post_init__(self):#need this for sorting
        object.__setattr__(self,'sort_index',self.position_ND[2])
    def _Tick(self,dTime):
        '''each draw, reduce lifetime of item'''
        self.CountDown=self.CountDown-dTime
    def IsLifeTimeFinished(self):
        if self.CountDown<=0:
            return True
        else:
            return False


@dataclass(order=True,repr=False)
class _ND_Body:
    '''ND physics object, ND = n dimension, so can change to 2d physics if required'''
    sort_index: float =field(init=False)#use this for sorting
    Name:str
    Mass:float
    position_ND:np.array([],dtype=float)
    velocity_ND:np.array([],dtype=float)
    ForceVector_ND:np.array([],dtype=float)
    Radius:float

    def GetTrailObject(self):
        '''Return an object to handle drawing trails complying with Z buffer'''
        return TrailUnit(1000,self.position_ND,self.Colour)

    def __post_init__(self):#need this for sorting
        object.__setattr__(self,'sort_index',self.position_ND[2])
        self.Force=np.array([0,0,0],dtype=float)
        self.position_ND=np.array(self.position_ND,dtype=float)
        self.ForceVector_ND=np.array(self.ForceVector_ND,dtype=float)
        self.velocity_ND=np.array(self.velocity_ND,dtype=float)
        #self.Radius=int(np.linalg.norm(self.ForceVector_ND)+self.Mass)
        #if self.Radius>30:
        #    self.Radius=30
        #if self.Radius<2:
        #    self.Radius=2
        self.Colour=(random.randint(0,255),random.randint(0,255),random.randint(0,255))

    def SimulateAsPhysicsBody(self,InputObject,dT):
        '''reference to simulate using conservation of momentum, 
        so will orbit the object'''
        #get directional vector from Camera to Ball
        #subtract end point from start point
        RelativePos_2me=InputObject.position_ND-self.position_ND
        #get distance from ball to camera
        DistanceObject2me=np.linalg.norm(RelativePos_2me)
        #set to unit vector
        DirectionToObject_unit=RelativePos_2me/DistanceObject2me
        #create custom force vector depending on what we want to do
        ApplicationForceVector=DirectionToObject_unit * (1/(DistanceObject2me+100))**2#sort of equasion for gravity
        #compute force on self
        #self.Force=np.array([0,0],dtype=float)
        self.Force=ApplicationForceVector * InputObject.ForceVector_ND
        #apply newtons second law, compute accelerate
        Acceleration=self.Force/self.Mass
        #integrate once to get velocity
        self.velocity_ND+=Acceleration*dT
        #integrate twice to get position
        self.position_ND+=self.velocity_ND* dT

        return


    def SimulateAsFollowCam(InputObject,dT):
        '''Application specific simulation
        a variation on conservation of momentum etc to smooth out camera following,
        or at least make it less aggressive/sharp '''
        pass

# @dataclass(order=True,repr=False)
# class _ND_Force:
#     '''ND force object'''
#     sort_index: str =field(init=False)#use this for sorting
#     Name:str
#     ForceVector_ND:np.array([],dtype=float)
#     def __post_init__(self):#need this for sorting
#         object.__setattr__(self,'sort_index',self.Name)

def mapFromTo(x,a,b,c,d):
    # x:input value; 
    # a,b:input range
    # c,d:output range
    # y:return value
        y=(x-a)/(b-a)*(d-c)+c
        return y

class TimeDiffObject:
    '''Class to create Dt for physics simulations'''
    def __init__(self) -> None:
        self.StartTime=time.perf_counter()
        self.StopTime=time.perf_counter()
    def Get_dT(self)-> float:
        self.StopTime=time.perf_counter()
        Difference=self.StopTime-self.StartTime
        #restart counter
        self.StartTime=time.perf_counter()
        return Difference

class SimViewBox():
    '''Display particles'''
    def __init__(self,Height,Width,ZDepth) -> None:
        self.BaseImage=np.zeros([Height,Width,3],dtype="uint8")
        self.ZDepth=ZDepth
        self.GridImg=cv2.imread(r"C:\Working\GridPerspective.jpg",cv2.IMREAD_GRAYSCALE)
        self.GridImg = cv2.cvtColor(self.GridImg,cv2.COLOR_GRAY2RGB)
        self.GridImg = cv2.resize(self.GridImg,(self.BaseImage.shape[0],self.BaseImage.shape[1]))
    def ResetImage(self):
        #self.BaseImage=cv2.imread(r"C:\Working\GridPerspective.jpg",cv2.IMREAD_GRAYSCALE)
        #self.BaseImage = cv2.cvtColor(self.BaseImage,cv2.COLOR_GRAY2RGB)
        #self.BaseImage = cv2.resize(self.BaseImage,(800,800))
        self.BaseImage=np.zeros([self.BaseImage.shape[0],self.BaseImage.shape[1],3],dtype="uint8")

    def UpdateImage_trails(self,TrailObject,_2Dpoint_if_exists):
        if _2Dpoint_if_exists is not None:
            center_coordinates=tuple(_2Dpoint_if_exists)
        else:
            # Center coordinates
             center_coordinates = tuple([int(x) for x in TrailObject.position_ND[0:2]] )

        try:
           
            LifeTimeColour=1#TrailObject.CountDown/100
            Zdepth=int(TrailObject.position_ND[2]/5)
            NewerColor=(int(TrailObject.Colour[0]*LifeTimeColour),
            int(TrailObject.Colour[1]*LifeTimeColour),
            int(TrailObject.Colour[2]*LifeTimeColour))
            
            CircleRadius=int(mapFromTo(Zdepth,-10,20,1,5))
            CircleRadius=max(0,CircleRadius)

            DistanceLoss=int(mapFromTo(Zdepth,-10,0,0.1,1))
            DistanceLoss=max(0.1,DistanceLoss)
            DistanceLoss=min(0.9,DistanceLoss)

            NewerColor=(NewerColor[0]*DistanceLoss,NewerColor[1]*DistanceLoss,NewerColor[2]*DistanceLoss)


            #self.BaseImage = cv2.circle(self.BaseImage, center_coordinates, CircleRadius, NewerColor, -1)
            self.BaseImage[center_coordinates[1],center_coordinates[0],0]=NewerColor[0]
            self.BaseImage[center_coordinates[1],center_coordinates[0],1]=NewerColor[1]
            self.BaseImage[center_coordinates[1],center_coordinates[0],2]=NewerColor[2]
        #can ignore out of bound error
        #warning - trys can be expensive if catching too much 
        except IndexError as error:
            pass
        return self.BaseImage

    def UpdateImage_ForSpheres(self,Object,_2Dpoint_if_exists,CameraObject,_2DProjectedRadius):
        '''update the 2D camera image with sphere subjects
        need 2D camera projection points of sphere centre and sphere radius'''

        if _2Dpoint_if_exists is not None:
            center_coordinates=tuple(_2Dpoint_if_exists)
        else:
            # Center coordinates
            center_coordinates = tuple([int(x) for x in Object.position_ND[0:2]] )
        Rules=[(CameraObject.Is_3DPointBehindCamera(Object.position_ND)) is False,center_coordinates[0]>0, center_coordinates[1]>0, center_coordinates[0]<self.BaseImage.shape[0],center_coordinates[0]<self.BaseImage.shape[1]]
        if not all(Rules):
            #print(f"Out of range to plot body {center_coordinates}")
            return self.BaseImage

        #get distance from input object to camera object centre
        Dist2Camera=CameraObject.GetDistanceFromCamera(Object.position_ND)
        # Radius of circle
        radius =int((300-Dist2Camera)/10)#_2DProjectedRadius#Object.Radius standard radius for all 
        #print(f"Dist2Camera{Dist2Camera}")
        if radius<2:
            radius=2
        if radius>100:
            radius=100

        #difference from z0
        Zdepth=int(Dist2Camera)
        #radius=radius+Zdepth
        

        NewColour=(np.array(Object.Colour)/2)
        NewColour=tuple(NewColour.astype(int))

        # Blue color in BGR
        color = Object.Colour
        DistanceFade=(mapFromTo(Zdepth,600,400,1,0.1))
        DistanceFade=max(0.1,DistanceFade)
        DistanceFade=min(1,DistanceFade)
        DistanceFade=1#debug
        NewerColor=(int(Object.Colour[0]*DistanceFade),
        int(Object.Colour[1]*DistanceFade),
        int(Object.Colour[2]*DistanceFade))
        
        # Line thickness of 2 px
        thickness = -1
        
        #try:
        #create small portion of image to create physics body and blur it
        #SmallFrame_body=np.array()


        #get copy of image
        #CopyImage=self.BaseImage.copy()

        #create mask - slightly bigger than current radius?
        Mask=np.zeros([self.BaseImage.shape[0],self.BaseImage.shape[1],3],dtype="uint8")
        image_Zlayer = cv2.circle(Mask, center_coordinates, int(radius), NewerColor, thickness)
        #self.BaseImage=cv2.circle(self.BaseImage, center_coordinates, int(2), NewerColor, thickness)
        #return self.BaseImage

        #blur z layer
        KernelSize=int(mapFromTo(Zdepth,-50,30,9,1))
        KernelSize=max(1,KernelSize)
        KernelSize=min(21,KernelSize)
        KernelSize=3
        if KernelSize%2==0:#kernels can only handle odd numbers
            KernelSize+=1

        kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing - maybe make smoother as such small images
        IncreaseRadius=5
        CentreAreaToSMoothX=center_coordinates[1]
        CentreAreaToSMoothY=center_coordinates[0]
        LeftAreaToSMoothX=CentreAreaToSMoothX-int(radius+IncreaseRadius)
        RightAreaToSMoothX=CentreAreaToSMoothX+int(radius+IncreaseRadius)
        TopAreaToSMoothX=CentreAreaToSMoothY-int(radius+IncreaseRadius)
        LowerAreaToSMoothX=CentreAreaToSMoothY+int(radius+IncreaseRadius)

        #check crop area is in range
        Rules=[LeftAreaToSMoothX>0, TopAreaToSMoothX>0, RightAreaToSMoothX<self.BaseImage.shape[0],LowerAreaToSMoothX<self.BaseImage.shape[1]]
        if not all(Rules):
            #print(f"Crop area out of range{LeftAreaToSMoothX,RightAreaToSMoothX,TopAreaToSMoothX,LowerAreaToSMoothX}")
            return self.BaseImage


        AreaCheck=image_Zlayer[LeftAreaToSMoothX:RightAreaToSMoothX,TopAreaToSMoothX:LowerAreaToSMoothX,:]
        BlurredFramedBody=cv2.filter2D(AreaCheck,-1,kernel)
        BlurredFramedBody_gray = cv2.cvtColor(BlurredFramedBody, cv2.COLOR_BGR2GRAY)
        (_, BinarisedImage) = cv2.threshold(BlurredFramedBody_gray, 50, 255, cv2.THRESH_BINARY)
        #BinarisedImage_inverted = cv2.bitwise_not(BinarisedImage)
        BinarisedImage_3Channel = cv2.cvtColor (BinarisedImage, cv2.COLOR_GRAY2BGR)
        #BinarisedImage_3Channel_inv = cv2.cvtColor (BinarisedImage_inverted, cv2.COLOR_GRAY2BGR)

        

        #cut out hole in original image
        OriginalArea=self.BaseImage[LeftAreaToSMoothX:RightAreaToSMoothX,TopAreaToSMoothX:LowerAreaToSMoothX,:].copy()

    
        self.BaseImage[LeftAreaToSMoothX:RightAreaToSMoothX,TopAreaToSMoothX:LowerAreaToSMoothX,:]=0#(BinarisedImage_3Channel_inv*255)

        plop=np.subtract(OriginalArea.astype(np.int16),BinarisedImage_3Channel.astype(np.int16))
        plop=np.clip(plop, 0, 255)
        plop=plop.astype(np.uint8)
        plop=np.add(plop,BlurredFramedBody)

        #_3DVisLabLib.ImageViewer_Quickv2(BinarisedImage_3Channel,0,True,False)
        #_3DVisLabLib.ImageViewer_Quickv2(BinarisedImage_3Channel_inv,0,True,False)
        #_3DVisLabLib.ImageViewer_Quickv2(OriginalArea,0,True,False)
        #_3DVisLabLib.ImageViewer_Quickv2(plop,0,True,False)


        self.BaseImage[LeftAreaToSMoothX:RightAreaToSMoothX,TopAreaToSMoothX:LowerAreaToSMoothX,:] = plop
        #now add to canvas with alpha blend
        #TestImage = cv2.addWeighted(TestImage, ParameterObject.AlphaBlend, ProcessedImage, 1.0, 0.0)

        return self.BaseImage#self.BaseImage

        #add back onto canvas

        # except Exception as e:
        #     print(e)
        #     self.BaseImage = cv2.circle(self.BaseImage , center_coordinates, int(radius), NewerColor, thickness)
        #     return self.BaseImage

refPt = []#global object to handle recording mouse position
GlobalMousePos=(250,250)#need to be global to handle capturing mouse on opencv UI
ImgView_Resize=1.3#HD image doesnt fit on screen
MouseClick_LeftDown=False
def FollowMouse(event, x, y, flags, param):
    #this is specifically for the s39 area selection and will need to be modified
    #for other applications
    #https://pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/#:~:text=Anytime%20a%20mouse%20event%20happens,details%20to%20our%20click_and_crop%20function.
	# grab references to the global variables
    global refPt,GlobalMousePos,ImgView_Resize,MouseClick_LeftDown
    #UI has to be shrunk to fit in window - working images are true size
    x=x*ImgView_Resize
    y=y*ImgView_Resize
    #set global variable
    GlobalMousePos=(int(x), int(y))
    if event == cv2.EVENT_LBUTTONDOWN:
        MouseClick_LeftDown=True
    if event == cv2.EVENT_LBUTTONUP:
        MouseClick_LeftDown=False
def euclid_dist(t1, t2):
    return np.sqrt(np.sum(np.square(t1-t2)))
def GetProjectedRadius(SphericalPhysBody,CameraObject: Practise_3dView.CameraClass ,_2D_CentrePoint):
    '''Get circle raidus to represent 3D spherical object in 2D projected view'''
    
    # #align circle with normal of camera
                #from centre of object, use rotation matrix of camera to offset radius in plane parallel to camera
                #get any vector except forward vector
    #             CameraPlane=CameraObject.RotationMatrix_R[0:3,1]
    #             RadiusOffset_3D=SphericalPhysBody.position_ND+(CameraPlane*SphericalPhysBody.Radius)
    #             #print(f"RadiusOffset_3D{RadiusOffset_3D}")
                

    #             #make sure vectors are aligned, should be "flat" to each other - but can flip polarity
    #             CheckV1=RadiusOffset_3D-SphericalPhysBody.position_ND
    #             CheckV1=CheckV1/np.linalg.norm(CheckV1)
    #             CheckV2=CameraObject.RotationMatrix_R[0:3,1]
    #             CheckV2=CheckV2/np.linalg.norm(CheckV2)
    #             DotProduct=np.dot(CheckV1,CheckV2)
    #             #print(f"DotProduct_ProjectCheck {DotProduct}")
    #             if not all([abs(DotProduct)>0.98,abs(DotProduct)<1.02]):
    #                 print(f"GetProjectedRadius Error, dotproduct {DotProduct}")
    #                 #raise Exception("Error with GetProjectedRadius check DotProduct")


    #circle will "look at" centre of camera
    #create a vector orthogonal to look-at vector
    Look_atVector=CameraObject.TranslationMatrix- SphericalPhysBody.position_ND
    #need arbitray vector as long as makes valid cross product, make sure can't align or will break
    ArbitraryVec=np.array([-Look_atVector[1],1,-Look_atVector[2]])
    CrossProd=np.cross(Look_atVector,ArbitraryVec)
    #check orthogonal (dot product=0)
    DotProduct_check=np.dot(Look_atVector,CrossProd)
    if round(DotProduct_check,2)!=0:
        print(f"Look_atVector {Look_atVector} ArbitraryVec {ArbitraryVec} DotProduct_check {DotProduct_check} ")
        raise Exception("Error with GetProjectedRadius, dotproduct not zero for orthogonal vector")

    CrossProd_norm=CrossProd / np.linalg.norm(CrossProd)
    RadiusOffset_3D=SphericalPhysBody.position_ND+(CrossProd_norm*SphericalPhysBody.Radius)

    #check the 3d distance - this should always be the radius unless we are using the wrong camera axis
    #or the camera is aligned with an axis 
    Check3D_Distance=euclid_dist(SphericalPhysBody.position_ND, RadiusOffset_3D)
    if abs(Check3D_Distance-SphericalPhysBody.Radius)>0.1:
        #pass
        raise Exception("Error with GetProjectedRadius Check3D_Distance- possibly incorrect camera axis ")


    #Calculate 2D projecti on of radius offset
    ProjectedPoint=CameraObject.Get2DProjectedPoints(np.expand_dims(RadiusOffset_3D, axis=0))
    if (ProjectedPoint.shape)!=(1,2):
        raise Exception("Error with GetProjectedRadius ProjectedPoint- probably bad cross product ")
    #calculate difference of position between 2D centre point projection and 2D radius offset
    #this figure can be used to create a circle in the draw loop
    dist = euclid_dist(_2D_CentrePoint,ProjectedPoint)
    return dist,RadiusOffset_3D

def CalculateCollisionSphere(Obj1,Obj2,Elastic=True):
    #https://physics.stackexchange.com/questions/598480/calculating-new-velocities-of-n-dimensional-particles-after-collision/598524#598524
    pass
def main():

    #create camera object
    MyCamera=Practise_3dView.CameraClass("ForProjectionMatrix",2,0.00551,500,500)
    MyCamera.TranslationMatrix=np.array([0,0,-150])
    ListBodies=[]
    ListTrails=[]
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", FollowMouse)
    #create objects
    force_Gravity=_ND_Body("User",10.0,[0,0,0],[0,0,0],[370,370,370],5)
    ListBodies.append(force_Gravity)
   
    #add little ones
    for Indexer, CreateBody in enumerate(range (0,10)):
        RandMass=1
        RandForce=2
        ListBodies.append(_ND_Body(str(Indexer),
        RandMass,
        [random.randint(-100,100),random.randint(-100,100),random.randint(-100,100)],
        [random.randint(-3,3)*0.1,random.randint(-3,3)*0.1,random.randint(1,3)*0.1],
        [RandForce,RandForce,RandForce],5))
        #[random.randint(-3,3)*0.01,random.randint(-3,3)*0.01,random.randint(-3,3)*0.01],
    #[-0.08*0,0,random.randint(-3,3)*random.random()*random.random()*0],
    for Indexer, CreateBody in enumerate(range (0,0)):
        RandMass=65
        RandForce=RandMass#random.randint(1,1)
        ListBodies.append(_ND_Body(str(Indexer),
        RandMass,
        [random.randint(-10,10),random.randint(-10,10),random.randint(-10,10)],
        [random.randint(-5,5)*0.01,random.randint(-3,3)*0.01,random.randint(-3,3)*0.01],
        [RandForce,RandForce,RandForce],5))
    


    dTime=TimeDiffObject()
    SimImage=SimViewBox(500,500,500)

    #global variable to store positon of users cursor in the opencv window
    global GlobalMousePos,MouseClick_LeftDown
    
    Counter=0
    ImgCounter=0
    dt=5
    while Counter<9999999:
        OutputSimImage=None
        Counter+=1
        #force_Gravity.position_ND=np.array([0,0,100-Counter-Counter])
        #MyCamera.RotateView_AroundLocation([10,0,0],[0,0,0])
        #MyCamera.RotateView_AroundLocation([0,3,0],[0,0,0])
        #MyCamera.RotateView_AroundLocation([0,0,2],[0,0,0])
        #MyCamera.Translation_Cam_XYZ(0,0,-2)
        #print(f"{Counter}", end="\r")
        
        if MouseClick_LeftDown==True:
            #force_Gravity.position_ND=np.array([0,0,10-Counter])
            MyCamera.RotateView_AroundLocation([0,5,0],[0,0,0])
            if Counter%3==0 : dt-=1
        else:
            if Counter%3==0 : dt+=1#dTime.Get_dT()
        dt=max(0,dt)
        dt=min(5,dt)


        #get time difference to keep it consistent
        #force_Gravity.position_ND=np.array(list(GlobalMousePos) + [0])#only want X and Y but need a Z to satisfy logic
        #time.sleep(0.01)
        SimImage.ResetImage()

        #refresh projection matrix
        MyCamera.BuildProjectionMatrix()

        #NOTE this should be handled in the dataclass - check out why not and get this workign properly
        
        ListBodies=ListBodies+ListTrails
        #ListBodies.sort(key=lambda x:x.position_ND[2])
        ListBodies.sort(key=lambda x:MyCamera.GetDistanceFromCamera(x.position_ND))
        ListBodies.reverse()#we want to draw from farthest away object to closest 

        ListTrails=[]
        DebugArray=[]
        for Index, PhysBody in enumerate(ListBodies):
            if type(PhysBody) is TrailUnit:
                #update trails according to z buffer
                ProjectedPoint=MyCamera.Get2DProjectedPoints(np.expand_dims(PhysBody.position_ND, axis=0))
                OutputSimImage= SimImage.UpdateImage_trails(PhysBody,ProjectedPoint[0])
                PhysBody._Tick(dt)
                if PhysBody.IsLifeTimeFinished()==True:
                    ListBodies.remove(PhysBody)

            #if is a physics body - update image and get a trails object
            if type(PhysBody) is _ND_Body:
                #if object is behind the camera, don't draw
                
                    
                #we are operating with spheres, so need to get the projected circle radius
                #need to extend radius from centre point out in same plane as camera
                
                #map 3D cartesian point to 2D with view frustrum
                #for doing 1 coordinate at a time, need to expand a dim 
                ProjectedPoint=MyCamera.Get2DProjectedPoints(np.expand_dims(PhysBody.position_ND, axis=0))
                #specifically for drawing spheres - need to get radius projection from point
                #using a vector from the camera rotation matrix - so the radius is plotted flat to camera
                
                _2DProjectedRadius,_3DCheckRadius=GetProjectedRadius(SphericalPhysBody=PhysBody,CameraObject=MyCamera,_2D_CentrePoint=ProjectedPoint)
                #print(f"_2DProjectedRadius{_2DProjectedRadius}")
                #generate image specifcally for spherical graphics
                OutputSimImage= SimImage.UpdateImage_ForSpheres(PhysBody,ProjectedPoint[0],MyCamera,_2DProjectedRadius)
                
                #print(f"Check3D_Distance {MyCamera.GetDistanceFromCamera(PhysBody.position_ND)}")
                #print(f"_2DProjectedRadius {_2DProjectedRadius}")
                #if no time don't get trail object - even if trail granted per distance unit
                if dt!=0: ListTrails.append(copy.deepcopy(PhysBody.GetTrailObject()))

                #debug 
                #DebugArray.append(_3DCheckRadius)
                DebugArray.append(PhysBody.position_ND)
            
        if Counter%1==0:
            if OutputSimImage is not None:
                OutputSimImage = cv2.resize(OutputSimImage,(OutputSimImage.shape[1]*2,OutputSimImage.shape[0]*2))
                cv2.imshow("img", OutputSimImage)
                if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                    #for some reason
                    pass 

        for PhysBody in ListBodies:
            if type(PhysBody) is _ND_Body: 
                if all([PhysBody.Name=="User"]):
                    continue#don't process the user object
                for PhysBodyExt in ListBodies:
                    if type(PhysBodyExt) is _ND_Body: 
                        rules=[PhysBodyExt.Name!=PhysBody.Name]
                        if all(rules):
                            PhysBody.SimulateAsPhysicsBody(PhysBodyExt,dt)

        #debug output 
        #draw camera rotation
        DebugArray.append(MyCamera.TranslationMatrix + (MyCamera.RotationMatrix_R[0:3,0]*5))
        DebugArray.append(MyCamera.TranslationMatrix + (MyCamera.RotationMatrix_R[0:3,1]*5))
        DebugArray.append(MyCamera.TranslationMatrix + (MyCamera.RotationMatrix_R[0:3,2]*15))
        DebugArray.append(MyCamera.TranslationMatrix)
        #DebugArray.append([0,0,0])
        XYZplotFilePath=r"C:\Working\TempImages\OrbitTempImgs\IMG00" + str(ImgCounter) + ".jpg"
        ImgCounter+=1
        ImgplotFilePath=r"C:\Working\TempImages\OrbitTempImgs\IMG00" + str(ImgCounter) + ".jpg"
        ImgCounter+=1
        
        #Practise_3dView._3D_Plotter(np.array(DebugArray),XYZplotFilePath,None,None)
        #cv2.imwrite(ImgplotFilePath, OutputSimImage)
if __name__ == "__main__":
    main()