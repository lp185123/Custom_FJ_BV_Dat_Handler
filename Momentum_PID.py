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
    def _Tick(self):
        '''each draw, reduce lifetime of item'''
        self.CountDown=self.CountDown-1
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
        return TrailUnit(100,self.position_ND,self.Colour)

    def __post_init__(self):#need this for sorting
        object.__setattr__(self,'sort_index',self.position_ND[2])
        self.Force=np.array([0,0,0],dtype=float)
        self.position_ND=np.array(self.position_ND,dtype=float)
        self.ForceVector_ND=np.array(self.ForceVector_ND,dtype=float)
        self.velocity_ND=np.array(self.velocity_ND,dtype=float)
        self.Radius=int(np.linalg.norm(self.ForceVector_ND)+self.Mass)
        if self.Radius>30:
            self.Radius=30
        if self.Radius<2:
            self.Radius=2
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

            #self.BaseImage = cv2.circle(self.BaseImage, center_coordinates, int(1), NewerColor, -1)
            self.BaseImage[center_coordinates[1],center_coordinates[0],0]=NewerColor[0]
            self.BaseImage[center_coordinates[1],center_coordinates[0],1]=NewerColor[1]
            self.BaseImage[center_coordinates[1],center_coordinates[0],2]=NewerColor[2]
        #can ignore out of bound error
        #warning - trys can be expensive if catching too much 
        except IndexError as error:
            pass
        return self.BaseImage

    def UpdateImage(self,Object,_2Dpoint_if_exists):
        if _2Dpoint_if_exists is not None:
            center_coordinates=tuple(_2Dpoint_if_exists)
        else:
            # Center coordinates
            center_coordinates = tuple([int(x) for x in Object.position_ND[0:2]] )
        Rules=[center_coordinates[0]>0, center_coordinates[1]>0, center_coordinates[0]<self.BaseImage.shape[0],center_coordinates[0]<self.BaseImage.shape[1]]
        if not all(Rules):
            print(f"Out of range to plot body {center_coordinates}")
            return self.BaseImage
        # Radius of circle
        radius = 15#Object.Radius standard radius for all 
        #difference from z0
        Zdepth=int(Object.position_ND[2]/5)
        radius=radius+Zdepth

        if Object.Mass==1:
            radius=radius/3
        if radius<2:
            radius=2
        if radius>100:
            radius=100

        NewColour=(np.array(Object.Colour)/2)
        NewColour=tuple(NewColour.astype(int))


        # Blue color in BGR
        color = Object.Colour
        Zdepth=Zdepth*3
        NewerColor=(Object.Colour[0]+Zdepth,Object.Colour[1]+Zdepth,Object.Colour[2]+Zdepth)
        
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
        if KernelSize%2==0:#kernels can only handle odd numbers
            KernelSize+=1

        kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing - maybe make smoother as such small images
        IncreaseRadius=10
        CentreAreaToSMoothX=center_coordinates[1]
        CentreAreaToSMoothY=center_coordinates[0]
        LeftAreaToSMoothX=CentreAreaToSMoothX-int(radius+IncreaseRadius)
        RightAreaToSMoothX=CentreAreaToSMoothX+int(radius+IncreaseRadius)
        TopAreaToSMoothX=CentreAreaToSMoothY-int(radius+IncreaseRadius)
        LowerAreaToSMoothX=CentreAreaToSMoothY+int(radius+IncreaseRadius)

        #check crop area is in range
        Rules=[LeftAreaToSMoothX>0, TopAreaToSMoothX>0, RightAreaToSMoothX<self.BaseImage.shape[0],LowerAreaToSMoothX<self.BaseImage.shape[1]]
        if not all(Rules):
            print(f"Crop area out of range{LeftAreaToSMoothX,RightAreaToSMoothX,TopAreaToSMoothX,LowerAreaToSMoothX}")
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

def FollowMouse(event, x, y, flags, param):
    #this is specifically for the s39 area selection and will need to be modified
    #for other applications
    #https://pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/#:~:text=Anytime%20a%20mouse%20event%20happens,details%20to%20our%20click_and_crop%20function.
	# grab references to the global variables
    global refPt,GlobalMousePos,ImgView_Resize
    #UI has to be shrunk to fit in window - working images are true size
    x=x*ImgView_Resize
    y=y*ImgView_Resize
    #set global variable
    GlobalMousePos=(int(x), int(y))


def main():

    #create camera object
    MyCamera=Practise_3dView.CameraClass("ForProjectionMatrix",5,0.00551,500,500)
    MyCamera.Translation_Cam_XYZ(-250,-250,609)


    ListBodies=[]
    ListTrails=[]
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", FollowMouse)
    #create objects
    force_Gravity=_ND_Body("User",10.0,[250,250,0],[0,0,0],[70,70,70],0)
    ListBodies.append(force_Gravity)
   
    #add little ones
    for Indexer, CreateBody in enumerate(range (0,2)):
        RandMass=1
        RandForce=RandMass
        ListBodies.append(_ND_Body(str(Indexer),
        RandMass,
        [random.randint(250,300),random.randint(250,300),random.randint(-20,20)],
        [-0.08,0,random.randint(-1,1)*random.random()*random.random()],
        [RandForce,RandForce,RandForce],0))

    for Indexer, CreateBody in enumerate(range (0,1)):
        RandMass=random.randint(200,200)
        RandForce=RandMass#random.randint(1,1)
        ListBodies.append(_ND_Body(str(Indexer),
        RandMass,
        [random.randint(220,220),random.randint(200,200),random.randint(-10,10)],
        [-0.01,0,0],
        [RandForce,RandForce,RandForce],0))
    


    dTime=TimeDiffObject()
    SimImage=SimViewBox(500,500,500)

    #global variable to store positon of users cursor in the opencv window
    global GlobalMousePos
    while True:
        #get time difference to keep it consistent
        #force_Gravity.position_ND=np.array(list(GlobalMousePos) + [0])#only want X and Y but need a Z to satisfy logic
        #time.sleep(0.01)
        dt=10#dTime.Get_dT()
        SimImage.ResetImage()
        #NOTE this should be handled in the dataclass - check out why not and get this workign properly
        
        ListBodies=ListBodies+ListTrails
        ListBodies.sort(key=lambda x:x.position_ND[2])
        ListTrails=[]
        for Index, PhysBody in enumerate(ListBodies):
            if type(PhysBody) is TrailUnit:
                #update trails according to z buffer
                ProjectedPoint=MyCamera.Get2DProjectedPoints(np.expand_dims(PhysBody.position_ND, axis=0))
                OutputSimImage= SimImage.UpdateImage_trails(PhysBody,ProjectedPoint[0])
                PhysBody._Tick()
                if PhysBody.IsLifeTimeFinished()==True:
                    ListBodies.remove(PhysBody)


            #if is a physics body - update image and get a trails object
            if type(PhysBody) is _ND_Body:
                #for doing 1 coordinate at a time, need to expand a dim 
                ProjectedPoint=MyCamera.Get2DProjectedPoints(np.expand_dims(PhysBody.position_ND, axis=0))
                OutputSimImage= SimImage.UpdateImage(PhysBody,ProjectedPoint[0])
                ListTrails.append(copy.deepcopy(PhysBody.GetTrailObject()))
            

        OutputSimImage = cv2.resize(OutputSimImage,(OutputSimImage.shape[1]*2,OutputSimImage.shape[0]*2))
        cv2.imshow("img", OutputSimImage)
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
            #for some reason
            pass 
        #_3DVisLabLib.ImageViewer_Quickv2_UserControl(OutputSimImage,0,False,False)

        for PhysBody in ListBodies:
            if type(PhysBody) is _ND_Body: 
                if all([PhysBody.Name=="User"]):
                    continue#don't process the user object
                for PhysBodyExt in ListBodies:
                    if type(PhysBodyExt) is _ND_Body: 
                        rules=[PhysBodyExt.Name!=PhysBody.Name]
                        if all(rules):
                            PhysBody.SimulateAsPhysicsBody(PhysBodyExt,dt)


if __name__ == "__main__":
    main()