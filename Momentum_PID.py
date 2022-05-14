import numpy as np
from dataclasses import dataclass,field
import cv2
import time

from scipy import rand
import _3DVisLabLib
import random
#good example

#https://www.reddit.com/r/godot/comments/6pd58n/2d_rocket_simulation/

@dataclass(order=True,repr=False)
class _2d_Body:
    '''2D physics object'''
    sort_index: float =field(init=False)#use this for sorting
    Name:str
    Mass:float
    position_2d:np.array([],dtype=float)
    velocity_2d:np.array([],dtype=float)
    ForceVector_2d:np.array([],dtype=float)
    Radius:float

    def __post_init__(self):#need this for sorting
        object.__setattr__(self,'sort_index',self.Mass)
        self.Force=np.array([0,0],dtype=float)
        self.position_2d=np.array(self.position_2d,dtype=float)
        self.ForceVector_2d=np.array(self.ForceVector_2d,dtype=float)
        self.velocity_2d=np.array(self.velocity_2d,dtype=float)
        self.Radius=int(np.linalg.norm(self.ForceVector_2d))
        if self.Radius>30:
            self.Radius=30
        if self.Radius<1:
            self.Radius=1
        self.Colour=(random.randint(50,255),random.randint(50,255),random.randint(50,255))

    def SimulateAsPhysicsBody(self,InputObject,dT):
        '''reference to simulate using conservation of momentum, 
        so will orbit the object'''
        #get directional vector from Camera to Ball
        #subtract end point from start point
        RelativePos_2me=InputObject.position_2d-self.position_2d
        #get distance from ball to camera
        DistanceObject2me=np.linalg.norm(RelativePos_2me)
        #set to unit vector
        DirectionToObject_unit=RelativePos_2me/DistanceObject2me
        #create custom force vector depending on what we want to do
        ApplicationForceVector=DirectionToObject_unit * (1/(DistanceObject2me+100))**2#sort of equasion for gravity
        #compute force on self
        self.Force=np.array([0,0],dtype=float)
        self.Force+=ApplicationForceVector
        #apply newtons second law, compute accelerate
        Acceleration=self.Force/self.Mass
        #integrate once to get velocity
        self.velocity_2d+=Acceleration*dT
        #FollowCam.velocity_2d=Acceleration*dt #use this for no inertia
        #integrate twice to get position
        self.position_2d+=self.velocity_2d* dT

        return


    def SimulateAsFollowCam(InputObject,dT):
        '''Application specific simulation
        a variation on conservation of momentum etc to smooth out camera following,
        or at least make it less aggressive/sharp '''
        pass

@dataclass(order=True,repr=False)
class _2d_Force:
    '''2D force object'''
    sort_index: str =field(init=False)#use this for sorting
    Name:str
    ForceVector_2d:np.array([],dtype=float)
    def __post_init__(self):#need this for sorting
        object.__setattr__(self,'sort_index',self.Name)

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
    def __init__(self,Height,Width) -> None:
        self.BaseImage=np.zeros([Height,Width],dtype="uint8")

    def ResetImage(self):
        self.BaseImage=np.zeros([self.BaseImage.shape[0],self.BaseImage.shape[1]],dtype="uint8")

    def UpdateImage(self,Object):
        # Center coordinates
        center_coordinates = tuple([int(x) for x in Object.position_2d] )
        # Radius of circle
        radius = Object.Radius
        # Blue color in BGR
        color = Object.Colour
        # Line thickness of 2 px
        thickness = -1
        #draw circle on image
        image = cv2.circle(self.BaseImage, center_coordinates, radius, color, thickness)

        return image

refPt = []#global object to handle recording mouse position
GlobalMousePos=(0,0)#need to be global to handle capturing mouse on opencv UI
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
    ListBodies=[]
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", FollowMouse)
    #create objects
    for Indexer, CreateBody in enumerate(range (0,10)):
        RandMass=random.random()
        RandForce=random.random()*10
        ListBodies.append(_2d_Body(str(Indexer),RandMass,[random.randint(100,400),random.randint(100,400)],[0,0],[RandForce,RandForce],0))
    force_Gravity=_2d_Body("User",0.01,[250,250],[0,0],[30,30],0)
    ListBodies.append(force_Gravity)


    dTime=TimeDiffObject()
    SimImage=SimViewBox(500,500)

    #global variable to store positon of users cursor in the opencv window
    global GlobalMousePos
    while True:
        #get time difference to keep it consistent
        force_Gravity.position_2d=np.array(GlobalMousePos)
        time.sleep(0.01)
        dt=dTime.Get_dT()+10
        SimImage.ResetImage()
        for PhysBody in ListBodies:
            OutputSimImage= SimImage.UpdateImage(PhysBody)

        _3DVisLabLib.ImageViewer_Quickv2_UserControl(OutputSimImage,0,False,False)


        for PhysBody in ListBodies:
            for PhysBodyExt in ListBodies:
                rules=[PhysBodyExt.Name!=PhysBody.Name,
                PhysBody.Name!="User"]
                if all(rules):
                    PhysBody.SimulateAsPhysicsBody(PhysBodyExt,dt)



def main2():
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", FollowMouse)
    #create objects
    force_Gravity=_2d_Force("Gravity",[10,10])
    FollowCam=_2d_Body("FollowCam",0.01,[250,250],[0,0],[0,0],0)
    FollowCam=_2d_Body("FollowCam",0.01,[250,250],[0,0],[0,0],0)
    dTime=TimeDiffObject()
    SimImage=SimViewBox(500,500)

    Counter=0
    #global variable to store positon of users cursor in the opencv window
    global GlobalMousePos
    while True:
        #get time difference to keep it consistent
        force_Gravity.pos
        time.sleep(0.01)
        dt=dTime.Get_dT()+10

        SimImage.ResetImage()
        OutputSimImage= SimImage.UpdateImage(FollowCam.position_2d)
        OutputSimImage= SimImage.UpdateImage(GlobalMousePos)
        _3DVisLabLib.ImageViewer_Quickv2_UserControl(OutputSimImage,0,False,False)

        #get directional vector from Camera to Ball
        #subtract end point from start point
        RelativeBallPos_2Cam=np.array(GlobalMousePos)-FollowCam.position_2d
        #get distance from ball to camera
        DistanceBall2Cam=np.linalg.norm(RelativeBallPos_2Cam)
        #set to unit vector
        DirectionToBall_unit=RelativeBallPos_2Cam/DistanceBall2Cam
        #create custom force vector depending on what we want to do
        ApplicationForceVector=DirectionToBall_unit * (1/(DistanceBall2Cam+100))**2
        #update gravity
        #force_Gravity.ForceVector_2d=#_2d_Force("Gravity",ApplicationForceVector)
        
        #compute force on object
        FollowCam.Force=np.array([0,0],dtype=float)
        FollowCam.Force+=ApplicationForceVector * force_Gravity.ForceVector_2d
        #apply newtons second law, compute accelerate
        Acceleration=FollowCam.Force/FollowCam.Mass
        #integrate once to get velocity
        FollowCam.velocity_2d+=Acceleration*dt
        #FollowCam.velocity_2d=Acceleration*dt
        #integrate twice to get position
        FollowCam.position_2d+=FollowCam.velocity_2d* dt
        
        if int(FollowCam.position_2d[0]) not in range (0,SimImage.BaseImage.shape[0]):
            FollowCam=_2d_Body("FollowCam",0.01,[250,250],[0,0],[0,0],0)
        if int(FollowCam.position_2d[1]) not in range (0,SimImage.BaseImage.shape[1]):
            FollowCam=_2d_Body("FollowCam",0.01,[250,250],[0,0],[0,0],0)

if __name__ == "__main__":
    main()