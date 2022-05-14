import numpy as np
from dataclasses import dataclass,field
import cv2
import time
import _3DVisLabLib
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
    def __post_init__(self):#need this for sorting
        object.__setattr__(self,'sort_index',self.Mass)
        self.Force=np.array([0,0],dtype=float)

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

    def UpdateImage(self,Coords):
        # Center coordinates
        center_coordinates = tuple([int(x) for x in Coords] )
        # Radius of circle
        radius = 20
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        #draw circle on image
        image = cv2.circle(self.BaseImage.copy(), center_coordinates, radius, color, thickness)

        return image


def main():
    
    #create objects
    force_Gravity=_2d_Force("Gravity",[0,9.81])
    FollowCam=_2d_Body("FollowCam",1,[250,250],[-20,-200])
    dTime=TimeDiffObject()
    SimImage=SimViewBox(500,500)

    while True:
        OutputSimImage= SimImage.UpdateImage(FollowCam.position_2d)
        _3DVisLabLib.ImageViewer_Quickv2_UserControl(OutputSimImage,0,False,False)
    
        dt=dTime.Get_dT()
        print(dt)
        #compute force on object
        FollowCam.Force+=force_Gravity.ForceVector_2d
        #apply newtons second law, compute accelerate
        Acceleration=FollowCam.Force/FollowCam.Mass
        #integrate once to get velocity
        FollowCam.velocity_2d+=Acceleration*dt
        #integrate twice to get position
        FollowCam.position_2d+=FollowCam.velocity_2d* dt
        print(FollowCam.position_2d)


if __name__ == "__main__":
    main()