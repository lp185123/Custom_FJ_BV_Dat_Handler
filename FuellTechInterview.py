import numpy as np
import copy
from difflib import Match
import numpy as np
import time
import random
import cv2
import _3DVisLabLib
import scipy.stats as stats
import statistics
import matplotlib.pyplot as pl
from statistics import mean 
import math
import shutil
import math
import matplotlib.pyplot as plt
import json









def calculate_number_of_distraction_episodes(*, landmarks, looking_straight):
    
    
    AnglesOverTime=dict()

    Feature1='left_eye'
    Feature2='right_eye'
    Feature3='forehead'
    Feature4='chin'

    BasePose_feature1=np.array(looking_straight[Feature1])
    BasePose_feature2=np.array(looking_straight[Feature2])
    BasePose_feature3=np.array(looking_straight[Feature3])
    BasePose_feature4=np.array(looking_straight[Feature4])
    Vector1=BasePose_feature1-BasePose_feature2
    Vector2=BasePose_feature3-BasePose_feature4
    BaseNormal=np.cross(Vector1,Vector2)
    BaseNormal=BaseNormal/np.linalg.norm(BaseNormal)

    #test base case
    DotProd=np.dot(BaseNormal,BaseNormal) 
    Angle=math.acos(DotProd)
    print(f"Base Angle is {Angle}")

    for FaceID, FacePose in enumerate(landmarks):
        #
        TestPose_feature1=np.array(FacePose[Feature1])
        TestPose_feature2=np.array(FacePose[Feature2])
        TestPose_feature3=np.array(FacePose[Feature3])
        TestPose_feature4=np.array(FacePose[Feature4])
        TestVector1=TestPose_feature1-TestPose_feature2
        TestVector2=TestPose_feature3-TestPose_feature4
        TestNormal=np.cross(TestVector1,TestVector2)
        TestNormal=TestNormal/np.linalg.norm(TestNormal)

        #now calculate dot product from Base normal and test normal
        DotProd=np.dot(BaseNormal,TestNormal) 
        Angle=math.acos(DotProd)
        print(f"Dot product angle is {math.degrees(Angle)}")
        #DotProd_mag=math.sqrt(np.dot(DotProd,DotProd))
        #DotProd_Norm=DotProd/DotProd_mag

        AnglesOverTime[FacePose['timestamp']]=math.degrees(Angle)


        test=1
        pass

    MaxAngle_deg=20
    MaxPeriod_sec=0


    DistractionCount=0
    TimeStampStart=None
    TimeStampEnd=None
    NoOfDistractions=0
    DistractionAlarmOn=False

    for Indexer,(timestamp,facepose_deg) in enumerate(AnglesOverTime.items()):

        if abs(facepose_deg)>=MaxAngle_deg:
            DistractionCount+=1
            if TimeStampStart is None:
                TimeStampStart=timestamp
        else:
            TimeStampStart=None
            DistractionCount=0
            DistractionAlarmOn=False
            continue
        
        TimeStampEnd=timestamp
        TotalTime=TimeStampEnd-TimeStampStart

        if TotalTime>=2:
            if DistractionAlarmOn is False:
                NoOfDistractions+=1
                DistractionAlarmOn = True
                print(f"Distraction Warning")

    return NoOfDistractions


def main():
    with open(r"C:\Users\LP185123\Downloads\test_data.json") as fh:
        test_data = json.load(fh)

    results = [
        test_case.pop('num_distraction_episodes')
            == calculate_number_of_distraction_episodes(**test_case)
        for test_case in test_data
    ]

    successes = sum(results)
    all_cases = len(test_data)
    assert successes == all_cases, f'{successes}/{all_cases}'


if __name__ == '__main__':
    main()
