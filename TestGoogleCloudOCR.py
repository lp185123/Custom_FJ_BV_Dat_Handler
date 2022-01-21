###test function to see what we can pass into the cloud ocr object
import VisionAPI_Demo
import io
import os
import _3DVisLabLib
import re
import cv2
import numpy as np
import time
import copy
from pathlib import Path
#sys.path.append('H:/CURRENT/PythonStuff/CUSTOM/')
import _3DVisLabLib
from copy import deepcopy
from collections import namedtuple
import enum
import DatScraper_tool

GoogleCloudOCR=VisionAPI_Demo.CloudOCR()
Default_Input=str( r"C:\Working\FindIMage_In_Dat\OutputTestSNR\SmallSet")
#get list of files in folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(Default_Input)
#filter for .jpg images
ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)

for img in ListAllImages:
    print(img)
    OCR_Result=GoogleCloudOCR.PerformOCR(img,None)
    print(OCR_Result)