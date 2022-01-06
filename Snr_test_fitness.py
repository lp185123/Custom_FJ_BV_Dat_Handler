import cv2
import numpy as np
import os
import sys
import imutils
import glob
#print(str( sys.exec_prefix.split(os.sep)[-1] ))
import math
import pickle
import math
import _3DVisLabLib
#from scipy.stats import norm
from random import randint
import time
import copy
import glob
import matplotlib.pyplot as plt
import traceback
import random
import pytesseract
from pytesseract import*
import BV_DatReader_Lib
from pathlib import Path
import difflib
import enum


#tips
#https://nanonets.com/blog/ocr-with-tesseract/
#tesseract ocr how to install
#https://pythonforundergradengineers.com/how-to-install-pytesseract.html#:~:text=Point%20pytesseract%20at%20your%20tesseract%20installation%20Create%20a,of%20the%20string%20that%20defines%20the%20file%20location.
#found C:\ProgramData\Anaconda3\envs\threedvislab\Library\bin\tesseract.exe - lab pc
#C:\Users\3DVisLab\.conda\envs\threedvislab\Library\bin\tesseract.exe - alienware laptop

class ImageProcessEnums(enum.Enum):
    #universal enums to standardise parameters between modules
    ResizeX="ResizeX"
    ResizeY="ResizeY"
    kernel="kernel"
    ConfidenceCutoff="ConfidenceCutoff"
    Canny="Canny"
    AdapativeThreshold="AdapativeThreshold"
    MedianBlurDist="MedianBlurDist"
    GausSize_Threshold="GausSize_Threshold"
    SubtractMean="SubtractMean"
    PSM="PSM"

def ResizeImage(InputImage, ResizePercentX,ResizePercentY):
    #appy scaling
    width = int(InputImage.shape[1] * ResizePercentX/ 100)
    height = int(InputImage.shape[0] * ResizePercentY / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(InputImage, dim, interpolation = cv2.INTER_AREA)

class SNR_Parameters():
    def __init__(self):
        self.ResizeX=100
        self.ResizeY=100
        self.kernel = np.ones((1,1), np.uint8)
        self.ConfidenceCutoff=80
        self.Canny=0
        self.AdapativeThreshold=255
        self.MedianBlurDist=5
        self.GausSize_Threshold=11
        self.SubtractMean=4
        self.tessedit_char_whitelist='0123456789'#use if necessary
        self.AlphaBlend=1
        self.CropPixels=0
        self.Mirror=True
        self.PSM=3#to throw error if not initialised
        #psm is page segment modes - see pytessaract manual
        #self.config='--oem 3 --psm 6 -c load_system_dawg=0 load_freq_dawg=0 load_punc_dawg=0'
        #self.config = '--oem 3 --psm 6' #known to work
        self.config="DUMMY"#to throw error if not initialised
        #oem modes
        # 0    Legacy engine only.
        # 1    Neural nets LSTM engine only.
        # 2    Legacy + LSTM engines.
        # 3    Default, based on what is available.
        #PSM modes
        # 0    Orientation and script detection (OSD) only.
        # 1    Automatic page segmentation with OSD.
        # 2    Automatic page segmentation, but no OSD, or OCR.
        # 3    Fully automatic page segmentation, but no OSD. (Default)
        # 4    Assume a single column of text of variable sizes.
        # 5    Assume a single uniform block of vertically aligned text.
        # 6    Assume a single uniform block of text.
        # 7    Treat the image as a single text line.
        # 8    Treat the image as a single word.
        # 9    Treat the image as a single word in a circle.
        # 10    Treat the image as a single character.
        # 11    Sparse text. Find as much text as possible in no particular order.
        # 12    Sparse text with OSD.
        # 13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

    def SetConfig(self):
        self.config='--oem 3 --psm ' + str(self.PSM) + ' -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ load_system_dawg=0 load_freq_dawg=0 load_punc_dawg=0'

    def Housekeep(self):
        self.ResizeX=int(self.ResizeX)
        self.ResizeY=int(self.ResizeY)
        self.ConfidenceCutoff=int(self.ConfidenceCutoff)
        self.AdapativeThreshold=int(self.AdapativeThreshold)
        self.MedianBlurDist=int(self.MedianBlurDist)
        self.Canny=int(self.Canny)

def CheckStringSimilarity(string1,string2):
     s = difflib.SequenceMatcher(lambda x: x == " ",string1,string2)
     similiarity_ratio=round(s.ratio(), 3)
     return similiarity_ratio

class TestSNR_Fitness():
#class which loads images into memory used to test fitness of input parameters
    def __init__(self):
        print("initialising Tesseract OCR")
        WhereTesseract=r'C:\Users\LP185123\Anaconda3\envs\threeDvis_withtesseract\Library\bin\tesseract.exe'
        WhereTesseract=r'X:\bin\tesseract.exe'
        pytesseract.tesseract_cmd = WhereTesseract
        os.environ['OMP_THREAD_LIMIT'] = '20'
        self.Known_SNR=False
        self.Known_SNR_string=""
        self.Known_Snr_Fitness=0

    

    def RunSNR_With_Parameters_multiprocess(self,MultiProcessSNR_Object):
        #multiprocess version of SNR fitness check - everything is in same namespace to avoid complications
        Name=MultiProcessSNR_Object[0]
        SNRparams=MultiProcessSNR_Object[1]
        ListFilenames=MultiProcessSNR_Object[2]
        for Index, Element in enumerate(ListFilenames):
            pass

            




    def RunSNR_With_Parameters(self,ImagePath,ParameterObject,TestImage=None,SkipOcr=False):

        #make sure no invalid values
        ParameterObject.Housekeep()
        ParameterObject.SetConfig()

        if TestImage is None:
            #load image
            TestImage=cv2.imread(ImagePath,cv2.IMREAD_GRAYSCALE)

        #get snr string if available (embedded in filename between "[" and "]")
        self.Known_SNR=False
        self.Known_SNR_string=""
        self.Known_Snr_Fitness=0

        try:

            
            Get_SNR_string=ImagePath.split("[")#delimit
            Get_SNR_string=Get_SNR_string[-1]#get last element of delimited string
            Get_SNR_string=Get_SNR_string.split("]")#delimit
            Get_SNR_string=Get_SNR_string[0]
            if Get_SNR_string is not None:
                if len(Get_SNR_string)>5:#TODO magic number
                    self.Known_SNR_string=Get_SNR_string
                    self.Known_SNR=True

        except Exception as e: 
            print("error extracting known snr string from file ",ImagePath )
            print(repr(e)) 

        #resize
        TestImage=ResizeImage(TestImage,ParameterObject.ResizeX,ParameterObject.ResizeY)

        
        #perform crop here
        Height,Width=TestImage.shape
        y=ParameterObject.CropPixels
        x=ParameterObject.CropPixels
        h=Height-ParameterObject.CropPixels
        w=Width-ParameterObject.CropPixels
        TestImage = TestImage[y:y+h, x:x+w]



        if ParameterObject.Mirror==True:
            #double up image in case its flipped 
            FlippedImage = cv2.flip(TestImage, -1)
            #create blank image twice the height
            #shape 0 is X shape 1 is Y
            blank_image = (np.ones((TestImage.shape[0]*2,TestImage.shape[1]), dtype = np.uint8))*255
            blank_image[0:TestImage.shape[0],:]=TestImage
            blank_image[TestImage.shape[0]:TestImage.shape[0]*2,:]=FlippedImage
            TestImage=blank_image.copy()

        ProcessedImage=TestImage.copy()
        #if ParameterObject.NoProcessing<0.5:#skip all modification of pixel values except for resizing and mirror
        if ParameterObject.MedianBlurDist!=0:
            ProcessedImage = cv2.medianBlur(ProcessedImage,ParameterObject.MedianBlurDist)

        if ParameterObject.AdapativeThreshold!=0:
            ProcessedImage = cv2.adaptiveThreshold(ProcessedImage, ParameterObject.AdapativeThreshold, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,ParameterObject.GausSize_Threshold,ParameterObject.SubtractMean) #cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)[1]
        
        if ParameterObject.Canny!=0:
            ProcessedImage=cv2.Canny(ProcessedImage, 100, 200)

        #blend processed and base(resized)image according to Alpha blend parameter
        beta = (1.0 - ParameterObject.AlphaBlend)
        TestImage = cv2.addWeighted(TestImage, ParameterObject.AlphaBlend, ProcessedImage, beta, 0.0)

        #TestImage = cv2.morphologyEx(TestImage, cv2.MORPH_OPEN, ParameterObject.kernel)
        #TestImage = cv2.morphologyEx(TestImage, cv2.MORPH_CLOSE, ParameterObject.kernel)
        #img = image_smoothening(img)
        #TestImage = cv2.bitwise_or(TestImage, closing)


        #run OCR
        results=""
        if SkipOcr==False:
            results = pytesseract.image_to_data(TestImage, config=ParameterObject.config,output_type=Output.DICT,lang='eng')


            #override all image preparation
            #TestImage=cv2.imread(ImagePath,cv2.IMREAD_GRAYSCALE)
            #results = pytesseract.image_to_data(TestImage,output_type=Output.DICT,lang='eng')


            # loop over each of the individual textlocalizations
            Collated_Snr_Text=""
            for i in range(0, len(results["text"])):
                # We can then extract the bounding box coordinates
                # of the text region from  the current result
                x = results["left"][i]
                y = results["top"][i]
                w = results["width"][i]
                h = results["height"][i]
                # We will also extract the OCR text itself along
                # with the confidence of the text localization
                text = results["text"][i].replace(" ","")
                conf = int(results["conf"][i])
                # filter out weak confidence text localizations
                if (conf > 50) and (len(text)>1):#:args["min_conf"]:
                    # We will display the confidence and text to
                    # our terminal
                    Collated_Snr_Text=Collated_Snr_Text+(str(text))
                    # We then strip out non-ASCII text so we can
                    # draw the text on the image We will be using
                    # OpenCV, then draw a bounding box around the
                    # text along with the text itself
                    text = "".join(text).strip()
                    cv2.rectangle(TestImage, (x, y),(x + w, y + h),(0, 0, 255), 2)
                    cv2.putText(TestImage,text,(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 255, 255), 3)

            
            
            #if found something - set fitness to above zero- as reading anything is a good start
            if len( Collated_Snr_Text)>4:
                self.Known_Snr_Fitness=round(0.1,3)
            if len( Collated_Snr_Text)>8:
                self.Known_Snr_Fitness=round(0.2,3)
            
            if self.Known_SNR==True:
                #extracted snr string from image filepath - either generated by user or automatically
                self.Known_Snr_Fitness= round(CheckStringSimilarity(self.Known_SNR_string,Collated_Snr_Text),3)
            
            if self.Known_Snr_Fitness>0.5:
                pass
                #print(self.Known_Snr_Fitness,self.Known_SNR_string,Collated_Snr_Text)
            
        return TestImage,self.Known_Snr_Fitness