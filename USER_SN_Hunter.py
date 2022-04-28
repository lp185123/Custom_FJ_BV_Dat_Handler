import os
#import ManyMuchS39
import cv2
import BV_DatReader_Lib
import _3DVisLabLib
import enum
import numpy as np
import copy
import random
import time
import datetime
import sys, os, shutil, binascii, math
import datetime
import time
import psutil
import multiprocessing
import snunterlib#need this to keep multiprocess happy - doesnt like non-imported function

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []#global object to handle recording mouse position
cropping = False
Global_Image=None#
GlobalMousePos=None#need to be global to handle capturing mouse on opencv UI
ImgView_Resize=1.3#HD image doesnt fit on screen


def kmeans_color_quantization(image, clusters=8, rounds=1):#
    #https://stackoverflow.com/questions/60197665/opencv-how-to-use-floodfill-with-rgb-image
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

def GetContours(image):
    blur = cv2.pyrMeanShiftFiltering(image, 11, 21)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
    cnt = contours[4]
    cv2.drawContours(image, contours, 0, (0,255,0), 3)
    return image

def FindRectangles(image):
    blur = cv2.pyrMeanShiftFiltering(image, 11, 21)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
   
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12),2)
    return image


def click_and_crop(event, x, y, flags, param):
    #this is specifically for the s39 area selection and will need to be modified
    #for other applications
    #https://pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/#:~:text=Anytime%20a%20mouse%20event%20happens,details%20to%20our%20click_and_crop%20function.
	# grab references to the global variables
    global refPt, cropping,GlobalMousePos,ImgView_Resize

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    
    #UI has to be shrunk to fit in window - working images are true size
    x=x*ImgView_Resize
    y=y*ImgView_Resize
    #if user is cropping - comply with s39 restriction for height & width being divisibly by 8
    if cropping==True:
        #get difference between start of area select and current position, then correct to be divisible by 8
        StartX=refPt[0][0]
        StartY=refPt[0][1]

        #stop user drawing from right to left
        if x<StartX:
            refPt = [(int(x), int(y))]
            return
        if y<StartY:
            refPt = [(int(x), int(y))]
            return

            
        DiffX=x-StartX
        DiffY=y-StartY
        ErrorX=DiffX%8
        ErrorY=DiffY%8
        x=x-ErrorX
        y=y-ErrorY
    #set global variable
    GlobalMousePos=(int(x), int(y))


    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(int(x), int(y))]
        cropping = True
	# check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((int(x), int(y)))
        cropping = False

def SetImageParams(SNunter_UserParamsSide_int,FlipSide,FlipWave,FlipHoriz):
    #this can probably be done more easily using enums
    if FlipSide==True:
        if SNunter_UserParamsSide_int.s39_side=="front":
            SNunter_UserParamsSide_int.s39_side="back"
        else:  
            SNunter_UserParamsSide_int.s39_side="front"

    if FlipWave==True:
        if SNunter_UserParamsSide_int.s39_wave=="red":
            SNunter_UserParamsSide_int.s39_wave="green"
        elif SNunter_UserParamsSide_int.s39_wave=="green":
            SNunter_UserParamsSide_int.s39_wave="blue"
        elif SNunter_UserParamsSide_int.s39_wave=="blue":
            SNunter_UserParamsSide_int.s39_wave="colour"
        elif SNunter_UserParamsSide_int.s39_wave=="colour":
            SNunter_UserParamsSide_int.s39_wave="red"
        print(SNunter_UserParamsSide_int.s39_wave)

    SNunter_UserParamsSide_int.FlipHorizontal=FlipHoriz
    
    return SNunter_UserParamsSide_int

def GetAllFilesInFolder_Recursive(root):
    ListOfFiles=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            FullpathOfFile=(os.path.join(path, name))
            ListOfFiles.append(FullpathOfFile)
    return ListOfFiles

def GetList_Of_ImagesInList(ListOfFiles, ImageTypes=(".jPg", ".Png",".gif")):
    
    Image_FileNames=[]
    
    #list comprehension [function-of-item for item in some-list
    ImageTypes_ForceLower=[x.lower()  for x in ImageTypes]
    ImageTypes_ForceLower_Tuple=tuple(ImageTypes_ForceLower)
    
    for filename in ListOfFiles:
    #if a known image filetype - copy file
        if str.endswith(str.lower(filename),ImageTypes_ForceLower_Tuple):
            Image_FileNames.append(filename)
    
    return Image_FileNames


def GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_int):
    OutputImage=None
    if SNunter_UserParams_Loaded_int.s39_side == 'front':
        if SNunter_UserParams_Loaded_int.GetFloodFillImg==False:
            OutputImage=SNunter_UserParams_Loaded_int.ColourBackGroundF.copy()
        else:
            OutputImage=SNunter_UserParams_Loaded_int.ColourBackGroundF_flood.copy()
    else:
        if SNunter_UserParams_Loaded_int.GetFloodFillImg==False:
            OutputImage=SNunter_UserParams_Loaded_int.ColourBackGroundB.copy()
        else:
            OutputImage=SNunter_UserParams_Loaded_int.ColourBackGroundB_flood.copy()
    
    if SNunter_UserParams_Loaded_int.s39_wave == 'red':
        OutputImage[:,:,0]=OutputImage[:,:,2]
        OutputImage[:,:,1]=OutputImage[:,:,2]
    if SNunter_UserParams_Loaded_int.s39_wave == 'green':
        OutputImage[:,:,0]=OutputImage[:,:,1]
        OutputImage[:,:,2]=OutputImage[:,:,1]
    if SNunter_UserParams_Loaded_int.s39_wave == 'blue':
        OutputImage[:,:,1]=OutputImage[:,:,0]
        OutputImage[:,:,2]=OutputImage[:,:,0]
    
    if SNunter_UserParams_Loaded_int.FlipHorizontal==True:
        # Use Flip code 0 to flip vertically
        OutputImage = snunterlib.RotateImage(OutputImage,180)# cv2.flip(OutputImage, 0)

    

    return OutputImage

def SN_HuntLoop(SNunter_UserParams_Loaded,InputDat):
    global ImgView_Resize#HD image doesnt fit on screen

    #maybe find least skewed note? Either auto or by user selecting #TODO

    #extract s39 data as image in RGB for user - save as base for UI
    #populate input parameters for s39 extraction
    s39Maker = snunterlib.S39Maker()
    s39Maker.files=[InputDat]
    s39Maker.outputDirectory=SNunter_UserParams_Loaded.OutputFolder +"\\"
    s39Maker.start()
    #1632*320 is full mm8 image
    OutputImageR=snunterlib.Get39Image(s39Maker,'front','red',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    OutputImageG=snunterlib.Get39Image(s39Maker,'front','green',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    OutputImageB=snunterlib.Get39Image(s39Maker,'front','blue',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    #pull out the hex mass and convert to an image
    #create dummy dictionary for cross-compatibility with other processes
    ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
    ColourImg[:,:,0]=OutputImageB
    ColourImg[:,:,1]=OutputImageG
    ColourImg[:,:,2]=OutputImageR
    SNunter_UserParams_Loaded.ColourBackGroundF=ColourImg.copy()
    ColourImg=snunterlib.FloodFill(ColourImg,SNunter_UserParams_Loaded)
    SNunter_UserParams_Loaded.ColourBackGroundF_flood=ColourImg.copy()
    OutputImageR=snunterlib.Get39Image(s39Maker,'back','red',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    OutputImageG=snunterlib.Get39Image(s39Maker,'back','green',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    OutputImageB=snunterlib.Get39Image(s39Maker,'back','blue',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    #pull out the hex mass and convert to an image
    ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
    ColourImg[:,:,0]=OutputImageB
    ColourImg[:,:,1]=OutputImageG
    ColourImg[:,:,2]=OutputImageR
    SNunter_UserParams_Loaded.ColourBackGroundB=ColourImg.copy()
    ColourImg=snunterlib.FloodFill(ColourImg,SNunter_UserParams_Loaded)
    SNunter_UserParams_Loaded.ColourBackGroundB_flood=ColourImg.copy()

    #enable global variables to be used
    global Global_Image
    global refPt
    global GlobalMousePos

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    Blur=1
    LaB=0
    
    print("Use keys 1/2/3 to cycle through viewmodes")
    print("select area in correct wave with mouse then press C to cut out Region of Interest")

    
    while True:
         #update image
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded)
        clone=Global_Image.copy()
        if len(refPt)==2:
            #smoothly reduce blurring if user has finished selecting area
            if Blur>1: Blur=Blur-1
            if LaB<SNunter_UserParams_Loaded.LocalAreaBuffer:LaB=LaB+8 
            if LaB>SNunter_UserParams_Loaded.LocalAreaBuffer:LaB=SNunter_UserParams_Loaded.LocalAreaBuffer
            #draw rectangle around area selected by user
            cv2.rectangle(Global_Image, refPt[0], refPt[1], (0, 255, 0), 2)
            #buffer rectangle (for grabbing more local features to help match template)
            #cv2.rectangle(Global_Image, (refPt[0][0]-LaB,refPt[0][1]-LaB), (refPt[1][0]+LaB,refPt[1][1]+LaB), (20, 150, 20), 1)
            #Draw Circle
            #get max dimension
            #DistanceDiagX=abs((refPt[0][0]-LaB)-(refPt[1][0]+LaB))
            MidPointX=int((refPt[0][0]+refPt[1][0])/2)
            MidPointY=int((refPt[0][1]+refPt[1][1])/2)
            Length=max(abs(refPt[0][0]-refPt[1][0]),abs(refPt[0][1]-refPt[1][1]))#assume X axis will be longes,
            cv2.circle(Global_Image, (MidPointX,MidPointY), int(Length/2), (20, 150, 20), 1)
        elif len(refPt)==1:
            LaB=0
            if Blur<35: Blur=Blur+2
            #cut out area of interest
            SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
            SNunter_UserParams_Loaded_temp.s39_wave = 'red'
            Global_Image_temp=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
            
            AoI=Global_Image_temp[refPt[0][1]:GlobalMousePos[1],refPt[0][0]:GlobalMousePos[0],:]
            #blur original image
            KernelSize=Blur
            kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing
            Global_Image = cv2.filter2D(Global_Image,-1,kernel)
            #place back AoI
            Global_Image[refPt[0][1]:GlobalMousePos[1],refPt[0][0]:GlobalMousePos[0],:]=AoI
            #draw on rectangle
            cv2.rectangle(Global_Image, refPt[0], GlobalMousePos, (100, 100, 100), 1)
        else:
            if Blur>1: Blur=Blur-2
            LaB=0
    
        # display the image and wait for a keypress
        Global_Image_view=cv2.resize(Global_Image,(int(Global_Image.shape[1]/ImgView_Resize),int(Global_Image.shape[0]/ImgView_Resize)))
        
        #Global_Image_view=FloodFill(Global_Image_view,(20,20))
        cv2.imshow("image", Global_Image_view)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            refPt=[]#clear out cropping rectangle
        # if the 'c' key is pressed, break from the loop
        if key == ord("1"):
            SNunter_UserParams_Loaded=SetImageParams(SNunter_UserParams_Loaded,True,False, SNunter_UserParams_Loaded.FlipHorizontal)
        if key == ord("2"):
            SNunter_UserParams_Loaded=SetImageParams(SNunter_UserParams_Loaded,False,True, SNunter_UserParams_Loaded.FlipHorizontal)
        if key == ord("3"):
            SNunter_UserParams_Loaded=SetImageParams(SNunter_UserParams_Loaded,False,False,not SNunter_UserParams_Loaded.FlipHorizontal)
        if key == ord("c"):
            if SNunter_UserParams_Loaded.s39_wave == 'colour':
                print("Please choose a wave using the keyboard - RGB is not compatible with .S39 extraction")
                
            else:
                if len(refPt) == 2:
                    Global_Image=clone.copy()
                    Global_Image=cv2.resize(Global_Image,(int(Global_Image.shape[1]/ImgView_Resize),int(Global_Image.shape[0]/ImgView_Resize)))
                    KernelSize=31
                    kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing
                    Global_Image_superblur = cv2.filter2D(Global_Image,-1,kernel)
                    cv2.imshow("image", Global_Image_superblur)
                    cv2.waitKey(1)#1 millisecond to refresh window
                    break
        if key==ord("4"):
            print("DEBUG - check flood function to remove background")
            SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
            SNunter_UserParams_Loaded_temp.s39_wave = 'colour'
            SNunter_UserParams_Loaded_temp.GetFloodFillImg=True
            Global_Image_Flood=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
            Global_Image_Flood=cv2.resize(Global_Image_Flood,(int(Global_Image_Flood.shape[1]/ImgView_Resize),int(Global_Image_Flood.shape[0]/ImgView_Resize)))
            cv2.imshow("image", Global_Image_Flood)
            cv2.waitKey(2000)
    # if there are two reference points, then crop the region of interest
    # from the image and display it
    if len(refPt) == 2:
        print("User parameter clipping - press any key to continue")
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded)
        roi_user = Global_Image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        SNunter_UserParams_Loaded.UserSN_ROI=(refPt)
        #cv2.imshow("imageclip", roi_user)
        #cv2.waitKey(0)
        #get pattern we will use to search other notes (full colour)
        print("Search clipping- press any key to continue")
        SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
        SNunter_UserParams_Loaded_temp.s39_wave = 'colour'
        SNunter_UserParams_Loaded_temp.GetFloodFillImg=True
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
        #Global_Image=FloodFill(Global_Image)
        roi_searchPattern = Global_Image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        #populate search template
        SNunter_UserParams_Loaded.SearchTemplate_col=roi_searchPattern#colour template for reference
        SNunter_UserParams_Loaded.SearchTemplate_bw=cv2.cvtColor(roi_searchPattern, cv2.COLOR_BGR2GRAY)#single channels template used for matching
        #cv2.imshow("imageclip", roi_searchPattern)
        #cv2.waitKey(0)
        #get a larger area of the note to help the template matcher
        #get the colour image again - double up the code incase we want to move this out
        SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
        SNunter_UserParams_Loaded_temp.s39_wave = 'colour'
        SNunter_UserParams_Loaded_temp.GetFloodFillImg=True
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
        #Global_Image=FloodFill(Global_Image)
        #use buffer parameter to increase size of selection to grab local features for template matching
        LaB=SNunter_UserParams_Loaded.LocalAreaBuffer
        SNunter_UserParams_Loaded.SearchTemplate_localArea_bw=cv2.cvtColor(Global_Image[refPt[0][1]-LaB:refPt[1][1]+LaB, refPt[0][0]-LaB:refPt[1][0]+LaB], cv2.COLOR_BGR2GRAY)
        #cv2.imshow("imageclip", SNunter_UserParams_Loaded.SearchTemplate_localArea_bw)
        #cv2.waitKey(0)

        

    else:
        raise Exception("No area selected - cannot proceed")

    #ready for checking through all MM8 data to find serial number matching
    #we must handle 4 note orientations and variations of angle
    #first - build template matching rotation series

    #create circular area cut - so have same blackspace during rotation - otherwise
    #if rectangular blackspace from rotation may affect template matching score

    #MOST OF THIS CAN BE REMOVED - ULTIMATELY DID NOT NEED MASKING SYSTEM

    MidPointX=int((refPt[0][0]+refPt[1][0])/2)
    MidPointY=int((refPt[0][1]+refPt[1][1])/2)

    SNunter_UserParams_Loaded.MidPoint_user_SN_XY=(MidPointX,MidPointY)
    Length=max(abs(refPt[0][0]-refPt[1][0]),abs(refPt[0][1]-refPt[1][1]))
    #create random noise image
    RandomNoiseImg=np.random.randint(255, size=(int(Length), int(Length),3),dtype="uint8")
    #cv2.imshow("imageclip", RandomNoiseImg)
    #cv2.waitKey(0)
    ##probably need circular mask
    Mask_circle=Global_Image[0:Length,0:Length,:]#use any donor image to avoid using Numpy library until we need it (keep size of exe down)
    Mask_circle[:,:,:]=0#set everything to 0
    #cv2.imshow("imageclip", Mask_circle)
    #cv2.waitKey(0)
    #create mask, of black image with white circle in the middle to be used as mask
    cv2.circle(Mask_circle,(int(Mask_circle.shape[0]/2),int(Mask_circle.shape[1]/2)),int(Length/2),(255, 255, 255),-1)
    SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle=Mask_circle
    #cv2.imshow("imageclip", Mask_circle)
    #cv2.waitKey(0)
    
    #create inverted version of mask
    Mask_circle_inverted= cv2.bitwise_not(Mask_circle)#set everything to 0
    SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle=Mask_circle_inverted
    #cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    #cv2.waitKey(0)

    #grab a square area of the image to be used for template matching
    #firstly blank out a rectangle in the image to mask out the SN
    SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
    SNunter_UserParams_Loaded_temp.s39_wave = 'colour'
    SNunter_UserParams_Loaded_temp.GetFloodFillImg=True
    Global_Image_blankSN=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
    #Global_Image=FloodFill(Global_Image)
    Global_Image_blankSN[:,:,:]=0#make black canvas
    Global_Image_blankSN[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]=255#create white strip where user rectangled SN
    #get coords from midpoint
    Y_up=int(MidPointY-Length/2)
    X_left=int(MidPointX-Length/2)
    SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea_ROI=(X_left,Y_up)
    SquareCutRegion_blankSNMask=Global_Image_blankSN[Y_up:Y_up+Length,X_left:X_left+Length,:]
    #cv2.imshow("imageclip", SquareCutRegion_blankSNMask)
    #cv2.waitKey(0)
    
    #get full colour image again in case we want to move code
    Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
    #Global_Image=FloodFill(Global_Image)
    SquareCutRegion=Global_Image[Y_up:Y_up+Length,X_left:X_left+Length,:]
    #subtract noise
    #SquareCutRegion=cv2.subtract(SquareCutRegion,SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    #SquareCutRegion=cv2.subtract(SquareCutRegion,SquareCutRegion_blankSNMask)
    #cv2.imshow("imageclip", SquareCutRegion)
    #cv2.waitKey(0)
    #https://pyimagesearch.com/2021/01/20/opencv-rotate-image/


    #update circular masks with SN mask
    SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle=cv2.add(SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle,SquareCutRegion_blankSNMask)
    #cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    #cv2.waitKey(0)
    #update inverted mask
    SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle=cv2.bitwise_not(SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    #cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle)
    #cv2.waitKey(0)

    #create noise around edges of square only and leave a circle to later add the rotation template
    MaskedNoise=cv2.subtract(RandomNoiseImg,SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle)
    #cv2.imshow("imageclip", MaskedNoise)
    #cv2.waitKey(0)

    #add together for composite image
    SNunter_UserParams_Loaded.CircularAoI_WithNoise=cv2.add(SquareCutRegion,MaskedNoise)
    #cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_WithNoise)
    #cv2.waitKey(0) 
  
    for RotateDeg in range (-SNunter_UserParams_Loaded.RotationRange_deg,SNunter_UserParams_Loaded.RotationRange_deg,SNunter_UserParams_Loaded.RotationStepSize):
        #RotatedImage=RotateImage(SNunter_UserParams_Loaded.SquareCutRegion,RotateDeg)
        #SubtractImg=cv2.subtract(RotatedImage,SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
        #RotatedImage=RotateImage(SquareCutRegion,RotateDeg)
        RotatedImage_mask=snunterlib.RotateImage(SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle,RotateDeg)
        #RotatedImg_andMask=cv2.add(MaskedNoise,SubtractImg)
        #SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea.append(RotatedImage)
        SNunter_UserParams_Loaded.RotateSeries_MaskOfSN.append(RotatedImage_mask)
        #cv2.imshow("imageclip", RotatedImage)
        #cv2.waitKey(0)
        #cv2.imshow("imageclip", RotatedImage_mask)
        #cv2.waitKey(0)

        #second rotation technique
        #rotate the main image and crop a square from the center of selection region
        M = cv2.getRotationMatrix2D((MidPointX,MidPointY), RotateDeg, 1.0)
        rotated = cv2.warpAffine(Global_Image, M, (Global_Image.shape[1], Global_Image.shape[0]))
        SquareCutRegion_rot=rotated[Y_up:Y_up+Length,X_left:X_left+Length,:]
        SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea.append(SquareCutRegion_rot)
        #cv2.imshow("imageclip", SquareCutRegion_rot)
        #cv2.waitKey(0)

        #save out rotation sequence for debugging
        RotateMatchPAttern_filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(RotateDeg) + "_RotateMatchPattern.jpg"
        #cv2.imwrite(RotateMatchPAttern_filename,SquareCutRegion_rot)


    return SNunter_UserParams_Loaded

def PrepareMultiProcessing(SNunter_UserParams_Loaded,InputDats_list):
    #set up multiprocessing
    #PhysicalCores=psutil.cpu_count(logical=False)#number of physical cores
    Cores_Available = int(os.environ['NUMBER_OF_PROCESSORS'])#hyperthreaded cores - not compatible with some processes
    #final core count available
    CoresTouse=1
    #user may have restricted performance to overcome memory errors or to leave system capacity for other tasks
    if SNunter_UserParams_Loaded.MemoryError_ReduceLoad[0]==True and Cores_Available>1:
        CoresTouse=min(Cores_Available,SNunter_UserParams_Loaded.MemoryError_ReduceLoad[1])#if user has over-specified cores restrict to cores available
        print("THROTTLING BY USER - Memory protection: restricting cores to", CoresTouse, "or less, user option MemoryError_ReduceLoad")
    else:
        CoresTouse=Cores_Available
    #if no restriction by user , leave a core anyway
    processes=max(CoresTouse-1,1)#rule is thumb is to use number of logical cores minus 1, but always make sure this number >0. Its not a good idea to blast CPU at 100% as this can reduce performance as OS tries to balance the load
    #find how much memory single process uses (windows)
    Currentprocess = psutil.Process(os.getpid())
    SingleProcess_Memory=Currentprocess.memory_percent()
    SystemMemoryUsed=psutil.virtual_memory().percent
    FreeMemoryBuffer_pc=SNunter_UserParams_Loaded.FreeMemoryBuffer_pc#arbitrary free memory to leave
    MaxPossibleProcesses=max(math.floor((100-FreeMemoryBuffer_pc-SystemMemoryUsed)/SingleProcess_Memory),0)#can't be lower than zero
    print(MaxPossibleProcesses,"parallel processes possible at system capacity (leaving",FreeMemoryBuffer_pc,"% memory free)")
    print("Each process will use ~ ",round(psutil.virtual_memory().total*SingleProcess_Memory/100000000000,1),"gb")#convert bytes to gb
    #cannot proceed if we can't use even one core
    if processes<1 or MaxPossibleProcesses<1:
        print(("Multiprocess Configuration error! Less than 1 process possible - memory or logic error"))
        processes=1
        MaxPossibleProcesses=1
        print("Forcing processors =1, may cause memory error")
        #raise Exception("Multiprocess Configuration error! Less than 1 process possible - memory or logic error")
    #check system has enough memory - if not restrict cores used
    if processes>MaxPossibleProcesses:
        print("WARNING!! possible memory overflow - restricting number of processes from",processes,"to",MaxPossibleProcesses)
        processes=MaxPossibleProcesses

    ThreadsNeeded=len(InputDats_list)
    #calculate how many processes each CPU will get per cycle
    chunksize=1#arbitrary setting to be able to get time feedback and not clog up system - but each reset of tasks can have memory overhead
    #how many jobs do we build up to pass off to the multiprocess pool, in this case in theory each core gets 3 stacked tasks
    ProcessesPerCycle=processes*chunksize#this might be bigger than amount of multiprocesses needed
    #randomise tasks - application specific and not required in this situation
    #SacrificialDictionary=copy.deepcopy(MatchImages.ImagesInMem_Pairing)
    #ImagesInMem_Pairing_ForThreading=dict()
    #while len(SacrificialDictionary)>0:
    #    RandomItem=random.choice(list(SacrificialDictionary.keys()))
    #    ImagesInMem_Pairing_ForThreading[RandomItem]=MatchImages.ImagesInMem_Pairing[RandomItem]
    #    del SacrificialDictionary[RandomItem]

    print("[Multiprocess info]","Taskstack per core:",chunksize,"  Taskpool size:",ProcessesPerCycle,"  Physical cores used:",processes,"   Image Threads:",ThreadsNeeded)

    return processes,chunksize,ProcessesPerCycle


def SearchMM8_forSN_Location(SNunter_UserParams_Loaded,InputDats_list):
    #1 6 3 2 * 6 4 0 full res
    #roll through all dats in input list
     #start timer and time metrics
    listTimings=[]
    listCounts=[]
    listAvgTime=[]
    ProcessOnly_start = time.perf_counter()
    t1_start = None
    #get multiprocess configuration
    processes,chunksize,ProcessesPerCycle=PrepareMultiProcessing(SNunter_UserParams_Loaded,InputDats_list)
    pool = multiprocessing.Pool(processes=processes)
    listJobs=[]
    CompletedProcesses=0

    #start multiprocess loop
    if processes>1:
        print("Multi process started")
        #populate list of jobs
        t1_start = time.perf_counter()
        for DatIndex,DatFile in enumerate(InputDats_list):
            listJobs.append((DatIndex,DatFile,SNunter_UserParams_Loaded))#parameters for multiprocess function
            #fire off list of jobs under certain conditions or if we have run out of jobs
            if (DatIndex%ProcessesPerCycle==0 and DatIndex!=0) or DatIndex==len(InputDats_list)-1:
                #fork off function loaded with list of input parameters and set CPU stack of jobs
                ReturnList=(pool.imap_unordered(snunterlib.ProcessMM8_forSN,listJobs,chunksize=chunksize))
                #seems like you have to iterate over the return to stop the process finishing before threads come back?
                for Item in ReturnList:
                    pass
                #will wait for return of processes
                CompletedProcesses=CompletedProcesses+len(listJobs)
                TimePerJob=round((time.perf_counter()-t1_start)/CompletedProcesses,1)
                print("Jobs done:", CompletedProcesses, "/", len(InputDats_list))
                print("Time per job:",str(TimePerJob),"seconds")
                print("Time left:",len(InputDats_list)-CompletedProcesses,"jobs:",str(datetime.timedelta(seconds=(TimePerJob*(len(InputDats_list)-CompletedProcesses)))))

                #clear list of jobs
                listJobs=[]

    #single process
    if processes==1:
        print("Single process started")
        for DatIndex,DatFile in enumerate(InputDats_list):
            try:
                if DatIndex>0:
                    listTimings.append(round(time.perf_counter()-t1_start,2))
                    #don't need linear regression as process is static - keep in case we add something funky
                    #slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(listTimings[1:-1])),listTimings[1:-1])
                    TimePerProcess=sum(listTimings)/len(listTimings)
                    JobsLeft=len(InputDats_list)-DatIndex
                    print("Estimated time left for" + str(len(InputDats_list)-DatIndex)+"jobs:",str(datetime.timedelta(seconds=(TimePerProcess*JobsLeft))))
                    print("Total time for",len(InputDats_list),"jobs:",str(datetime.timedelta(seconds=(TimePerProcess*len(InputDats_list)))))
                    print("Time per Snunt:",str(datetime.timedelta(seconds=(TimePerProcess))))
                #start timer again
                t1_start = time.perf_counter()
            except:
                print("Timing code broken")
            

            snunterlib.ProcessMM8_forSN((DatIndex,DatFile,SNunter_UserParams_Loaded))

def main():
    #instantiate user params class which we will load during user interactivity
    SNunter_UserParams_toLoad=snunterlib.SNunter_UserParams()

    #load in user folders   
    #SNunter_UserParams_toLoad.InputFolder=r"E:\NCR\Currencies\Bangladesh_SR2800\Bangladesh\SR DC\MM8\Sorted_Denoms\Denom_3"
    SNunter_UserParams_toLoad.InputFolder=r"E:\NCR\Currencies\Bangladesh_SR2800\Bangladesh\SR DC\MM8\Sorted_Denoms\Denom_3"
    #SNunter_UserParams_toLoad.InputFolder=r"E:\NCR\Currencies\Bangladesh_SR2800\Bangladesh\SR DC\MM8\1000\2008"
    #set output folder
    SNunter_UserParams_toLoad.OutputFolder=r"C:\Working\FindIMage_In_Dat\OutputFindPattern"
    print("Please check output folders can be deleted:",SNunter_UserParams_toLoad.OutputFolder)
    Response=_3DVisLabLib.yesno("Continue?")
    if Response==True:
        #delete output folder
        _3DVisLabLib.DeleteFiles_RecreateFolder(SNunter_UserParams_toLoad.OutputFolder)


    #get all files in folders recursively
    print("Looking in",SNunter_UserParams_toLoad.InputFolder,"for .dat files")
    List_all_Files=GetAllFilesInFolder_Recursive(SNunter_UserParams_toLoad.InputFolder)
    #filter out non .dats
    List_all_Dats=GetList_Of_ImagesInList(List_all_Files,".dat")
    print(len(List_all_Dats),".dat files found")
    #randomise dat files
    #load images in random order for testing
    print("Randomising input order, set of",int(min(SNunter_UserParams_toLoad.SubSetOfData,len(List_all_Dats))),"MM8 files")
    randomdict=dict()
    for Index, ImagePath in enumerate(List_all_Dats):
        if ImagePath.split(".")[-1].lower()=="dat":
            #shouldnt have to do this twice
            randomdict[ImagePath]=Index
    #user may have specified a subset of data
    List_all_Dats=[]
    while (len(List_all_Dats)<SNunter_UserParams_toLoad.SubSetOfData) and (len(randomdict)>0):
        randomchoice_img=random.choice(list(randomdict.keys()))
        List_all_Dats.append(randomchoice_img)
        del randomdict[randomchoice_img]

    #PrepareMultiProcessing(SNunter_UserParams_toLoad,List_all_Dats)

    #UI to select part of note, then create details necessary to find SN in other notes 
    SNunter_UserParams_Loaded=SN_HuntLoop(SNunter_UserParams_toLoad,List_all_Dats[0])#use first .dat file - even better if we can find least skewed one

    #for each mm8 file try to find SN area - note can be in any orientation and true orientation must
    #be recorded in the S39 file 
    SearchMM8_forSN_Location(SNunter_UserParams_Loaded,List_all_Dats)


if __name__ == "__main__":
    #entry point
    #try:
    
    main()
    #except Exception as e:
    #    ##note: cleaning up after exception should be to set os.chdir(anything else) or it will lock the folder
    #    print(e)
    #    # printing stack trace
    #    print("Press any key to continue")
    #    os.system('pause')