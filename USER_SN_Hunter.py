import os
import ManyMuchS39
import cv2
import BV_DatReader_Lib
import _3DVisLabLib
import enum
import numpy as np
import copy
import random
import time
import datetime
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []#global object to handle recording mouse position
cropping = False
Global_Image=None#
GlobalMousePos=None#need to be global to handle capturing mouse on opencv UI
ImgView_Resize=5#HD image doesnt fit on screen

def RotateImage(InputImage,RotationDeg):
    #set point of rotation to centre of image - can offset if we need to
    #less verbose method would be to use imutils library
    M = cv2.getRotationMatrix2D((int((InputImage.shape[1])/2), int((InputImage.shape[0])/2)), RotationDeg, 1.0)
    rotated = cv2.warpAffine(InputImage, M, (InputImage.shape[1], InputImage.shape[0]))
    return rotated

def click_and_crop(event, x, y, flags, param):
    
    #https://pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/#:~:text=Anytime%20a%20mouse%20event%20happens,details%20to%20our%20click_and_crop%20function.
	# grab references to the global variables
    global refPt, cropping,GlobalMousePos,ImgView_Resize

    
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    GlobalMousePos=(x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x*ImgView_Resize, y*ImgView_Resize)]
        cropping = True
	# check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x*ImgView_Resize, y*ImgView_Resize))
        cropping = False

class SNunter_UserParams():
    def __init__(self) -> None:
        self.InputFolder=r"C:\Working\FindIMage_In_Dat\Input"
        self.OutputFolder=r"C:\Working\FindIMage_In_Dat\Output"
        self.s39_shouldPrintProgress = False
        self.s39_directory = '.\\'#repopulated later
        self.s39_outputDirectory = '.\\s39\\'#repopulated later
        self.s39_wave = 'colour' # or green or blue
        self.s39_side = 'front' # or back
        self.s39_validation = '80080103'
        self.s39_width = 336
        self.s39_height = 88
        self.s39_x = 681+358#519+336+40+40+40#add these together
        self.s39_y = 101#keep scroll point at 320 (weird coordinate systems)
        self.ColourBackGroundF=None
        self.ColourBackGroundB=None
        self.WorkingImage=None
        self.FlipHorizontal=False
        self.SearchTemplate_col=None
        self.SearchTemplate_bw=None
        self.SearchTemplate_localArea_bw=None
        self.CircularAoI_WithNoise=None
        self.CircularAoI_Mask_WhiteCircle=None
        self.CircularAoI_Mask_BlackCircle=None
        self.RotateSeries_SquareLocalArea=[]
        self.RotateSeries_MaskOfSN=[]
        self.LocalAreaBuffer=50#add to area once user has selected serial number
        self.RotationRange_deg=45#rotation range - bear in mind first note may be badly skewed already
        self.RotationStepSize=2#how many steps to cover range of rotation
        self.MM8_fullResX=1632#1632
        self.MM8_fullResY=640#640
        self.PatternSearchDivideImg=3#resize image - might help search as template matching seems to die if we make image too big
        self.PatternSearchBlur=5#a little blur can help matching
        self.SubSetOfData=9999

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

def Get39Image(S39MakerObject,Side,wave,Width,Height,Xoffset,Yoffset,Validation):
    S39MakerObject.images=[]#clean out images list
    S39MakerObject.wave =wave
    S39MakerObject.side=Side
    S39MakerObject.validation=Validation
    S39MakerObject.x=Xoffset
    S39MakerObject.y=Yoffset
    S39MakerObject.width=Width
    S39MakerObject.height=Height
    S39MakerObject.extractS39()
    #need a dummy class for cross compatibility with other libraries
    FakedClass=BV_DatReader_Lib. DummyImageClass()
    FakedClass.offsetStart=0
    FakedClass.offsetEnd=len(S39MakerObject.images[0])
    FakedClass.width=S39MakerObject.width
    FakedClass.height=S39MakerObject.height
    filteredImages=dict()
    filteredImages["DummyNote"]=FakedClass
    #interpret hex mass as image 
    (OutputImage,dummy)=BV_DatReader_Lib.Image_from_Automatic_mode(filteredImages,"DummyNote",S39MakerObject.images[0],False)
    OutputImage=cv2.resize(OutputImage,(int(OutputImage.shape[1]),int(OutputImage.shape[0]*2)))
    return OutputImage

def GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_int):
    OutputImage=None
    if SNunter_UserParams_Loaded_int.s39_side == 'front':
        OutputImage=SNunter_UserParams_Loaded_int.ColourBackGroundF.copy()
    else:
        OutputImage=SNunter_UserParams_Loaded_int.ColourBackGroundB.copy()
    
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
        OutputImage = RotateImage(OutputImage,180)# cv2.flip(OutputImage, 0)

    

    return OutputImage

def SN_HuntLoop(SNunter_UserParams_Loaded,InputDat):
    global ImgView_Resize#HD image doesnt fit on screen

    #maybe find least skewed note? Either auto or by user selecting #TODO

    #extract s39 data as image in RGB for user - save as base for UI
    #populate input parameters for s39 extraction
    s39Maker = ManyMuchS39.S39Maker()
    s39Maker.files=[InputDat]
    s39Maker.outputDirectory=SNunter_UserParams_Loaded.OutputFolder +"\\"
    s39Maker.start()
    #1632*320 is full mm8 image
    OutputImageR=Get39Image(s39Maker,'front','red',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
    OutputImageG=Get39Image(s39Maker,'front','green',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
    OutputImageB=Get39Image(s39Maker,'front','blue',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
    #pull out the hex mass and convert to an image
    #create dummy dictionary for cross-compatibility with other processes
    ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
    ColourImg[:,:,0]=OutputImageB
    ColourImg[:,:,1]=OutputImageG
    ColourImg[:,:,2]=OutputImageR
    SNunter_UserParams_Loaded.ColourBackGroundF=ColourImg
    OutputImageR=Get39Image(s39Maker,'back','red',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
    OutputImageG=Get39Image(s39Maker,'back','green',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
    OutputImageB=Get39Image(s39Maker,'back','blue',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
    #pull out the hex mass and convert to an image
    ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
    ColourImg[:,:,0]=OutputImageB
    ColourImg[:,:,1]=OutputImageG
    ColourImg[:,:,2]=OutputImageR
    SNunter_UserParams_Loaded.ColourBackGroundB=ColourImg

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
            #cv2.rectangle(Global_Image, refPt[0], GlobalMousePos, (255, 0, 0), 2)
        else:
            if Blur>1: Blur=Blur-2
            LaB=0
    
        # display the image and wait for a keypress
        Global_Image_view=cv2.resize(Global_Image,(int(Global_Image.shape[1]/ImgView_Resize),int(Global_Image.shape[0]/ImgView_Resize)))
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
        elif key == ord("c"):
            if SNunter_UserParams_Loaded.s39_wave == 'colour':
                print("Please choose a wave using the keyboard - colour is not compatible with S39 extraction")
            else:
                if len(refPt) == 2:
                    Global_Image=clone.copy()
                    break
       
    # if there are two reference points, then crop the region of interest
    # from the image and display it
    if len(refPt) == 2:
        print("User parameter clipping - press any key to continue")
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded)
        roi_user = Global_Image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("imageclip", roi_user)
        #cv2.waitKey(0)
        #get pattern we will use to search other notes (full colour)
        print("Search clipping- press any key to continue")
        SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
        SNunter_UserParams_Loaded_temp.s39_wave = 'colour'
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
        roi_searchPattern = Global_Image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        #populate search template
        SNunter_UserParams_Loaded.SearchTemplate_col=roi_searchPattern#colour template for reference
        SNunter_UserParams_Loaded.SearchTemplate_bw=cv2.cvtColor(roi_searchPattern, cv2.COLOR_BGR2GRAY)#single channels template used for matching
        cv2.imshow("imageclip", roi_searchPattern)
        #cv2.waitKey(0)
        #get a larger area of the note to help the template matcher
        #get the colour image again - double up the code incase we want to move this out
        SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
        SNunter_UserParams_Loaded_temp.s39_wave = 'colour'
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
        #use buffer parameter to increase size of selection to grab local features for template matching
        LaB=SNunter_UserParams_Loaded.LocalAreaBuffer
        SNunter_UserParams_Loaded.SearchTemplate_localArea_bw=cv2.cvtColor(Global_Image[refPt[0][1]-LaB:refPt[1][1]+LaB, refPt[0][0]-LaB:refPt[1][0]+LaB], cv2.COLOR_BGR2GRAY)
        cv2.imshow("imageclip", SNunter_UserParams_Loaded.SearchTemplate_localArea_bw)
        #cv2.waitKey(0)

        

    else:
        raise Exception("No area selected - cannot proceed")

    #ready for checking through all MM8 data to find serial number matching
    #we must handle 4 note orientations and variations of angle
    #first - build template matching rotation series

    #create circular area cut - so have same blackspace during rotation - otherwise
    #if rectangular blackspace from rotation may affect template matching score

    MidPointX=int((refPt[0][0]+refPt[1][0])/2)
    MidPointY=int((refPt[0][1]+refPt[1][1])/2)
    Length=max(abs(refPt[0][0]-refPt[1][0]),abs(refPt[0][1]-refPt[1][1]))
    #create random noise image
    RandomNoiseImg=np.random.randint(255, size=(int(Length), int(Length),3),dtype="uint8")
    cv2.imshow("imageclip", RandomNoiseImg)
    #cv2.waitKey(0)
    ##probably need circular mask
    Mask_circle=Global_Image[0:Length,0:Length,:]#use any donor image to avoid using Numpy library until we need it (keep size of exe down)
    Mask_circle[:,:,:]=0#set everything to 0
    cv2.imshow("imageclip", Mask_circle)
    #cv2.waitKey(0)
    #create mask, of black image with white circle in the middle to be used as mask
    cv2.circle(Mask_circle,(int(Mask_circle.shape[0]/2),int(Mask_circle.shape[1]/2)),int(Length/2),(255, 255, 255),-1)
    SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle=Mask_circle
    cv2.imshow("imageclip", Mask_circle)
    #cv2.waitKey(0)
    
    #create inverted version of mask
    Mask_circle_inverted= cv2.bitwise_not(Mask_circle)#set everything to 0
    SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle=Mask_circle_inverted
    cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    #cv2.waitKey(0)

    


    #grab a square area of the image to be used for template matching
    #firstly blank out a rectangle in the image to mask out the SN
    SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
    SNunter_UserParams_Loaded_temp.s39_wave = 'colour'
    Global_Image_blankSN=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
    Global_Image_blankSN[:,:,:]=0#make black canvas
    Global_Image_blankSN[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]=255#create white strip where user rectangled SN
    #get coords from midpoint
    Y_up=int(MidPointY-Length/2)
    X_left=int(MidPointX-Length/2)
    SquareCutRegion_blankSNMask=Global_Image_blankSN[Y_up:Y_up+Length,X_left:X_left+Length,:]
    cv2.imshow("imageclip", SquareCutRegion_blankSNMask)
    #cv2.waitKey(0)
    
    #get full colour image again in case we want to move code
    Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
    SquareCutRegion=Global_Image[Y_up:Y_up+Length,X_left:X_left+Length,:]
    #subtract noise
    #SquareCutRegion=cv2.subtract(SquareCutRegion,SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    #SquareCutRegion=cv2.subtract(SquareCutRegion,SquareCutRegion_blankSNMask)
    cv2.imshow("imageclip", SquareCutRegion)
    #cv2.waitKey(0)
    #https://pyimagesearch.com/2021/01/20/opencv-rotate-image/


    #update circular masks with SN mask
    SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle=cv2.add(SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle,SquareCutRegion_blankSNMask)
    cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    #cv2.waitKey(0)
    #update inverted mask
    SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle=cv2.bitwise_not(SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle)
    #cv2.waitKey(0)

    #create noise around edges of square only and leave a circle to later add the rotation template
    MaskedNoise=cv2.subtract(RandomNoiseImg,SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle)
    cv2.imshow("imageclip", MaskedNoise)
    #cv2.waitKey(0)

    #add together for composite image
    SNunter_UserParams_Loaded.CircularAoI_WithNoise=cv2.add(SquareCutRegion,MaskedNoise)
    cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_WithNoise)
    #cv2.waitKey(0) 
  
    for RotateDeg in range (-SNunter_UserParams_Loaded.RotationRange_deg,SNunter_UserParams_Loaded.RotationRange_deg,SNunter_UserParams_Loaded.RotationStepSize):
        #RotatedImage=RotateImage(SNunter_UserParams_Loaded.SquareCutRegion,RotateDeg)
        #SubtractImg=cv2.subtract(RotatedImage,SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
        RotatedImage=RotateImage(SquareCutRegion,RotateDeg)
        RotatedImage_mask=RotateImage(SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle,RotateDeg)
        #RotatedImg_andMask=cv2.add(MaskedNoise,SubtractImg)
        SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea.append(RotatedImage)
        SNunter_UserParams_Loaded.RotateSeries_MaskOfSN.append(RotatedImage_mask)
        cv2.imshow("imageclip", RotatedImage)
        #cv2.waitKey(0)
        cv2.imshow("imageclip", RotatedImage_mask)
        #cv2.waitKey(0)

    return SNunter_UserParams_Loaded

def SearchMM8_forSN_Location(SNunter_UserParams_Loaded,InputDats_list):

    #1 6 3 2 * 6 4 0 full res
    #roll through all dats in input list
     #start timer and time metrics
    listTimings=[]
    listCounts=[]
    listAvgTime=[]
    ProcessOnly_start = time.perf_counter()
    t1_start = None
    for DatIndex,DatFile in enumerate(InputDats_list):
        try:
            if DatIndex>0:
                listTimings.append(round(time.perf_counter()-t1_start,2))
                #don't need linear regression as process is static - keep in case we add something funky
                #slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(listTimings[1:-1])),listTimings[1:-1])
                TimePerProcess=sum(listTimings)/len(listTimings)
                JobsLeft=len(InputDats_list)-DatIndex
                print("Estimated time left=",str(datetime.timedelta(seconds=(TimePerProcess*JobsLeft))))
                print("Total time for",len(InputDats_list),"jobs:",str(datetime.timedelta(seconds=(TimePerProcess*len(InputDats_list)))))
                print("Time per Snunt=",str(datetime.timedelta(seconds=(TimePerProcess))))
            #start timer again
            t1_start = time.perf_counter()
        except:
            print("Timing code broken")
        #pull out entire mm8 image in colour
        LoadedImg_dict={}
        s39Maker = ManyMuchS39.S39Maker()
        s39Maker.files=[]
        s39Maker.files=[DatFile]
        s39Maker.outputDirectory=SNunter_UserParams_Loaded.OutputFolder +"\\"
        s39Maker.start()
        #1632*320 is full mm8 image
        #1632*320 is full mm8 image
        OutputImageR=Get39Image(s39Maker,'front','red',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
        OutputImageG=Get39Image(s39Maker,'front','green',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
        OutputImageB=Get39Image(s39Maker,'front','blue',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
        #persist waves into dictionary
        LoadedImg_dict[('front','red')]=cv2.cvtColor(OutputImageR.copy(),cv2.COLOR_GRAY2RGB)
        LoadedImg_dict[('front','green')]=cv2.cvtColor(OutputImageG.copy(),cv2.COLOR_GRAY2RGB)
        LoadedImg_dict[('front','blue')]=cv2.cvtColor(OutputImageB.copy(),cv2.COLOR_GRAY2RGB)
        

        #pull out the hex mass and convert to an image
        #create dummy dictionary for cross-compatibility with other processes
        ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
        ColourImg[:,:,0]=OutputImageB
        ColourImg[:,:,1]=OutputImageG
        ColourImg[:,:,2]=OutputImageR
        ColourBackGroundF=ColourImg#combine channels into colour image
        OutputImageR=Get39Image(s39Maker,'back','red',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
        OutputImageG=Get39Image(s39Maker,'back','green',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
        OutputImageB=Get39Image(s39Maker,'back','blue',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103")
        #persist waves into dictionary
        LoadedImg_dict[('back','red')]=cv2.cvtColor(OutputImageR.copy(),cv2.COLOR_GRAY2RGB)
        LoadedImg_dict[('back','green')]=cv2.cvtColor(OutputImageG.copy(),cv2.COLOR_GRAY2RGB)
        LoadedImg_dict[('back','blue')]=cv2.cvtColor(OutputImageB.copy(),cv2.COLOR_GRAY2RGB)
        #pull out the hex mass and convert to an image
        ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
        ColourImg[:,:,0]=OutputImageB
        ColourImg[:,:,1]=OutputImageG
        ColourImg[:,:,2]=OutputImageR
        ColourBackGroundB=ColourImg#combine channels into colour image

        #stack all images - we need to do this due to some template matching methods compatible with masking
        #give us results more difficult to decipher if analysing each note orientation independantly
        #multiply Y by 4 (# of orientations)
        StackOrientations=np.zeros((ColourBackGroundB.shape[0],ColourBackGroundB.shape[1]*4,ColourBackGroundB.shape[2]),dtype='uint8')
        StackOrientations_UserWave=np.zeros((ColourBackGroundB.shape[0],ColourBackGroundB.shape[1]*4,ColourBackGroundB.shape[2]),dtype='uint8')
        
        
        Offset=0
        
        #add front (A)
        StackOrientations[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=ColourBackGroundF
        StackOrientations_UserWave[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=LoadedImg_dict[('front',SNunter_UserParams_Loaded.s39_wave)]

        #add front rotate(B)
        Offset=Offset+ColourBackGroundF.shape[1]
        StackOrientations[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=RotateImage(ColourBackGroundF,180)
        StackOrientations_UserWave[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=RotateImage(LoadedImg_dict[('front',SNunter_UserParams_Loaded.s39_wave)],180)
        
         #add back rotate(C)
        Offset=Offset+ColourBackGroundF.shape[1]
        StackOrientations[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=RotateImage(ColourBackGroundB,180)
        StackOrientations_UserWave[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=RotateImage(LoadedImg_dict[('back',SNunter_UserParams_Loaded.s39_wave)],180)
       
        #add back (D)
        Offset=Offset+ColourBackGroundF.shape[1]
        StackOrientations[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=ColourBackGroundB
        StackOrientations_UserWave[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=LoadedImg_dict[('back',SNunter_UserParams_Loaded.s39_wave)]

        #StackOrientations=ColourBackGroundF
        StackOrientations=cv2.resize(StackOrientations,(int(StackOrientations.shape[1]/SNunter_UserParams_Loaded.PatternSearchDivideImg),int(StackOrientations.shape[0]/SNunter_UserParams_Loaded.PatternSearchDivideImg)))
        #blurring might help template matching
        KernelSize=9
        kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing - maybe make smoother as such small images
        StackOrientations = cv2.filter2D(StackOrientations,-1,kernel)
        #cv2.imshow("imageclip", ColourBackGroundB)
        #StackOrientations=cv2.resize(ColourBackGroundF,(int(ColourBackGroundF.shape[0]/4),int(ColourBackGroundF.shape[1]/4)))
        cv2.imshow("imageclip",cv2.resize(StackOrientations,(700,700)))
        #cv2.waitKey(0)

        # Apply template Matching
        #roll through our list of the template and mask through a rotation range to handle skew of notes
        #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
        Latch_MaxValue=None
        Latch_MaxValueIndex=None
        Latch_MaxValueTopLeft=None
        Latch_MaxValueTopRight=None
        Latch_SavedPatternMatch=None
        Latch_SavedBestPattern=None
        Orientation="EMPTY"
        for RotateIndex, RotationStage in enumerate(SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea):
            ImgToProcess=StackOrientations.copy()
            RotatedTemplate=SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea[RotateIndex]
            RotatedMask=SNunter_UserParams_Loaded.RotateSeries_MaskOfSN[RotateIndex]
            RotatedTemplate=cv2.resize(RotatedTemplate,(int(RotatedTemplate.shape[1]/SNunter_UserParams_Loaded.PatternSearchDivideImg),int(RotatedTemplate.shape[0]/SNunter_UserParams_Loaded.PatternSearchDivideImg)))
            RotatedMask=cv2.resize(RotatedMask,(int(RotatedMask.shape[1]/SNunter_UserParams_Loaded.PatternSearchDivideImg),int(RotatedMask.shape[0]/SNunter_UserParams_Loaded.PatternSearchDivideImg)))
            cv2.imshow("imageclipTemplate", RotatedTemplate)
            #cv2.waitKey(0)
            #cv2.imshow("imageclip", RotatedMask)
            #cv2.waitKey(0)
            ch, h,w = RotatedTemplate.shape[::-1]
            #only two methods accept masks
            #(cv.TM_SQDIFF == match_method or match_method == cv.TM_CCORR_NORMED)
            #normalising methods make it harder to find a true match - for instance if looking at wrong side of note 
            res = cv2.matchTemplate(StackOrientations,RotatedTemplate,cv2.TM_CCORR_NORMED,None,None)#warning: only two methods work with mask
            
            #cv2.normalize( res, res, 0, 1, cv2.NORM_MINMAX, -1 )#so we can visualise result - don't normalise the result
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            InputVal=max_val#max or min result depends on method used in match template
            #print("max value",InputVal)
            #latch max/min values
            if RotateIndex==0 or InputVal>Latch_MaxValue:
                Latch_MaxValue=InputVal
                Latch_MaxValueIndex=RotateIndex

                #draw initial rectangle
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(ImgToProcess,top_left, bottom_right, 90, 1)
                Latch_MaxValueTopLeft=top_left
                Latch_MaxValueTopRight=bottom_right
                #latch image to save out
                Latch_SavedPatternMatch=ImgToProcess
                #Latch_SavedBestPattern=RotatedTemplate
                Div=SNunter_UserParams_Loaded.PatternSearchDivideImg
                Latch_SavedBestPattern=StackOrientations_UserWave[top_left[1]*Div:bottom_right[1]*Div,top_left[0]*Div:bottom_right[0]*Div,:]

                #get what orientation search pattern was found - current method relies on orientations A B C D being stacked horizontally
                StackWidth=StackOrientations_UserWave.shape[1]/4
                #get position - need to multiply by user parameter for shrinking search stack image
                PatternPos=Latch_MaxValueTopLeft[0]*SNunter_UserParams_Loaded.PatternSearchDivideImg
                #get index of orientation (1=a,2=b etc)
                OrientationIndex=np.floor(PatternPos/StackWidth)+1
                Orientation="ERROR"
                if OrientationIndex==1:Orientation="A"
                if OrientationIndex==2:Orientation="B"
                if OrientationIndex==3:Orientation="C"
                if OrientationIndex==4:Orientation="D"

            
            cv2.rectangle(ImgToProcess,Latch_MaxValueTopLeft, Latch_MaxValueTopRight, 255, 2)
            
            #cv2.imshow("imageclip", res)
            #cv2.waitKey(0)
            cv2.imshow("imageclip",cv2.resize(ImgToProcess,(800,800)))
            #cv2.waitKey(0)

        
        #save search info out to folder
        BestMatchImg_filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(DatIndex) + "_" + Orientation + "_BestMatch.jpg"
        SearchPattern_filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(DatIndex) + "_" + Orientation + "_SearchPattern.jpg"
        UserWave_filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(DatIndex) + "_" + Orientation + "_UserWave.jpg"
        
        cv2.imwrite(BestMatchImg_filename,Latch_SavedPatternMatch)
        cv2.imwrite(SearchPattern_filename,Latch_SavedBestPattern)
        #cv2.imwrite(UserWave_filename,StackOrientations_UserWave)


#instantiate user params class which we will load during user interactivity
SNunter_UserParams_toLoad=SNunter_UserParams()

#load in user folders
SNunter_UserParams_toLoad.InputFolder=r"C:\Working\FindIMage_In_Dat\TestMedia\PatternMatch_mm8_denom2"
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
print("Randomising input order, using max subset of",SNunter_UserParams_toLoad.SubSetOfData,"MM8 files")
randomdict=dict()
for Index, ImagePath in enumerate(List_all_Dats):
    randomdict[ImagePath]=Index
#user may have specified a subset of data
List_all_Dats=[]
while (len(List_all_Dats)<SNunter_UserParams_toLoad.SubSetOfData) and (len(randomdict)>0):
    randomchoice_img=random.choice(list(randomdict.keys()))
    List_all_Dats.append(randomchoice_img)
    del randomdict[randomchoice_img]

#UI to select part of note, then create details necessary to find SN in other notes 
SNunter_UserParams_Loaded=SN_HuntLoop(SNunter_UserParams_toLoad,List_all_Dats[0])#use first .dat file - even better if we can find least skewed one

#for each mm8 file try to find SN area - note can be in any orientation and true orientation must
#be recorded in the S39 file 
SearchMM8_forSN_Location(SNunter_UserParams_Loaded,List_all_Dats)