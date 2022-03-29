import os
import ManyMuchS39
import cv2
import BV_DatReader_Lib
import _3DVisLabLib
import enum
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
Global_Image=None
def click_and_crop(event, x, y, flags, param):
    #https://pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/#:~:text=Anytime%20a%20mouse%20event%20happens,details%20to%20our%20click_and_crop%20function.
	# grab references to the global variables
	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
		cv2.rectangle(Global_Image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", Global_Image)


class SNunter_UserParams():
    def __init__(self) -> None:
        self.InputFolder=r"C:\Working\FindIMage_In_Dat\Input"
        self.OutputFolder=r"C:\Working\FindIMage_In_Dat\Output"
        self.s39_shouldPrintProgress = False
        self.s39_directory = '.\\'#repopulated later
        self.s39_outputDirectory = '.\\s39\\'#repopulated later
        self.s39_wave = 'red' # or green or blue
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
        
    return OutputImage

def SN_HuntLoop(SNunter_UserParams_Loaded):
    #get all files in folders recursively
    print("Looking in",SNunter_UserParams_Loaded.InputFolder,"for .dat files")
    List_all_Files=GetAllFilesInFolder_Recursive(SNunter_UserParams_Loaded.InputFolder)
    #filter out non .dats
    List_all_Dats=GetList_Of_ImagesInList(List_all_Files,".dat")
    print(len(List_all_Dats),".dat files found")

    #maybe find least skewed note? Either auto or by user selecting #TODO

    #extract s39 data as image in RGB for user - save as base for UI
    #populate input parameters for s39 extraction
    s39Maker = ManyMuchS39.S39Maker()
    s39Maker.files=[List_all_Dats[0]]
    s39Maker.outputDirectory=SNunter_UserParams_Loaded.OutputFolder +"\\"
    s39Maker.start()
    #1632*640 is full image
    OutputImageR=Get39Image(s39Maker,'front','red',1632,640,0,0,"80080103")
    OutputImageG=Get39Image(s39Maker,'front','green',1632,640,0,0,"80080103")
    OutputImageB=Get39Image(s39Maker,'front','blue',1632,640,0,0,"80080103")
    #pull out the hex mass and convert to an image
    #create dummy dictionary for cross-compatibility with other processes
    ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
    ColourImg[:,:,0]=OutputImageB
    ColourImg[:,:,1]=OutputImageG
    ColourImg[:,:,2]=OutputImageR
    SNunter_UserParams_Loaded.ColourBackGroundF=ColourImg
    OutputImageR=Get39Image(s39Maker,'back','red',1632,640,0,0,"80080103")
    OutputImageG=Get39Image(s39Maker,'back','green',1632,640,0,0,"80080103")
    OutputImageB=Get39Image(s39Maker,'back','blue',1632,640,0,0,"80080103")
    #pull out the hex mass and convert to an image
    #create dummy dictionary for cross-compatibility with other processes
    ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
    ColourImg[:,:,0]=OutputImageB
    ColourImg[:,:,1]=OutputImageG
    ColourImg[:,:,2]=OutputImageR
    SNunter_UserParams_Loaded.ColourBackGroundB=ColourImg
    #use code from serial number cropper to allow user to home into area using arrow keys or whathaveyou
    #_3DVisLabLib.ImageViewer_Quickv2_UserControl(SNunter_UserParams_Loaded.ColourBackGroundB,0,True,True)
    #_3DVisLabLib.ImageViewer_Quickv2_UserControl(SNunter_UserParams_Loaded.ColourBackGroundF,0,True,True)

    global Global_Image
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    while True:
         #update image
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded)
        clone=Global_Image.copy()
        try:
            cv2.rectangle(Global_Image, refPt[0], refPt[1], (0, 255, 0), 2)
        except:
            pass
        # display the image and wait for a keypress
        cv2.imshow("image", Global_Image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded)
        # if the 'c' key is pressed, break from the loop
        if key == ord("1"):
            SNunter_UserParams_Loaded=SetImageParams(SNunter_UserParams_Loaded,True,False,False)
        if key == ord("2"):
            SNunter_UserParams_Loaded=SetImageParams(SNunter_UserParams_Loaded,False,True,False)
        if key == ord("3"):
            SNunter_UserParams_Loaded=SetImageParams(SNunter_UserParams_Loaded,False,False,True)
        elif key == ord("c"):
            break
       
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

SNunter_UserParams_toLoad=SNunter_UserParams()

SN_HuntLoop(SNunter_UserParams_toLoad)

