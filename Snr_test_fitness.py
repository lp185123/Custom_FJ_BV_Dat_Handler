import cv2
import numpy as np
import os
import pytesseract
from pytesseract import*
import difflib
import enum
import random
import _3DVisLabLib
import VisionAPI_Demo
from GeneticAlg_SNR import GA_Parameters#, SNR_fitnessTest
import CompareSNR_Reads
import copy
import TileImages_for_OCR
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
        self.CannyThresh1=100
        self.CannyThresh2=200
        self.AdapativeThreshold=0
        self.MedianBlurDist=0
        self.GausSize_Threshold=0
        self.SubtractMean=0
        self.tessedit_char_whitelist='0123456789'#use if necessary
        self.AlphaBlend=1#one means raw image, 0 is processed image
        self.CropPixels=0
        self.Mirror=True
        self.PSM=3#default is 3
        self.Denoise=0
        self.Negative=0
        self.Equalise=0
        
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
        self.AdapativeThreshold=(self.AdapativeThreshold)
        self.MedianBlurDist=int(self.MedianBlurDist)
        self.Canny=int(self.Canny)

def CheckStringSimilarity(string1,string2):
    if (string1 is None) or (string2 is None):
        return 0

    s = difflib.SequenceMatcher(lambda x: x == " ",string1,string2)
    similiarity_ratio=round(s.ratio(), 3)
    return similiarity_ratio

class CompareOCR_ReadsEnum(enum.Enum):
    Charac="C"
    Num="N"
    WildCard="*"
    Nums_Arabic="0123456789"
    Chars_Latin_UpperCase="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    UnknownChar="?"
    ErrorLength="Length Mismatch"
    ErrorNoMatchFound="No Match Found"
    ErrorMissingData="Missing input data"
    ErrorBadFielding="Bad Fielding error"

class OCR_analysisCard():
    def __init__(self) -> None:
        self.TemplateSNR=None
        self.ExternalSNR=None
        self.ExpectedFielding=None
        self.Error=None
        self.InfoString=None
        self.Pass=None
        self.RepairedExternalOCR=None

    def DoubleSetError(self,Error):
        raise Exception("Error - double load of SNR result card, ",Error )

    def SetInput_TemplateSNR(self,Input):
        if self.TemplateSNR is not None :self.DoubleSetError("TemplateSNR")
        else:
            self.TemplateSNR=Input

    def SetInput_ExternalSNR(self,Input):
        if self.ExternalSNR is not None :self.DoubleSetError("ExternalSNR")
        else:
            self.ExternalSNR=Input

    def SetInput_ExpectedFielding(self,Input):
        if self.ExpectedFielding is not None :self.DoubleSetError("ExpectedFielding")
        else:
            self.ExpectedFielding=Input

    def SetInput_Error(self,Input):
        if self.Error is not None :self.DoubleSetError("Error")
        else:
            self.Error=Input

    def SetInput_InfoString(self,Input):
        if self.InfoString is not None :self.DoubleSetError("InfoString")
        else:
            self.InfoString=Input

    def SetInput_Pass(self,Input):
        if self.Pass is not None :self.DoubleSetError("Pass")
        else:
            self.Pass=Input

def locations_of_substring(string, substring):
    """Return a list of locations of a substring."""
    """https://stackoverflow.com/questions/4664850/how-to-find-all-occurrences-of-a-substring"""
    substring_length = len(substring)    
    def recurse(locations_found, start):
        location = string.find(substring, start)
        if location != -1:
            return recurse(locations_found + [location], location+substring_length)
        else:
            return locations_found

    return recurse([], 0)

def Repair_ExternalOCR(TemplateSNR,ExternalSNR,ExpectedFielding):
    #print("testing for ",TemplateSNR,ExternalSNR,ExpectedFielding)
    #use priors to fix external ocr (such as fielding)
    #also try and align external OCR read with template ocr - as will have a lot of noise
    Subsetsize=3
    Substringlocs=None
    MatchStrings_dict=dict()
    #print("String:",TemplateSNR,ExternalSNR)
    for OuterIndex in range (Subsetsize,len(TemplateSNR)):
        for Index,TemplateSNRChar in enumerate(TemplateSNR):
            #depending on substring size, don't go beyond string size
            if Index==len(TemplateSNR)-OuterIndex+1:break
            TestSubString=TemplateSNR[Index:Index+OuterIndex]
            #try and match subsets of input snr
            #we have have more than 1 instance as images could be mirrored!
            #print(TestSubString)
            Substringlocs_temp=(locations_of_substring(ExternalSNR,TestSubString))
            #build potential semi-matching strings
            #print("test substring",TestSubString)
            for MatchLoc in Substringlocs_temp:
                try:
                    MatchString=ExternalSNR[MatchLoc-Index:(MatchLoc-Index)+len(TemplateSNR)]
                    MatchStrings_dict[MatchString]=MatchString
                    #if string is not correct length - it could have gone over string limit and truncated
                    if len(MatchString)==len(TemplateSNR):
                        MatchStrings_dict[MatchString]=MatchString
                    #print("Match string found",MatchString)
                except:
                    raise("non critical error Repair_ExternalOCR")
            #print("---")
    #roll through discovered strings and align with any fielding
    ListMatches=list(MatchStrings_dict.keys())
    OutputString_list=[]
    for OuterIndex, item in enumerate(ListMatches):
        #align to fielding - if fielding exists
        for indexer, FieldingType in enumerate(ExpectedFielding):
            TempValue=None
            if FieldingType==CompareOCR_ReadsEnum.Charac.value:
                if indexer>len(item)-1:break
                if item[indexer]=="5":TempValue="S"
                if item[indexer]=="1":TempValue="I"
                if item[indexer]=="6":TempValue="G"
                if item[indexer]=="0":TempValue="O"
                if item[indexer]=="8":TempValue="B"
            if FieldingType==CompareOCR_ReadsEnum.Num.value:
                if indexer>len(item)-1:break
                if item[indexer]=="S":TempValue="5"
                if item[indexer]=="I":TempValue="1"
                if item[indexer]=="i":TempValue="1"
                if item[indexer]=="G":TempValue="6"
                if item[indexer]=="O":TempValue="0"
                if item[indexer]=="o":TempValue="0"
                if item[indexer]=="B":TempValue="8"
                if item[indexer]=="D":TempValue="0"
                if item[indexer]=="Q":TempValue="0"
                
            
            if TempValue is None:
                OutputString_list.append(str(ListMatches[OuterIndex][indexer]))
            else:
                #print("repaired",TempValue)
                OutputString_list.append(str(TempValue))
        OutputString_list.append("     ")

    #join list of chars into a string
    OutputString = ''.join(OutputString_list)

    if OutputString is None:
        return ExternalSNR
    if len(OutputString)==0:
        return ExternalSNR

    return OutputString

def CompareOCR_Reads(TemplateSNR,ExternalSNR,ExpectedFielding=None):

    
    #compare OCR reads, and return info card with pass/fail and error messages
    #issue is same chars being confused by both OCR services (such as O/0, I/1 etc) - maybe warn user if confusion chars are present
    #should we assume at this stage that the template has pre-filtered serial number reads that dont match the expected fielding?
    
    #print(vars(CompareOCR_Reads("AD48F5-BB","AD48F5-BBAD48F5BBCC12s3CC 3CCc","CCNNCN*CC")))
    OCR_analysis=OCR_analysisCard()
    OCR_analysis.ExpectedFielding=ExpectedFielding
    OCR_analysis.TemplateSNR=TemplateSNR
    OCR_analysis.ExternalSNR=ExternalSNR

    ConfusionPairs=[("8","B"),("S","5"),("I","1"),("3","8"),("G","6"),("0","O"),("6","9"),("I","1")]
    #roll through test cases and break out if test fails

    while True:#loop through tests - should use internal function with scoped variables for this probably nicer

        if (TemplateSNR is None) or (TemplateSNR=="") or (ExternalSNR is None) or (ExternalSNR==""):
            OCR_analysis.SetInput_InfoString("missing input data")
            OCR_analysis.SetInput_Error(CompareOCR_ReadsEnum.ErrorMissingData.value)
            OCR_analysis.SetInput_Pass(False)
        
        if OCR_analysis.Pass==False:break

        #expected fielding has been provided - do base checks on TemplateSNR
        if ExpectedFielding is not None:
            if len(ExpectedFielding)!=len(TemplateSNR):
                OCR_analysis.SetInput_InfoString("test snr length does not match fielding length")
                OCR_analysis.SetInput_Error(CompareOCR_ReadsEnum.ErrorLength.value)
                OCR_analysis.SetInput_Pass(False)

        if OCR_analysis.Pass==False:break

        if ExpectedFielding is not None:
            #check if characters match expected fielding
            for Index, Elem in enumerate(ExpectedFielding):
                #test that input matches expected fieldings: alpha
                if Elem ==CompareOCR_ReadsEnum.Charac.value:
                    if TemplateSNR[Index] not in CompareOCR_ReadsEnum.Chars_Latin_UpperCase.value:
                        OCR_analysis.SetInput_InfoString(TemplateSNR[Index] +" not found in " + CompareOCR_ReadsEnum.Chars_Latin_UpperCase.value)
                        OCR_analysis.SetInput_Pass(False)
                        OCR_analysis.SetInput_Error(CompareOCR_ReadsEnum.ErrorBadFielding.value)
                        break
                        
                #test that input matches expected fieldings: numerals
                elif Elem ==CompareOCR_ReadsEnum.Num.value:
                    if TemplateSNR[Index] not in CompareOCR_ReadsEnum.Nums_Arabic.value:
                        OCR_analysis.SetInput_InfoString(TemplateSNR[Index] +" not found in " + CompareOCR_ReadsEnum.Nums_Arabic.value)
                        OCR_analysis.SetInput_Pass(False)
                        OCR_analysis.SetInput_Error(CompareOCR_ReadsEnum.ErrorBadFielding.value)
                        break
                        
                #test that input matches expected fieldings: wildcard
                elif Elem ==CompareOCR_ReadsEnum.WildCard.value:
                    pass
                #input fielding does not match - error 
                else:
                    OCR_analysis.SetInput_InfoString(ExpectedFielding + " does not conform to expected inputs")
                    OCR_analysis.SetInput_Pass(False)
                    OCR_analysis.SetInput_Error(CompareOCR_ReadsEnum.ErrorBadFielding.value)
                    break

        if OCR_analysis.Pass==False:break

        #repair external OCR if we have prior knowledge
        RepairedExternalOCR=Repair_ExternalOCR(TemplateSNR,ExternalSNR,ExpectedFielding)
        OCR_analysis.RepairedExternalOCR=RepairedExternalOCR

        #Basic check - can we find TemplateSNR in the ExternalSNR string
        if (( TemplateSNR in RepairedExternalOCR)==False) and (( TemplateSNR in ExternalSNR)==False):
            OCR_analysis.SetInput_InfoString("test snr not found in external snr")
            OCR_analysis.SetInput_Pass(False)
            OCR_analysis.SetInput_Error(CompareOCR_ReadsEnum.ErrorNoMatchFound.value)

        if OCR_analysis.Pass==False:break

        #explicit check of string matching
        if TemplateSNR in RepairedExternalOCR:
            OCR_analysis.SetInput_InfoString("string found")
            OCR_analysis.SetInput_Pass(True)
            OCR_analysis.SetInput_Error(None)
            break

        #default break
        break

    #return analysis card
    return(OCR_analysis)

def GenerateSN_Fielding(InputListSNRReads):
    #roll through input list of snr reads in format "xxx[....]xxx" and calculate potential character fielding (consistent format of nums/alphabet)
    SNR_Dict=dict()
    for Elemn in InputListSNRReads:

        Known_SNR_string=None
        #extract snr - by previous process will be bookened by "[" and "]"
        try:
            if not "[" in Elemn:
                continue
            if not "]" in Elemn:
                continue
            Get_SNR_string=Elemn.split("[")#delimit
            Get_SNR_string=Get_SNR_string[-1]#get last element of delimited string
            Get_SNR_string=Get_SNR_string.split("]")#delimit
            Get_SNR_string=Get_SNR_string[0]
            if (Get_SNR_string is not None):
                if (len(Get_SNR_string))>0:
                    Known_SNR_string=Get_SNR_string
                    #load image file path and SNR as key and value
                    SNR_Dict[Elemn]=Known_SNR_string
                    #print(Elemn,Known_SNR_string)
        except Exception as e: 
            print("error extracting known snr string from file ",Elemn )
            print(repr(e)) 
        
    if len(SNR_Dict.keys())==0:
        raise Exception("GenerateSN_Fielding: No serial numbers found!")
    
    #check general format of SNR
    #check 1 - are all the same length? if not - can't proceed as format will be inconsistent
    TestLength=(random.choice(list(SNR_Dict.values())))
    for Elem in SNR_Dict.values():
        if len(Elem)!=len(TestLength):
            raise Exception("SNR size mismatch - cannot automatically define fielding: ", Elem, "vs",TestLength)

    #all SNR reads same length - can test each element for type (num/alphabet)
    TestLength=(random.choice(list(SNR_Dict.values())))#repeat get length
    ListSNR_values=list(SNR_Dict.values())#convert to list for readibility
    FieldingString=[]
    #build fielding list
    for BuildList in range (len(TestLength)):
        FieldingString.append(None)
    #take slices of each char position for list of SNR
    for CharPosition in range (0,len(TestLength)):
        ListAllSNR_Charposition=([x[CharPosition] for x in ListSNR_values])
        for charac in ListAllSNR_Charposition:#
            FieldingString[CharPosition]=SetCharType(charac,FieldingString,CharPosition)

    return FieldingString

def SetCharType(InputChar,FieldingString,CharPosition):
    #if CharPosition in FieldingString has not been set - then set it
    #if not and its a different type of character (num/alpha)- set to WildCard character
    CompareOCR_ReadsEnum
    ReturnValue=None
    if InputChar in CompareOCR_ReadsEnum.UnknownChar.value:
        ReturnValue=FieldingString[CharPosition]
        return ReturnValue
    if FieldingString[CharPosition] is None:
        if InputChar in CompareOCR_ReadsEnum.Nums_Arabic.value:
            ReturnValue=CompareOCR_ReadsEnum.Num.value
            return ReturnValue
        elif InputChar in CompareOCR_ReadsEnum.Chars_Latin_UpperCase.value:
            ReturnValue=CompareOCR_ReadsEnum.Charac.value
            return ReturnValue
        elif InputChar in CompareOCR_ReadsEnum.UnknownChar.value:
            ReturnValue=None
            return ReturnValue
    if (FieldingString[CharPosition] == CompareOCR_ReadsEnum.Num.value) and (InputChar in CompareOCR_ReadsEnum.Nums_Arabic.value):
        ReturnValue=FieldingString[CharPosition]
        return ReturnValue
    if (FieldingString[CharPosition] == CompareOCR_ReadsEnum.Charac.value) and (InputChar in CompareOCR_ReadsEnum.Chars_Latin_UpperCase.value):
        ReturnValue=FieldingString[CharPosition]
        return ReturnValue

    #default - wildcard - no match found or mismatch for character position or special character
    return CompareOCR_ReadsEnum.WildCard.value
    
def GetFieldingOf_SNR_ProcessedImages():
    #test function to automatically find fielding of snr processed files in format:
    #B6799032875A]SRU SNR image1 None_0007 record 36 file 6.jpg
    Result = input("Please enter folder for analysis:")
    InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(Result)
    ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)
    print(GenerateSN_Fielding(ListAllImages))

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def CreateTestImages():
    #Quick test function - to be d
    #Do import here as we will probably move this function or delete it
    import uuid
    def my_random_string(string_length=10):
        """Returns a random string of length string_length."""
    # Convert UUID format to a Python string.
        random = str(uuid.uuid4())
    # Make all characters uppercase.
        random = random.upper()
    # Remove the UUID '-'.
        random = random.replace("-","")
    # Return the random string.
        return random[0:string_length]

    
    #quick tool to create test images for OCR
    ResultFolder = input("Please enter folder for test images:")
    NumberOfImages=300
    ImageSizeX=170
    ImageSizeY=140
    Xbuffer=10
    BackgroundImage_toload=r"C:\Working\FindIMage_In_Dat\Media\DirtyBackground.jpg"
    BackGroundImage=cv2.imread(BackgroundImage_toload)

    for Img in range (NumberOfImages):

        #grab a portion of the background image
        #get range for grabbing a window from the background image that fits export image size
        BackgroundImg_WindowRangeX=(BackGroundImage.shape[1]-ImageSizeX)
        BackgroundImg_WindowRangeY=(BackGroundImage.shape[0]-ImageSizeY)
        BackGroundWindowStart_X=random.randint(0,BackgroundImg_WindowRangeX)
        BackGroundWindowStart_Y=random.randint(0,BackgroundImg_WindowRangeY)
        
        #create blank canvases for serial number
        RGB_Image = np.zeros((int(ImageSizeY),int(ImageSizeX),3), np.uint8)
        RGB_Image_Alphamatte = np.zeros((int(ImageSizeY),int(ImageSizeX),3), np.uint8)
        RandomCroppedBackground=np.zeros((int(ImageSizeY),int(ImageSizeX),3), np.uint8)
        #we have a starting position now - should be able to cut a section of the background out
        RandomCroppedBackground=BackGroundImage[BackGroundWindowStart_Y:BackGroundWindowStart_Y+ImageSizeY,BackGroundWindowStart_X:BackGroundWindowStart_X+ImageSizeX,:]

        
        #generate random text string
        Text=my_random_string(10)
        #fit text string into canvas
        TextScale=_3DVisLabLib.get_optimal_font_scale(Text,ImageSizeX-Xbuffer,ImageSizeY)#warning: this does not assume the same font
        font = cv2.FONT_HERSHEY_SIMPLEX
        #draw on text
        cv2.putText(RGB_Image, Text, (int(Xbuffer/2),int(ImageSizeY/2)), font, TextScale, (255, 255, 20), 1, cv2.LINE_AA)
        #to use alpha blending we need a matte image - so all black except for text
        cv2.putText(RGB_Image_Alphamatte, Text, (int(Xbuffer/2),int(ImageSizeY/2)), font, TextScale, (255, 255, 255), 1, cv2.LINE_AA)
        # Normalize the alpha mask to keep intensity between 0 and 1
        RGB_Image_Alphamatte = RGB_Image_Alphamatte.astype(float)/255
        # Convert uint8 to float
        foreground = RGB_Image.astype(float)
        background = RandomCroppedBackground.astype(float)
        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(RGB_Image_Alphamatte, foreground)
        # Multiply the background with ( 1 - alpha )
        #background = cv2.multiply(1.0 - RGB_Image_Alphamatte, background*random.random())
        background = cv2.multiply(1.0 - RGB_Image_Alphamatte, background*0)
        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)

        #inverse
        #outImage= cv2.bitwise_not(outImage)
        #add noise
        #outImage=sp_noise(outImage,random.random()/100)
        #resize
        #RGB_Image=ResizeImage(RGB_Image,50,150)
        Savestring=ResultFolder + "\\TEST_IMAGE_" + "[" + Text + "] "+ str(Img) + ".jpg" 
        cv2.imwrite(Savestring,outImage)

def GetCollimatedImages_and_Answers(ImagePaths_Dict,ColSize,DelimitText,Xbuffer):
# #collimate images and load images into memory alongside SN answers
    DelimiterImage,ImageX,ImageY=TileImages_for_OCR.CreateDelimiterImage(ImagePaths_Dict,DelimitText,Xbuffer)#use valid image file from set to create delimiter image
    
    ImgPath_VS_ImageAndAnswer=TileImages_for_OCR.TileImages_with_delimiterImage(DelimiterImage,ImagePaths_Dict,ImageY,ImageX,ColSize,"Dummy parameter not needed")
        #display out images and answer files
    #for Index, CollumnItem in enumerate(ImgPath_VS_ImageAndAnswer):
     #   _3DVisLabLib.ImageViewer_Quickv2(ImgPath_VS_ImageAndAnswer[CollumnItem][2],0,True,True)
    
    return ImgPath_VS_ImageAndAnswer

def MirrorImage(InputImage):
    #double up image 
    FlippedImage = cv2.flip(InputImage, -1)
    #create blank image twice the height
    #shape 0 is X shape 1 is Y
    blank_image = (np.ones((InputImage.shape[0]*2,InputImage.shape[1]), dtype = np.uint8))*255
    blank_image[0:InputImage.shape[0],:]=InputImage
    blank_image[InputImage.shape[0]:InputImage.shape[0]*2,:]=FlippedImage
    TestImage=blank_image.copy()
    return TestImage



def ProcessImage(TestImage,ParameterObject):
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
        ProcessedImage = cv2.GaussianBlur(ProcessedImage, (7, 7), 0)
        ProcessedImage=cv2.adaptiveThreshold(ProcessedImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,ParameterObject.GausSize_Threshold,ParameterObject.SubtractMean)
        #ProcessedImage = cv2.adaptiveThreshold(ProcessedImage, ParameterObject.AdapativeThreshold, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,ParameterObject.GausSize_Threshold,ParameterObject.SubtractMean) #cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)[1]
    
    if ParameterObject.Canny!=0:
        ProcessedImage=cv2.Canny(ProcessedImage, ParameterObject.CannyThresh1, ParameterObject.CannyThresh2)

    if ParameterObject.Denoise==1:
        ProcessedImage=cv2.fastNlMeansDenoising(ProcessedImage)

    if ParameterObject.Equalise==1:
        ProcessedImage = cv2.equalizeHist(ProcessedImage)

    #blend processed and base(resized)image according to Alpha blend parameter
    beta = (1.0 - ParameterObject.AlphaBlend)
    TestImage = cv2.addWeighted(TestImage, ParameterObject.AlphaBlend, ProcessedImage, beta, 0.0)

    if ParameterObject.Negative==1:
        #TestImage = cv2.bitwise_not(TestImage)
        TestImage = 1 - TestImage

    #TestImage = cv2.morphologyEx(TestImage, cv2.MORPH_OPEN, ParameterObject.kernel)
    #TestImage = cv2.morphologyEx(TestImage, cv2.MORPH_CLOSE, ParameterObject.kernel)
    #img = image_smoothening(img)
    #TestImage = cv2.bitwise_or(TestImage, closing)

    return TestImage


def GoogleOCR(TestImage,GenParams):
    if GenParams.CloudOCRObject is not None and GenParams.UseCloudOCR==True:
        Savestring="X://tempimg.jpg"
        cv2.imwrite(Savestring,TestImage)
        #get OCR from Google VIsion API
        #filtered response - only alphanumeric chars
        OCR_Result=GenParams.CloudOCRObject.PerformOCR(Savestring,None)
        return OCR_Result

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

        self.LastRequestImages=None#remember last set of image paths to be requested
        self.ColImagesVAnswers=None#
        self.ImgVsPath_dict=None
        #self.CloudOCR=VisionAPI_Demo.CloudOCR()



    def RunSNR_With_Parameters(self,ImagePaths,ParameterObject,TestImages=None,SkipOcr=False,GenParams=None,ColSize=None):
        
        #make sure no invalid values
        ParameterObject.Housekeep()
        ParameterObject.SetConfig()

        
        if TestImages is None:
            #load image#
            #TODO dont think we need this at this point
            TestImages=[]
            for ImagePath in ImagePaths:
                TestImages.append(cv2.imread(ImagePath,cv2.IMREAD_GRAYSCALE))

        #convert keys of dictionary to list
        ImagePaths_list=list(ImagePaths.keys())
        #check we havent processed these images alread
        if self.LastRequestImages==ImagePaths_list:
            pass
        else:
            print("Loading images into memory")
            #load all iamges and process them
            self.ImgVsPath_dict=None
            self.ImgVsPath_dict=dict()
            for Img in ImagePaths_list:
                self.ImgVsPath_dict[Img]=cv2.imread(Img,cv2.IMREAD_GRAYSCALE)
            #update last set of images to avoid reloading 
            self.LastRequestImages=ImagePaths_list

        #hold the processed image and answers
        ProcessedImgVsPath_dict=dict()
        #process pre-loaded raw images and create collimated image
        for ImagetoProcess in self.ImgVsPath_dict.keys():
            ProcessedImage=ProcessImage(self.ImgVsPath_dict[ImagetoProcess],ParameterObject)
            ProcessedImgVsPath_dict[ImagetoProcess]=ProcessedImage.copy()
        #image to user
        TempImage=ProcessedImgVsPath_dict[list(ProcessedImgVsPath_dict.keys())[0]]
        _3DVisLabLib.ImageViewer_Quick_no_resize(TempImage,0,False,False)

        #now collimate processed images
        ProcessedColImagesVAnswers=GetCollimatedImages_and_Answers(ProcessedImgVsPath_dict,ColSize,"DISPATCH",20)
        
        #Perform OCR
        if SkipOcr==False:
            if GenParams.CloudOCRObject is not None and GenParams.UseCloudOCR==True:
                ProcessedIMg_VS_CloudOCR=dict()
                for Img in ProcessedColImagesVAnswers:
                    RawOCRResult=GoogleOCR(ProcessedColImagesVAnswers[Img][2],GenParams)
                    DelimitedLines=RawOCRResult.split("DISPATCH")#TODO make this common
                    Cleanedlines=[]
                    CountSingleAnswers=0
                    #can get empty lines with delimiter
                    for Index, Singleline in enumerate(DelimitedLines):
                        #delimiter may give us an empty final line which can throw up an error for alignment
                        #and may also legimiately get empty lines back from external OCR
                        if (Index< len(DelimitedLines)-1) or (len(DelimitedLines)==1):
                            CleanedLine=CompareSNR_Reads.CleanUpExternalOCR(Singleline)
                            Cleanedlines.append(CleanedLine)
                            CountSingleAnswers=CountSingleAnswers+1
                    ProcessedIMg_VS_CloudOCR[Img]=(Cleanedlines,CountSingleAnswers)
                    if (len(ProcessedColImagesVAnswers[Img][1])!=len(ProcessedIMg_VS_CloudOCR[Img][0])):
                        raise Exception("RunSNR_With_Parameters, answers different lengths",len(ProcessedColImagesVAnswers[Img][1]),len(ProcessedIMg_VS_CloudOCR[Img][0]))
        
                #now align answers with traceback to original image file
                TemplateSNR_list=[]
                ExternalSNR_list=[]
                for Img in ProcessedIMg_VS_CloudOCR:
                    for OCRread in ProcessedIMg_VS_CloudOCR[Img][0]:
                        ExternalSNR_list.append(OCRread)
                    for OCRread in ProcessedColImagesVAnswers[Img][1].values():
                        TemplateSNR_list.append(OCRread[0])
                if len(ExternalSNR_list)!=len(TemplateSNR_list):
                    raise Exception("RunSNR_With_Parameters, list of snr and ocr do not match",len(ExternalSNR_list),len(TemplateSNR_list))
                #now have two lists which align, one with template SNR and one with external OCR

                #roll through results and get report card - will handle fielding and other issues
                ResultsObjectList=[]
                for Index in range (len(ExternalSNR_list)):
                    ResultsObjectList.append(CompareSNR_Reads.CheckSNR_Reads(TemplateSNR_list[Index],ExternalSNR_list[Index],GenParams.Fielding))

                #tally up results
                FitnessTally=[]
                FitnessScore=0
                for ResultCard in ResultsObjectList:
                    if ResultCard.Pass==True:
                        FitnessTally.append(1)
                        FitnessScore=FitnessScore+1.0
                    else:
                        #get comparison ratio
                        CompareStringScore=round(CheckStringSimilarity(ResultCard.TemplateSNR,ResultCard.RepairedExternalOCR),5)
                        FitnessTally.append(0.0)
                        #try having a score of zero instead of a continuum
                        #FitnessScore=FitnessScore+CompareStringScore

                FinalScore=FitnessScore/len(ResultsObjectList)
                return TempImage,FinalScore



        if SkipOcr==True:
            return TempImage,0.0#return dummy final score


        if GenParams.CloudOCRObject is not None and GenParams.UseCloudOCR==True:
                #need to save image to give to google
                #save to RAM DRIVE for now

                Savestring="X://tempimg.jpg"
                cv2.imwrite(Savestring,TestImage)
                #get OCR from Google VIsion API
                #filtered response - only alphanumeric chars
                OCR_Result=GenParams.CloudOCRObject.PerformOCR(Savestring,None)
                #have to add square brackets to make it compatible with this function
                CompareResult=CompareSNR_Reads.CheckSNR_Reads("[" +self.Known_SNR_string +"]",OCR_Result,GenParams.Fielding)
                self.Known_Snr_Fitness= round(CheckStringSimilarity(self.Known_SNR_string,CompareResult.RepairedExternalOCR),3)
                #print("Cloud OCR score",self.Known_Snr_Fitness,"raw cloud OCR",OCR_Result,"Repaired cloud OCR",CompareResult.RepairedExternalOCR, "test ocr",self.Known_SNR_string)
                return TestImage,self.Known_Snr_Fitness



        #run OCR
        #use google OCR if enabled
        if SkipOcr==False:
            if GenParams.CloudOCRObject is not None and GenParams.UseCloudOCR==True:
                #need to save image to give to google
                #save to RAM DRIVE for now

                Savestring="X://tempimg.jpg"
                cv2.imwrite(Savestring,TestImage)
                #get OCR from Google VIsion API
                #filtered response - only alphanumeric chars
                OCR_Result=GenParams.CloudOCRObject.PerformOCR(Savestring,None)
                #have to add square brackets to make it compatible with this function
                CompareResult=CompareSNR_Reads.CheckSNR_Reads("[" +self.Known_SNR_string +"]",OCR_Result,GenParams.Fielding)
                self.Known_Snr_Fitness= round(CheckStringSimilarity(self.Known_SNR_string,CompareResult.RepairedExternalOCR),3)
                #print("Cloud OCR score",self.Known_Snr_Fitness,"raw cloud OCR",OCR_Result,"Repaired cloud OCR",CompareResult.RepairedExternalOCR, "test ocr",self.Known_SNR_string)
                return TestImage,self.Known_Snr_Fitness
                


        #default to pytesseract ocr
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