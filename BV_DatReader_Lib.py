
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


class OperationCodes(enum.Enum):
    ERROR="error"


class UserOperationStrings(enum.Enum):
    #TODO map user key press here rather than in two places - interrogate the enum instead of dictionary
    UP="w"
    DOWN="s"
    LEFT="a"
    RIGHT="d"
    RANDOM="r"
    UP2="i"
    DOWN2="k"
    LEFT2="j"
    RIGHT2="l"
    A="z"
    B="x"
    EXTRACT="c"
    COLOURS="C"
    RED="b"
    GREEN="n"
    BLUE="m"
    COLOURUP="W"
    COLOURDOWN="S"
    SNRFIND="t"
    Test="p"
    SHIFT="o"
    GetALLSNR="["

    def MapKeyPressToString(self,keypress):
        if keypress in self.MapKeyPress_to_String:
            return self.MapKeyPress_to_String[keypress]
        else:
            return None

class UserInputParameters():

    def __init__(self):
        self.InputFilePath=""
        self.OutputFilePath=""
        self.FirstImageOnly=True
        self.AutomaticMode=True
        self.BlockType_ImageFormat=""
        self.BlockTypeWave=""
        self.GetSNR=False
        self.UserInput_and_test()
        self.PrintAllUserInputs()

    def PrintAllUserInputs(self):
        #self test for developer
        ListParams=[self.InputFilePath,
        self.OutputFilePath,
        self.FirstImageOnly,
        self.AutomaticMode,
        self.BlockType_ImageFormat,
        self.BlockTypeWave]

        print(ListParams)

    def UserInput_and_test(self):
        ##build user requests for operation
        

        #get all files in input folder
        while True:
            #ask what is input folder
            Result = input("Please enter folder for analysis:")
            InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(Result)
            #clean files
            InputFiles_cleaned=[]
            for elem in InputFiles:
                if ((".dat") in elem.lower()):
                    InputFiles_cleaned.append(elem) 
            print("Dat files found in folder: ", InputFiles_cleaned)
            if (InputFiles_cleaned) is None or len(InputFiles_cleaned)<1:
                print("Invalid folder or folder & nested folders have no .dat files")
            else:
                self.InputFilePath=Result
                break
        
            #output folder hardcoded for now - warning we dont want the user to set same 
            #folder for output and input as files will be deleted - but generally will put preview images
            #alongside data

        self.AutomaticMode=_3DVisLabLib.yesno("Automatic(y) or Manual (m) mode?")
        if self.AutomaticMode==False: return False
        self.FirstImageOnly=_3DVisLabLib.yesno("All Images (y) or First image only (n)?")

        while True:
            #ask user what block type (format of image)
            for Elem in DatScraper_tool.ImageExtractor.BLOCK_TYPES:
                if Elem in DatScraper_tool.ImageExtractor.IMAGE_TYPES:
                    print(Elem, DatScraper_tool.ImageExtractor.BLOCK_TYPES[Elem])
            Result = input("Please enter image format using ID number - non-image data available but not developed:")
            numeric_filter = filter(str.isdigit, Result)
            numeric_string = "".join(numeric_filter)
            if int(numeric_string) in DatScraper_tool.ImageExtractor.IMAGE_TYPES:#TODO this will crash if user puts in non numeric chars probably
                self.BlockType_ImageFormat=DatScraper_tool.ImageExtractor.BLOCK_TYPES[int(numeric_string)]
                break
            else:
                print("\n \n \n \n \n")
                print(Result, "is an invalid choice - please try again")

        while True:
            #ask user what block type (format of image)
            print("\n \n \n \n \n")
            for Elem in DatScraper_tool.ImageExtractor.WAVE_DICTIONARY:
                print(DatScraper_tool.ImageExtractor.WAVE_DICTIONARY[Elem])
            Result = input("Please enter wave type:")
            if Result in DatScraper_tool.ImageExtractor.WAVE_DICTIONARY.values():
                self.BlockTypeWave=Result
                break
            else:
                print("\n \n \n \n \n")
                print(Result, "is an invalid choice - please try again")
            
        while True:
            GetSNR=_3DVisLabLib.yesno("Get Serial Number if available? y/n")

def AutomaticExtraction(UserParameters):
    #process according to user input parameters
    #     self.InputFilePath=""
    #     self.OutputFilePath=""
    #     self.FirstImageOnly=True
    #     self.AutomaticMode=True
    #     self.BlockType_ImageFormat=""
    #     self.BlockTypeWave=""

    #get all files in input folder - already done but in case we change logic 
    InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(UserParameters.InputFilePath)
    #clean files
    InputFiles_cleaned=[]
    for elem in InputFiles:
        if ((".dat") in elem.lower()):
            InputFiles_cleaned.append(elem)
    for DatFile in InputFiles_cleaned:
    #get all images from first .dat file in input folder(s) (nested)
        images = DatScraper_tool.ImageExtractor(DatFile)
        #filter for SNR images
        filteredImages = images.filter(UserParameters.BlockType_ImageFormat,UserParameters.BlockTypeWave)
        


        if UserManualMemoryScan.SNR_extractMode==SNR_string_extract.Automatic:
            #user has activated automatic extraction of serial number reads
            images = DatScraper_tool.ImageExtractor(FileName)
            for Index, Item in enumerate(images.notes):
                Tempsnr=str(Item.snr)
                if Tempsnr is None:
                    SNR_Results_per_record[Index+1]=(OperationCodes.ERROR)
                    raise Exception("Could not parse SNR")
                elif Tempsnr =="":
                    SNR_Results_per_record[Index+1]=(OperationCodes.ERROR)
                    raise Exception("Could not parse SNR")
                else:
                    SNR_Results_per_record[Index+1]="[" + (str(Item.snr)) + "]" + str(UserManualMemoryScan.DatFileDataType)






class DatFileInfos():
    #do not instance class - reference only
    KnownWidths_Heights=[(186,88),(204,160),(22,10),(160,40),(336,56),(128,32),(1632,640),((1632*2),640)]
    KnownColourChannel_OffsetBytes=[49008,32784]
    SNR_Result_Chunk=190#arbitrary size of chunk we strip out of .dat file to grab all SNR read result
    KnownHeaders=["000000000000B006","0000000000000003","000000000000414B", "0000000000003132","000000000000987F","0000000000000840","0000000000001832"  ]

def Load_Hex_File(Filepath):
    print("Attempting to open ", Filepath)
    with open(Filepath, 'rb') as f:
        hexdata = f.read().hex()
        print(len(hexdata)/2)#length in hex 
        print("Opened")
        return hexdata

def ConvertHex_to_decimal(InputHex):
        #example 0000000F = 15
        #so 15 sets of 255, 15 sets of double hex chars
        i = (int(InputHex, 16))
        return i

def ImageViewer_Quickv2(inputimage,pausetime_Secs=0,presskey=False,destroyWindow=True):
    ###handy quick function to view images with keypress escape andmore options
    CopyOfImage=cv2.resize(inputimage.copy(),(800,800))
    cv2.imshow("img", CopyOfImage); 
    
    if presskey==True:
        cv2.waitKey(0); #any key

    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                #for some reason
                pass
            
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()

def ImageViewer_Quick_noResize(inputimage,pausetime_Secs=0,presskey=False,destroyWindow=True):
    ###handy quick function to view images with keypress escape andmore options
    #CopyOfImage=cv2.resize(inputimage.copy(),(800,800))
    cv2.imshow("img", inputimage); 
    
    if presskey==True:
        cv2.waitKey(0); #any key

    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                #for some reason
                pass
            
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()


def CleanDataFromNonHex(InputList):
    data_clean=[]
    for Elem in InputList:
        if Elem in ["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f","A","B","C","D","E","F"]:
            data_clean.append(Elem)
    return data_clean
    
def Return_SubSet_of_Memory(List_of_Bytes,StartHex,EndHEx,Offset_bytes_decimal):
    StartDecimal=(ConvertHex_to_decimal(StartHex))
    EndDecimal=(ConvertHex_to_decimal(EndHEx))
    Subset=List_of_Bytes[Offset_bytes_decimal+StartDecimal:Offset_bytes_decimal+EndDecimal]
    return Subset

def CleanDataFromNonHex(InputList):
    data_clean=[]
    for Elem in InputList:
        if Elem in ["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f","A","B","C","D","E","F"]:
            data_clean.append(Elem)
    return data_clean

def ConvertListHex_to_integers(InputListHex_str):
    #print(InputListHex_str)
    ListOfGrayScaleInt=[]
    for i in range (0,len(InputListHex_str),2):
        try:
            #create set of two hex elements
            HexSet=""
            HexSet=InputListHex_str[i] + InputListHex_str[i+1]#sets of two
            Int_from_Hex = int(HexSet, 16)
            ListOfGrayScaleInt.append(Int_from_Hex)
        except Exception as e: 
            print(repr(e))
            print(HexSet, " not valid hex")

    return ListOfGrayScaleInt

def mapFromTo(x,a,b,c,d):
    # x:input value; 
    # a,b:input range
    # c,d:output range
    # y:return value
   y=(x-a)/(b-a)*(d-c)+c
   return y

def ConvertListHex_to_GrayScale_bitdepth(InputListHex_str,bytedepth,OutputGrayScale_Decimal):
    ListOfGrayScaleInt=[]
    for i in range (0,len(InputListHex_str),bytedepth):
        try:
            #create set of n hex elements
            HexSet=""
            for j in range(i,i+bytedepth):
                HexSet=HexSet+InputListHex_str[j]
            #what base do we convert set of hex into,images are usually 2 bytes (256 per channel) or can be 4 bytes (1024) etc
            Int_from_Hex = int(HexSet,16)#TODO if this is a bitdepth greater than 16 bits why doesnt "bytedepth*8" work?
            #Int_from_Hex=mapFromTo(Int_from_Hex,0,(bytedepth*8)*(bytedepth*8),0,OutputGrayScale_Decimal)
            ListOfGrayScaleInt.append(Int_from_Hex)
        except Exception as e: 
            print(repr(e))
            print(HexSet, " not valid hex")

    return ListOfGrayScaleInt

def CreateGrayImage_from_memory(List_decimalValues, WidthImage,HeightImage):
    #create grayscale and colour images - colour here is for undefined areas of memory (red) and defined (green)
    DataVerify_image = np.zeros((HeightImage,WidthImage,3), np.uint8)
    gray_image = cv2.cvtColor(DataVerify_image, cv2.COLOR_BGR2GRAY)
    ListOfGrayScale_temp=copy.copy(List_decimalValues)

    for Yelement in range (0,HeightImage,1):
        for Xelement in range (0,WidthImage,1):
            IndexRequired=((Yelement)*WidthImage)+Xelement
            if IndexRequired>len(ListOfGrayScale_temp)-1:
                gray_image[Yelement,Xelement]=0
                DataVerify_image[Yelement,Xelement,0]=0
                DataVerify_image[Yelement,Xelement,1]=0
                DataVerify_image[Yelement,Xelement,2]=255
            else:
                #valid grayscale values will be in green channel
                if ListOfGrayScale_temp[IndexRequired]>-1 or ListOfGrayScale_temp[IndexRequired]<256:
                    gray_image[Yelement,Xelement]=ListOfGrayScale_temp[IndexRequired]
                    DataVerify_image[Yelement,Xelement,0]=0
                    DataVerify_image[Yelement,Xelement,1]=ListOfGrayScale_temp[IndexRequired]
                    DataVerify_image[Yelement,Xelement,2]=0
                else:
                    #invalid greyscale values in blue channel
                    gray_image[Yelement,Xelement]=ListOfGrayScale_temp[IndexRequired]
                    DataVerify_image[Yelement,Xelement,0]=255
                    DataVerify_image[Yelement,Xelement,1]=0
                    DataVerify_image[Yelement,Xelement,2]=0

    return (gray_image,DataVerify_image)
   
def GetDataFile(InputPath):
    #Load input data
    #data=Load_Hex_File(r"C:\Working\FindIMage_In_Dat\01_5_A.dat")
    data=Load_Hex_File(InputPath)

    print("skipping cleaning out")
    #Clean out non hex characters
    #data_clean=CleanDataFromNonHex(data)
    data_clean=data
    print("Fin cleaning out")

    print("Convert to grayscale")
    #convert list of hex to list of 00-FF range elements (grayscale)
    ListOfGrayScale=ConvertListHex_to_integers(data_clean)
    print("Fin Convert")

class UserManualMemoryScan_helper():
    def __init__(self):
        self.Offset=0
        self.OffsetEnd=0
        self.Width=200
        self.Height=100
        self.Multiplier=1
        self.WidthRandomRange=1000
        self.CycleThroughKnownDims=0

        self.RedChannel=None
        self.GreenChannel=None
        self.BlueChannel=None
        self.RedChannelOffset_bytes=None
        self.GreenChannelOffset_bytes=None
        self.BlueChannelOffset_bytes=None
        self.ColourChannelOffset=None
        self.HeaderStartGuideImg=None
        self.ColourOffsetManual=0

        self.DatFileDataType=""

        self.UserControlMapping=None
        self.SNR_extractMode=SNR_string_extract.Off
        self.SNR_PrefixData=None#for manual SNR

        #datamass skimmer images
        self.DataVerify_image=None
        self.gray_image=None




    def PrintInfo(self):
        ByteOffset=hex(int(self.Offset/2))
        print("ByteOffset: ", ByteOffset, " Offset: ",hex(self.Offset), "   OffsetEnd: ", self.OffsetEnd, "   Width:", self.Width,"   Height: ", self.Height)

#SNR mode automatic
class SNR_string_extract(enum.Enum):
    Off="off"
    Manual="manual"
    Automatic="automatic"

def ReplaceHexStrings(InputhHex,DatFileInfos):
    for Item in DatFileInfos.KnownHeaders:
        print(str(ConvertHex_to_decimal(Item)))
    #replace strings of text with 

def GetImage_fromHex(Data_hex,UserManualMemoryScan,OverRide_StartOffset):
    #Cut out subset of memory for specific datacc
    UserManualMemoryScan.OffsetEnd=hex((OverRide_StartOffset)+(UserManualMemoryScan.Width*UserManualMemoryScan.Height))

    #calculate offsets
    EndHex=hex((ConvertHex_to_decimal(UserManualMemoryScan.OffsetEnd))*2)
    ListOfHex_Subset=Return_SubSet_of_Memory(Data_hex,hex(OverRide_StartOffset*2),EndHex,0)

    #convert to grayscale
    #ListOfGrayScale_Subset=ConvertListHex_to_integers(ListOfHex_Subset)
    ListOfGrayScale_Subset=ConvertListHex_to_GrayScale_bitdepth(ListOfHex_Subset,2,255)

    #ReplaceHexStrings(ListOfGrayScale_Subset,DatFileInfos)
    #interpret area of memory as image
    (gray_image,DataVerify_image)=CreateGrayImage_from_memory(ListOfGrayScale_Subset,UserManualMemoryScan.Width,UserManualMemoryScan.Height)

    return (gray_image,DataVerify_image)

def UpdateSkimmer(data_hex,UserManualMemoryScan):
    #update tasks for skimming datamass
    #update display image, housekeeping for skimmer parameters

    #limit user params
    UserManualMemoryScan.Width=max(1, UserManualMemoryScan.Width)
    UserManualMemoryScan.Offset=max(0, UserManualMemoryScan.Offset)
    UserManualMemoryScan.Height=max(1, UserManualMemoryScan.Height)
    UserManualMemoryScan.ColourOffsetManual=max(0, UserManualMemoryScan.ColourOffsetManual)
    
    #update colour offsets from user
    if UserManualMemoryScan.ColourOffsetManual!=0:
        #user has started manual seeking of colour channels - cant escape once active!!
        
        #Green channel is current header - mem this out
        UserManualMemoryScan.RedChannelOffset_bytes=UserManualMemoryScan.Offset+UserManualMemoryScan.ColourOffsetManual
        UserManualMemoryScan.BlueChannelOffset_bytes= UserManualMemoryScan.Offset+(UserManualMemoryScan.ColourOffsetManual)*2

    #update debug info
    UserManualMemoryScan.PrintInfo()

    #cut image out of hex datamass
    (gray_image,DataVerify_image)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.Offset)

    #add header graphic if user has established record length discovery
    if UserManualMemoryScan.HeaderStartGuideImg is not None:
        #user has started header - lets turn on guiding graphics so can sync it up
        if gray_image.shape==UserManualMemoryScan.HeaderStartGuideImg.shape:
            DataVerify_image[:,:,0]=UserManualMemoryScan.HeaderStartGuideImg[:,:]

    #if user has established the colour channel combination function - enable the overlay graphics
    if UserManualMemoryScan.GreenChannel is not None:
        (green_image,dummyImg)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.GreenChannelOffset_bytes)
        UserManualMemoryScan.ColourOffsetManual=UserManualMemoryScan.Offset-UserManualMemoryScan.GreenChannelOffset_bytes
        print(UserManualMemoryScan.ColourOffsetManual)
        (blue_image,dummyImg)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.Offset+UserManualMemoryScan.ColourOffsetManual)
        DataVerify_image[:,:,0]=gray_image[:,:]
        DataVerify_image[:,:,1]=green_image[:,:]
        DataVerify_image[:,:,2]=blue_image[:,:]

    UserManualMemoryScan.DataVerify_image=DataVerify_image
    UserManualMemoryScan.gray_image=gray_image

    return


#preview capture
def PreviewCapture_RawHex(RawHex,UserManualMemoryScanA_ref,UserManualMemoryScanB_ref,ExtractionHelper,UserManualMemoryScan_ref,NumberToBreakAt):

    #make sure Pythons "pass by assignment" nonsense isnt screwing up the objects
    UserManualMemoryScanA=copy.deepcopy(UserManualMemoryScanA_ref)
    UserManualMemoryScanB=copy.deepcopy(UserManualMemoryScanB_ref)
    UserManualMemoryScan=copy.deepcopy(UserManualMemoryScan_ref)

    List_Images=[]
    Extraction=ExtractionHelper()
    ColourChannelOffset_HEx=0
    ColourChannelOffset_bytes=0
    if UserManualMemoryScan.ColourOffsetManual!=0:
        ColourChannelOffset_HEx=hex(UserManualMemoryScan.ColourOffsetManual*2)
        ColourChannelOffset_bytes=int(UserManualMemoryScan.ColourOffsetManual)

    Extraction.StartOffset_HEX=hex(UserManualMemoryScanA.Offset*2)
    Extraction.EndOfElementOffset_HEX=hex((ConvertHex_to_decimal(UserManualMemoryScanA.OffsetEnd))*2)
    Extraction.RecordSize_BYTES=(UserManualMemoryScanB.Offset-UserManualMemoryScanA.Offset)*2
    Extraction.ImageWidth=UserManualMemoryScanA.Width
    Extraction.ImageHeight=UserManualMemoryScanA.Height

    ExtractionRed_StartOffset_HEX=hex((ColourChannelOffset_bytes+UserManualMemoryScanA.Offset)*2)
    ExtractionRed_EndOffset_HEX=hex(((ConvertHex_to_decimal(UserManualMemoryScanA.OffsetEnd))*2)+(ColourChannelOffset_bytes*4))
    ExtractionBlue_StartOffset_HEX=hex((ColourChannelOffset_bytes+ColourChannelOffset_bytes+UserManualMemoryScanA.Offset)*2)
    ExtractionBlue_EndOffset_HEX=hex(((ConvertHex_to_decimal(UserManualMemoryScanA.OffsetEnd))*2)+(ColourChannelOffset_bytes*4))
    

    for counter, OffsetDecimal in enumerate(range (0,len(RawHex)-(Extraction.RecordSize_BYTES),(Extraction.RecordSize_BYTES))):
        
        #Cut out subset of memory for specific data
        ListOfHex_Subset=Return_SubSet_of_Memory(RawHex,((Extraction.StartOffset_HEX)),Extraction.EndOfElementOffset_HEX,OffsetDecimal)
        
        #convert to grayscale
        ListOfGrayScale_Subset=ConvertListHex_to_integers(ListOfHex_Subset)
        
        #interpret area of memory as image
        (green_image,DataVerify_image)=CreateGrayImage_from_memory(ListOfGrayScale_Subset,Extraction.ImageWidth,Extraction.ImageHeight)
        
        #if user has started colour channel process we have to add red and blue channels (seems to be GRB in the .dat files)
        if UserManualMemoryScan.ColourOffsetManual!=0:
            print("ColourOffsetManual image")
            #Next colour channel (RED)
            #Cut out subset of memory for specific data
            ListOfHex_Subset=Return_SubSet_of_Memory(RawHex,((ExtractionRed_StartOffset_HEX)),ExtractionRed_EndOffset_HEX,OffsetDecimal)

            #convert to grayscale
            ListOfGrayScale_Subset=ConvertListHex_to_integers(ListOfHex_Subset)

            #interpret area of memory as image
            (red_image,DataVerify_image)=CreateGrayImage_from_memory(ListOfGrayScale_Subset,Extraction.ImageWidth,Extraction.ImageHeight)

            
            #last colour channel (BLUE)
            ListOfHex_Subset=Return_SubSet_of_Memory(RawHex,((ExtractionBlue_StartOffset_HEX)),ExtractionBlue_EndOffset_HEX,OffsetDecimal)

            #convert to grayscale
            ListOfGrayScale_Subset=ConvertListHex_to_integers(ListOfHex_Subset)

            #interpret area of memory as image
            (blue_image,DataVerify_image)=CreateGrayImage_from_memory(ListOfGrayScale_Subset,Extraction.ImageWidth,Extraction.ImageHeight)

            DataVerify_image[:,:,0]=red_image[:,:]
            DataVerify_image[:,:,1]=green_image[:,:]
            DataVerify_image[:,:,2]=blue_image[:,:]
        
        #check if expecting grayscale image - user has not activate colour process
        if UserManualMemoryScan.ColourOffsetManual==0:
            DataVerify_image = cv2.cvtColor(DataVerify_image, cv2.COLOR_BGR2GRAY)

        #display image
        ImageViewer_Quickv2(DataVerify_image,0,False,False)

        #build up list of images
        List_Images.append(DataVerify_image)

        #inconvenient to user if preview is very long
        if NumberToBreakAt is not None:
            if counter>NumberToBreakAt:
                break

    return List_Images
        

def ImageViewer_Quickv2_UserControl(inputimage,pausetime_Secs=0,presskey=False,destroyWindow=True):
    ###handy quick function to view images with keypress escape andmore options
    if inputimage is None:
        return None
    CopyOfImage=cv2.resize(inputimage.copy(),(800,600))
    cv2.imshow("img", CopyOfImage); 
    UserRequest=""
    if presskey==True:
        while(1):
            cv2.imshow('img',CopyOfImage)
            k = cv2.waitKey(33)
            
            #return character from keyboard input ascii code
            if k != -1:#no input detected
                try:
                    UserRequest=(chr(k))
                    break
                except:
                    UserRequest=None
            else:
                continue
             

    if presskey==False:
        if cv2.waitKey(20) & 0xFF == 27:#need [waitkey] for GUI to update
                #for some reason
                pass
            
    if pausetime_Secs>0:
        time.sleep(pausetime_Secs)
    if destroyWindow==True: cv2.destroyAllWindows()

    return UserRequest

def compare(l1, l2):
    # here l1 and l2 must be lists
    if len(l1) != len(l2):
        return False
    set1 = set(l1)
    set2 = set(l2)
    if set1 == set2:
        return True
    else:
        return False

def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i+1)

def DecodeSNR_MultipleReads(AsciiCodeSNR_list,SNRposOffset):
            TempChars1=[]
            for Index, AsciiElement in enumerate(AsciiCodeSNR_list):
                if (Index+SNRposOffset)%6==0:
                    TempChars1.append(AsciiElement)
            #convert from hex to chars
            Joined=(''.join(TempChars1))
            byte_array = bytearray.fromhex(Joined)
            SNRCharstring=str((byte_array.decode()))#can specify "utf-8" here etc
            #try to remove null characters
            #TODO this should be more general and remove all weird characters
            if '\x00' in SNRCharstring:
                SNRCharstring=' '.join(SNRCharstring.split('\x00'))
            return SNRCharstring

def SNR_FIND(hexdata,PreknownPrefixData=None):

    
    FailedToFind=False
    #find serial number using known input string
    #user can provide the result from a previous search if conducting a batch

    if PreknownPrefixData is None:#user hasnt activated serial number string extract function

        StringToSearch=str(input("Type in an SNR result found in .dat viewer with correct capitalisation"))
        #StringToSearch="1CG"
        #StringToSearch="9FR"
        print("Scanning for string, ", StringToSearch)
        #convert ASCII to HEX
        HexToSearch=[hex(ord(c)) for c in StringToSearch]
        HexZeroChar="00"

        #format hex prefix base to match input raw hex format
        for counter, element in enumerate(HexToSearch):
            HexToSearch[counter]=element.replace('0x','')

        #dictionary to hold search string variations found in SNR read field
        SearchStrings=dict()
        #empty search lists
        SearchStrings[1]=[]
        SearchStrings[2]=[]
        SearchStrings[3]=[]

        #pad out according to padded variation of SNR read 
        for Element in HexToSearch:
            #so far found three variations on SNR result in .dat file - TODO will be lots more 
            TempElement1=[Element,HexZeroChar,HexZeroChar,HexZeroChar,HexZeroChar,HexZeroChar]
            TempElement2=[Element,HexZeroChar,Element,HexZeroChar,HexZeroChar,HexZeroChar]
            TempElement3=[Element,HexZeroChar,Element,HexZeroChar,Element,HexZeroChar]
            SearchStrings[1]=SearchStrings[1]+TempElement1
            SearchStrings[2]=SearchStrings[2]+TempElement2
            SearchStrings[3]=SearchStrings[3]+TempElement3
        
        #find position of string in the string of hex
        #To convert to hex remember to divide by two and round
        BytePosition=None
        PrefixingData=None
        SearchStringVariation=None
        for counter, SearchItem in enumerate(SearchStrings):
            SearchString = ''.join(SearchStrings[SearchItem])
            x = (hexdata).find(SearchString)
            if x>-1:
                BytePosition=x
                #before SNR read field at time of writing are common bookends per record
                PrefixingData=hexdata[x-400:x]
                #record format of snr result
                SearchStringVariation=SearchItem
                #print("SNR variation ", SearchStringVariation)
                #print("Found ", SearchString, " at ", hex(round(x/2)))
                #print(ascii(PrefixingData))
        
        #if user input string not found - return to main data scanning logic
        if BytePosition is None:
            print(StringToSearch , " string not found! ")
            FailedToFind=True
            return FailedToFind,None,None
    else:
        PrefixingData=PreknownPrefixData

    #get every instance of prefix in .dat file - should match number of SNR
    AllInstancesOfPrefixData=([(i, hexdata[i:i+2]) for i in findall(PrefixingData, hexdata)])
    #print("Instances of SNR found: ", len(AllInstancesOfPrefixData))

    SNR_Results_per_record=dict()
    #roll through instances and calculate start position of each SNR read
    for COunter, PrefixInstance in enumerate(AllInstancesOfPrefixData):
        #get subset of data where SNR should be
        #ChunkSizeBytes=len(''.join(SearchStrings["1"]))
        ChunkSizeBytes=DatFileInfos.SNR_Result_Chunk#arbitrary size of SNR chunk
        AsciiCodeSNR=(hexdata[PrefixInstance[0]+len(PrefixingData):PrefixInstance[0]+len(PrefixingData)+ChunkSizeBytes])
        #convert to list of two bytes for ascii codes
        AsciiCodeSNR_list=[AsciiCodeSNR[i:i+2] for i in range(0, len(AsciiCodeSNR), 2)]
        #convert to char list
        

        SNRCharstring=DecodeSNR_MultipleReads(AsciiCodeSNR_list,0)

        SNR_Results_per_record[COunter+1]=SNRCharstring

    for dictelem in SNR_Results_per_record:
        #pass
        print(dictelem,SNR_Results_per_record[dictelem] )
    if PrefixingData is None:
        print("Couldnt find prefix data for SNR search")
        FailedToFind=True
    if SNR_Results_per_record is None:
        print("Couldnt find prefix data for SNR search")
        FailedToFind=True
    if len(SNR_Results_per_record)==0:
        print("Couldnt find prefix data for SNR search")
        FailedToFind=True

    return not FailedToFind, SNR_Results_per_record,PrefixingData

def GetEveryNthChar(InputString, StepSize):
    #get every 1st/2nd etc char of a string - python slicing [::n] doesnt seem to work reliably
    SteppedString=""
    for I, Charac in enumerate(InputString):
        if I%StepSize==0:
            SteppedString=SteppedString+Charac
            print(SteppedString)
    return SteppedString

def RipAllImagePerDat(OutputFolder,InputFiles_cleaned,RawHex,UserManualMemoryScan,UserManualMemoryScanA,UserManualMemoryScanB,ExtractionHelper):
    
    #clean output folder
    print ("Cleaning output folder :", OutputFolder)
    result=_3DVisLabLib.yesno("Continue to delete all contents (including nested) of folder?")
    if result==True: _3DVisLabLib.DeleteFiles_RecreateFolder(OutputFolder)
    

    for counter2, FileName in enumerate(InputFiles_cleaned): 
        if not ((".dat") in FileName.lower()):
            print("Skipping over file ", FileName)
            continue
        #get raw hex
        data_hex=Load_Hex_File(FileName)
        #get list of images from single dat
        List_of_images=PreviewCapture_RawHex(data_hex,UserManualMemoryScanA,UserManualMemoryScanB,ExtractionHelper,UserManualMemoryScan,None)

        #initialise SNR boolean which determines if SNR string will be used as image save filenames
        SNRSuccess=False
        SNR_Results_per_record=dict()#placeholder

        if UserManualMemoryScan.SNR_extractMode==SNR_string_extract.Manual:
        #if serial numbers have been MANUALLY activated by user - extract them as well
            if UserManualMemoryScan.SNR_PrefixData is not None:
                #previous operation by user has found serial number read
                SNRSuccess,SNR_Results_per_record,PrefixingData=SNR_FIND(data_hex,UserManualMemoryScan.SNR_PrefixData)
                if SNRSuccess==False:
                    print("Could not find serial number string!!")
                    raise Exception("Could not parse SNR")
                else:
                    if len(List_of_images)!=len(SNR_Results_per_record):
                        raise Exception("User has activated SNR string read, but array length does not match records; "," List_of_images ",len(List_of_images)," SNR_Results_per_record ",len(SNR_Results_per_record) )

        if UserManualMemoryScan.SNR_extractMode==SNR_string_extract.Automatic:
            #user has activated automatic extraction of serial number reads
            images = DatScraper_tool.ImageExtractor(FileName)
            for Index, Item in enumerate(images.notes):
                Tempsnr=str(Item.snr)
                if Tempsnr is None:
                    SNR_Results_per_record[Index+1]=(OperationCodes.ERROR)
                    raise Exception("Could not parse SNR")
                elif Tempsnr =="":
                    SNR_Results_per_record[Index+1]=(OperationCodes.ERROR)
                    raise Exception("Could not parse SNR")
                else:
                    SNR_Results_per_record[Index+1]="[" + (str(Item.snr)) + "]" + str(UserManualMemoryScan.DatFileDataType)

            SNRSuccess=True



        for counter1, imageinstance in enumerate(List_of_images):
            
            snrRead=""
            if SNRSuccess==True:
                snrRead=(SNR_Results_per_record[counter1+1])

            Delimit_FilePath_slash=FileName.split('\\')
            Delimit_FilePath_Dot=Delimit_FilePath_slash[len(Delimit_FilePath_slash)-1].split('.')
            DatFileName=Delimit_FilePath_Dot[0]
            print(DatFileName)
            Savestring=OutputFolder + "\\"  + str(snrRead) + "_" + str(DatFileName) + " record " + str(counter1)+ " file " + str(counter2)+  ".jpg"
            print("saving image to ", Savestring)
            try:
                cv2.imwrite(Savestring,imageinstance.copy())
            except:
                pass


def RipOneImagePerDat(InputFiles_cleaned,UserManualMemoryScan):
    for counter2, FileName in enumerate(InputFiles_cleaned): 
        if not ((".dat") in FileName.lower()):
            print("Skipping over file ", FileName)
            continue
        #get raw hex instead of grayscales
        data_hex=Load_Hex_File(FileName)

        #get current offset
        (gray_image,DataVerify_image)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.Offset)

        #if user has started colour channel process
        if UserManualMemoryScan.ColourOffsetManual!=0:

                # #get GREEN channel
                # (green_image,DataVerify_image)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.Offset)
                # #red channel
                # (red_image,DataVerify_image)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.Offset+UserManualMemoryScan.ColourOffsetManual)
                # #blue channel
                # (blue_image,DataVerify_image)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.Offset+UserManualMemoryScan.ColourOffsetManual+UserManualMemoryScan.ColourOffsetManual)
                
                
                #get GREEN channel
                (green_image,DataVerify_image)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.Offset)
                #red channel
                (red_image,DataVerify_image)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.RedChannelOffset_bytes)
                #blue channel
                (blue_image,DataVerify_image)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.BlueChannelOffset_bytes)
                
                DataVerify_image[:,:,0]=red_image[:,:]
                DataVerify_image[:,:,1]=green_image[:,:]
                DataVerify_image[:,:,2]=blue_image[:,:]


                # (green_image,dummyImg)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.GreenChannelOffset_bytes)
                # UserManualMemoryScan.ColourOffsetManual=UserManualMemoryScan.Offset-UserManualMemoryScan.GreenChannelOffset_bytes
                # print(UserManualMemoryScan.ColourOffsetManual)
                # (blue_image,dummyImg)=GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.Offset+UserManualMemoryScan.ColourOffsetManual)
                # DataVerify_image[:,:,0]=gray_image[:,:]
                # DataVerify_image[:,:,1]=green_image[:,:]
                # DataVerify_image[:,:,2]=blue_image[:,:]

        #check if expecting grayscale image - user has not activate colour process
        if UserManualMemoryScan.ColourOffsetManual==0:
            DataVerify_image = cv2.cvtColor(DataVerify_image, cv2.COLOR_BGR2GRAY)

        ImageViewer_Quickv2_UserControl(DataVerify_image,0,False,False)

        #save out thumbnail
        Savestring=FileName.lower().replace(".dat",".jpg")
        print("saving image to ", Savestring)
        #CopyOfImage=cv2.resize(DataVerify_image.copy(),(1000,800))
        cv2.imwrite(Savestring,DataVerify_image)



class ExtractionHelper():
    def __init__(self):
        self.StartOffset_HEX=None#hex(UserManualMemoryScanA.Offset)#start of extraction - also start of first element we want extracted
        self.EndOfElementOffset_HEX=None#UserManualMemoryScanA.OffsetEnd# end of memory chunk defining element we want extracted
        self.RecordSize_BYTES=None#(UserManualMemoryScanB.Offset-UserManualMemoryScanA.Offset)#bytes from start of first element to start of equivalent
        self.ImageWidth=None#UserManualMemoryScanA.Width#should be same for both records - TODO flag up an error here if user has changed 
        self.ImageHeight=None#UserManualMemoryScanA.Height#should be same for both records - TODO flag up an error here if user has changed 
        #element in next record - in theory the length of each record

   

def FindColourChannels(InputHex,UserManualMemoryScan):
    #grab current position and add some memory
    #Cut out subset of memory for specific data

        #add guesstimate offset for next channel
        Guesstimate_bytes=600

        UserManualMemoryScan.OffsetEnd=hex((UserManualMemoryScan.Offset)+(UserManualMemoryScan.Width*Guesstimate_bytes))

        #calculate offsets
        EndHex=hex(((ConvertHex_to_decimal(UserManualMemoryScan.OffsetEnd))*2)+Guesstimate_bytes)
        ListOfHex_Subset=Return_SubSet_of_Memory(InputHex,hex(UserManualMemoryScan.Offset*2),EndHex,0)

        #convert to grayscale
        ListOfGrayScale_Subset=ConvertListHex_to_integers(ListOfHex_Subset)

        #interpret area of memory as image
        (gray_image,DataVerify_image)=CreateGrayImage_from_memory(ListOfGrayScale_Subset,UserManualMemoryScan.Width,Guesstimate_bytes)#,UserManualMemoryScan.Height


        #get edge filter
        GradientImage_laplacian=cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        #display image, wait for user input keypress to determine manual scan parameters
        UserRequest=ImageViewer_Quickv2_UserControl(GradientImage_laplacian,0,True,False)
    #print out image to check

    #create edge or gradient image

    #cross correlate

    #find peak of correlation - does it make sense

    #create offsets for next two channels
    