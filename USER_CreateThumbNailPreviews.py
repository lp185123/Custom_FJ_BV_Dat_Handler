
import cv2
import numpy as np
from pathlib import Path
import _3DVisLabLib
from copy import deepcopy
import BV_DatReader_Lib
import DatScraper_tool
import gc
gc.disable()

def main(S39_only=False):
    
    #get parameters object will be loaded with user selection
    GetParametersFromUser=BV_DatReader_Lib.UserInputParameters()
    if _3DVisLabLib.yesno("Delete output folder" + GetParametersFromUser.OutputFilePath):
        _3DVisLabLib.DeleteFiles_RecreateFolder(GetParametersFromUser.OutputFilePath)

    if S39_only==False:
        GetParametersFromUser.UserPopulateParameters()
    else:
        # #load parameter with S39 specific data
        # GetParametersFromUser.AutomaticMode=True
        # GetParametersFromUser.BlockTypeWave='None'
        # GetParametersFromUser.BlockType_ImageFormat='SRU SNR image1'
        # GetParametersFromUser.FirstImageOnly=False
        # GetParametersFromUser.GetRGBImage=False
        # GetParametersFromUser.GetSNR=True
        # GetParametersFromUser.InputFilePath=GetParametersFromUser.InputFilePath
        # GetParametersFromUser.OutputFilePath=GetParametersFromUser.OutputFilePath

        
        #load parameter with S39 specific data
        GetParametersFromUser.AutomaticMode=True
        GetParametersFromUser.BlockTypeWave='C'
        GetParametersFromUser.BlockType_ImageFormat='SRU MM1 side image'
        GetParametersFromUser.FirstImageOnly=True
        GetParametersFromUser.GetRGBImage=True
        GetParametersFromUser.GetSNR=False
        Result = input("Please enter folder for analysis:")
        GetParametersFromUser.InputFilePath=Result
        GetParametersFromUser.OutputFilePath=GetParametersFromUser.OutputFilePath


        
    #placeholder until dynamic folder discovery
    class GeneralData():
        InputFolder="Input\\"
        OutputFolder="Output\\"

    def ManualMode(GetParametersFromUser):

        #prototype code to manually find images in the datamass
        FilePath=GetParametersFromUser.InputFilePath

        #get file path of .py file
        FilePath_of_script=str(Path(__file__).resolve().parent) + "\\"
        #FilePath=FilePath.replace("\\","/")
        print("Folder for analysis: ", FilePath)

        #get all files in input folder
        InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(FilePath)
        print("Files found in folder: ", InputFiles)

        #clean files
        InputFiles_cleaned=[]
        for elem in InputFiles:
            if ((".dat") in elem.lower()):
                InputFiles_cleaned.append(elem)
            
                       
        #load as hex
        data_hex=BV_DatReader_Lib.Load_Hex_File(InputFiles_cleaned[0])
        
        #let user select area of memory
        #create instance of helper class
        UserManualMemoryScan=BV_DatReader_Lib.UserManualMemoryScan_helper()
        #create reader heads that operator will use to define area of memory of interest and memory between records
        UserManualMemoryScanA=BV_DatReader_Lib.UserManualMemoryScan_helper()
        UserManualMemoryScanB=BV_DatReader_Lib.UserManualMemoryScan_helper()

        #user control loop
        while(1):

                #update display image, housekeeping for skimmer parameters
                BV_DatReader_Lib.UpdateSkimmer(data_hex,UserManualMemoryScan)

                #display image, wait for user input keypress to determine manual scan parameters
                User_keypress=BV_DatReader_Lib.ImageViewer_Quickv2_UserControl(UserManualMemoryScan.DataVerify_image,0,True,False)

                #read in request from user
                UserRequest=None
                for element in BV_DatReader_Lib.UserOperationStrings:
                    if element.value==User_keypress:
                        UserRequest=element
                    if User_keypress in ["0","1","2","3","4","5","6","7","8","9"]:
                        UserRequest=int(User_keypress)

                #handle returned user keypress
                if UserRequest in [0,1,2,3,4,5,6,7,8,9]:
                    UserManualMemoryScan.Multiplier=UserRequest
                    print("user requested multiplication of ", UserManualMemoryScan.Multiplier)

                if UserRequest==BV_DatReader_Lib.UserOperationStrings.UP:
                    if UserManualMemoryScan.Multiplier==0:
                        UserManualMemoryScan.Offset=int(UserManualMemoryScan.Offset+(UserManualMemoryScan.Width))
                    else:
                        UserManualMemoryScan.Offset=int(UserManualMemoryScan.Offset+(UserManualMemoryScan.Width *(UserManualMemoryScan.Multiplier*UserManualMemoryScan.Multiplier)))
                
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.DOWN:
                    if UserManualMemoryScan.Multiplier==0:
                        UserManualMemoryScan.Offset=int(UserManualMemoryScan.Offset-(UserManualMemoryScan.Width))
                    else:
                        UserManualMemoryScan.Offset=int(UserManualMemoryScan.Offset-(UserManualMemoryScan.Width *(UserManualMemoryScan.Multiplier*UserManualMemoryScan.Multiplier)))

                if UserRequest==BV_DatReader_Lib.UserOperationStrings.LEFT2:
                    UserManualMemoryScan.Offset=int(UserManualMemoryScan.Offset+(1 *UserManualMemoryScan.Multiplier))
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.RIGHT2:
                    UserManualMemoryScan.Offset=int(UserManualMemoryScan.Offset-(1 *UserManualMemoryScan.Multiplier))

                if UserRequest==BV_DatReader_Lib.UserOperationStrings.LEFT:
                    UserManualMemoryScan.Width=int(UserManualMemoryScan.Width-(1*UserManualMemoryScan.Multiplier))
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.RIGHT:
                    UserManualMemoryScan.Width=int(UserManualMemoryScan.Width+(1*UserManualMemoryScan.Multiplier))

                if UserRequest==BV_DatReader_Lib.UserOperationStrings.UP2:
                    UserManualMemoryScan.Height=int(UserManualMemoryScan.Height-(1*(UserManualMemoryScan.Multiplier^2)))
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.DOWN2:
                    UserManualMemoryScan.Height=int(UserManualMemoryScan.Height+(1*(UserManualMemoryScan.Multiplier^2)))

                if UserRequest==BV_DatReader_Lib.UserOperationStrings.RANDOM:
                    if (UserManualMemoryScan.CycleThroughKnownDims>len(BV_DatReader_Lib.DatFileInfos.KnownWidths_Heights)-1):
                        UserManualMemoryScan.CycleThroughKnownDims=0
                    UserManualMemoryScan.Width=BV_DatReader_Lib.DatFileInfos.KnownWidths_Heights[UserManualMemoryScan.CycleThroughKnownDims][0]
                    UserManualMemoryScan.CycleThroughKnownDims=UserManualMemoryScan.CycleThroughKnownDims+1

                if UserRequest==BV_DatReader_Lib.UserOperationStrings.A:
                    UserManualMemoryScan.HeaderStartGuideImg=UserManualMemoryScan.gray_image.copy()
                    UserManualMemoryScan.HeaderStartGuideImg=np.array((UserManualMemoryScan.gray_image.copy()*0.7))
                    UserManualMemoryScanA=deepcopy(UserManualMemoryScan)
                    #auto boost skimming
                    UserManualMemoryScan.Multiplier=9
                    print(":::::::::::Reader head set START:::::::::::")

                if UserRequest==BV_DatReader_Lib.UserOperationStrings.B:
                    UserManualMemoryScanB=deepcopy(UserManualMemoryScan)
                    print(":::::::::::Reader head set END:::::::::::")
                    BV_DatReader_Lib.PreviewCapture_RawHex(data_hex,UserManualMemoryScanA,UserManualMemoryScanB,BV_DatReader_Lib.ExtractionHelper,UserManualMemoryScan,10)
                
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.COLOURS:
                    BV_DatReader_Lib.FindColourChannels(data_hex,UserManualMemoryScan)
                
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.RED and UserManualMemoryScan.ColourOffsetManual==0:
                    print("Green channel (1st colour channel)")
                    UserManualMemoryScan.GreenChannel=UserManualMemoryScan.gray_image.copy()
                    UserManualMemoryScan.GreenChannelOffset_bytes=UserManualMemoryScan.Offset
                    UserManualMemoryScan.ColourChannelOffset=0
                
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.RED and UserManualMemoryScan.ColourOffsetManual!=0:
                    print("Finishing Colour channel - colour channel offset staged as ",UserManualMemoryScan.ColourOffsetManual , " bytes" )
                    print("moving back to start of colour channel search process")
                    UserManualMemoryScan.Offset=UserManualMemoryScan.GreenChannelOffset_bytes
                    UserManualMemoryScan.GreenChannel=None#TODO this is getting a bit janky - need a better way to organise capture modes
                    #UserManualMemoryScan.GreenChannel=gray_image.copy()
                    UserManualMemoryScan.GreenChannelOffset_bytes=UserManualMemoryScan.Offset
                    UserManualMemoryScan.ColourChannelOffset=0

                if UserRequest==BV_DatReader_Lib.UserOperationStrings.GREEN:#"GREEN":
                    print("Red channel(2nd colour channel)")
                    UserManualMemoryScan.RedChannel=UserManualMemoryScan.gray_image.copy()
                    UserManualMemoryScan.RedChannelOffset_bytes=UserManualMemoryScan.Offset
                    if UserManualMemoryScan.GreenChannel is not None:
                        #auto calculate next colour channel position
                        UserManualMemoryScan.ColourChannelOffset=UserManualMemoryScan.RedChannelOffset_bytes-UserManualMemoryScan.GreenChannelOffset_bytes
                        AutoBlueOffset=UserManualMemoryScan.RedChannelOffset_bytes-UserManualMemoryScan.GreenChannelOffset_bytes
                        UserManualMemoryScan.Offset= UserManualMemoryScan.Offset+AutoBlueOffset
                        (UserManualMemoryScan.gray_image,UserManualMemoryScan.DataVerify_image)=BV_DatReader_Lib.GetImage_fromHex(data_hex,UserManualMemoryScan,UserManualMemoryScan.Offset)
                        print("jumped ", AutoBlueOffset, " bytes to anticipate next colour channel- forced on SELECT BLUE, reloaded image")
                        print(AutoBlueOffset)
                        UserRequest="BLUE"
                
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.BLUE:#"BLUE":
                    print("blue channel (3rd colour channel)")
                    UserManualMemoryScan.BlueChannel=UserManualMemoryScan.gray_image.copy()
                    UserManualMemoryScan.BlueChannelOffset_bytes=UserManualMemoryScan.Offset
                    if UserManualMemoryScan.RedChannel is not None:
                        #auto calculate next colour channel position
                        AutoBlueOffset=UserManualMemoryScan.BlueChannelOffset_bytes-UserManualMemoryScan.RedChannelOffset_bytes
                        print(AutoBlueOffset)
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.SHIFT:#"EXTRACT":
                    data_hex="0" + data_hex
                    #TODO very bad work test!!!

                if UserRequest==BV_DatReader_Lib.UserOperationStrings.EXTRACT:#"EXTRACT":
                    if UserManualMemoryScanB.Offset ==0:#user hasnt used header functions so single image only
                        BV_DatReader_Lib.RipOneImagePerDat(InputFiles_cleaned,UserManualMemoryScan)
                    else:
                        #user has used header function so request all images from the .dat file with user dictated record length
                        #get filepath of script
                        FilePath_of_script=str(Path(__file__).resolve().parent) + "/" + GeneralData.OutputFolder
                        #loop through all input files and extract images
                        BV_DatReader_Lib.RipAllImagePerDat(FilePath_of_script,InputFiles_cleaned,data_hex,UserManualMemoryScan,UserManualMemoryScanA,UserManualMemoryScanB,BV_DatReader_Lib.ExtractionHelper)

                if UserRequest==BV_DatReader_Lib.UserOperationStrings.COLOURUP:#"COLOURUP":
                    UserManualMemoryScan.ColourOffsetManual=UserManualMemoryScan.ColourOffsetManual+(UserManualMemoryScan.Multiplier*UserManualMemoryScan.Multiplier*UserManualMemoryScan.Multiplier)
                    print(UserManualMemoryScan.ColourOffsetManual)
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.COLOURDOWN:#"COLOURDOWN":
                    UserManualMemoryScan.ColourOffsetManual=UserManualMemoryScan.ColourOffsetManual-(UserManualMemoryScan.Multiplier*UserManualMemoryScan.Multiplier*UserManualMemoryScan.Multiplier)
                    print(UserManualMemoryScan.ColourOffsetManual)
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.SNRFIND:#"SNRFIND":
                    #manual SNR read method
                    UserManualMemoryScan.SNR_extractMode=BV_DatReader_Lib.SNR_string_extract.Manual
                    Success,SNR_Results_per_record,PrefixingData=BV_DatReader_Lib.SNR_FIND(data_hex,None)
                    if Success==False:
                        pass
                    else:
                        UserManualMemoryScan.SNR_PrefixData=PrefixingData

                if UserRequest==BV_DatReader_Lib.UserOperationStrings.GetALLSNR:
                    #ask user for an SNR string found in collection, then get S39 image for each record in each .dat
                    #file and align with SNR string found

                    #get all images from first .dat file in input folder(s) (nested)
                    images = DatScraper_tool.ImageExtractor(InputFiles_cleaned[0])
                    #filter for SNR images
                    filteredImages = images.filter("SRU SNR image1","None")
                    UserManualMemoryScan.DatFileDataType="SRU SNR image1" + " " + "None"
                    if filteredImages is None:
                        print("No SNR images found")
                        continue
                    if len(filteredImages)<1:
                        print("No images found")
                        continue

                    #Just duplicate manual methods - TODO tidy this up
                    #set extract mode for extraction function to select correct SNR option
                    UserManualMemoryScan.SNR_extractMode=BV_DatReader_Lib.SNR_string_extract.Automatic
                    #reset memory skimmer helpers
                    UserManualMemoryScanA=BV_DatReader_Lib.UserManualMemoryScan_helper()
                    UserManualMemoryScanB=BV_DatReader_Lib.UserManualMemoryScan_helper()

                    #set manual skimmer position
                    UserManualMemoryScan.Offset=int(filteredImages[1].offsetStart)
                    UserManualMemoryScan.OffsetEnd=hex(int(filteredImages[1].offsetEnd))
                    UserManualMemoryScan.Width=int(filteredImages[1].width)
                    UserManualMemoryScan.Height=int(filteredImages[1].height)#if mm8 image we need to double height as is 4 bytes image depth
                    #set skimmer "A start" - warning this is a bit janky
                    UserManualMemoryScanA=deepcopy(UserManualMemoryScan)
                    #duplicate manual process with housekeeping between actions
                    BV_DatReader_Lib.UpdateSkimmer(data_hex,UserManualMemoryScan)

                    # skimmer "B end" - still janky way of doing it
                    UserManualMemoryScan.Offset=int(filteredImages[2].offsetStart)
                    UserManualMemoryScanB=deepcopy(UserManualMemoryScan)
                    #duplicate manual process with housekeeping between actions
                    BV_DatReader_Lib.UpdateSkimmer(data_hex,UserManualMemoryScan)
                    
                    #preview for user to input characters to find
                    BV_DatReader_Lib.PreviewCapture_RawHex(data_hex,UserManualMemoryScanA,UserManualMemoryScanB,BV_DatReader_Lib.ExtractionHelper,UserManualMemoryScan,1)

                    #preview for user - to check if any drift for sequential images
                    BV_DatReader_Lib.PreviewCapture_RawHex(data_hex,UserManualMemoryScanA,UserManualMemoryScanB,BV_DatReader_Lib.ExtractionHelper,UserManualMemoryScan,10)
                    
                if UserRequest==BV_DatReader_Lib.UserOperationStrings.Test:#"TEST":
                    images = DatScraper_tool.ImageExtractor(InputFiles_cleaned[0],True)# Use this to only get images of particular wavelengths
                    print(images)
                    #filteredImages = images.filter("SRU MM8 image","C")
                    filteredImages = images.filter("SRU SNR image1","None")
                    if filteredImages is None:
                        print("No images found")
                        continue

                    if len(filteredImages)<1:
                        print("No images found")
                        continue

                    print(filteredImages)
                    UserManualMemoryScan.Offset=int(filteredImages[1].offsetStart)
                    UserManualMemoryScan.Width=int(filteredImages[1].width)
                    UserManualMemoryScan.Height=int(filteredImages[1].height)#if mm8 image we need to double height as is 4 bytes image depth


        cv2.destroyAllWindows()

    if GetParametersFromUser.AutomaticMode==False:
        ManualMode(GetParametersFromUser)
    else:
        BV_DatReader_Lib.AutomaticExtraction(GetParametersFromUser)



if __name__ == "__main__":
    #entry point
    #try:
    
    print("WARNING! MM8 images WIP - still issue with image depth size WIP")
    if _3DVisLabLib.yesno("Get SR thumbnails only?"):main(True)
    else:
        main(False)
    #except Exception as e:
    #    ##note: cleaning up after exception should be to set os.chdir(anything else) or it will lock the folder
    #    print(e)
    #    # printing stack trace
    #    print("Press any key to continue")
    #    os.system('pause')