import VisionAPI_Demo
import _3DVisLabLib


#instancing class will initialise Cloud service and attempt to authenticate agent
GoogleCloudOCR=VisionAPI_Demo.CloudOCR()
Default_Input=str( r"C:\Working\FindIMage_In_Dat\OutputTestSNR\CollimatedOutput")
Default_Output=str( r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\CloudOCR")
#get input folder of images from extract S39 function - can be single images with embedded template answer or
#collimated images with answer text files with same filename but different extensions
Result_Input = input("Please enter images input folder - press enter to use default" + "   " + Default_Input)
if len(Result_Input)<2:
    Result_Input=Default_Input
Result_Output = input("Please enter Cloud OCR answers output folder - press enter to use default" + "   " + Default_Output)
if len(Result_Output)<2:
    Result_Output=Default_Output

#prompt user to check filepaths are OK for deletion
print("Please check output folders can be deleted:\n",Result_Output)
Response=_3DVisLabLib.yesno("Continue?")
if Response==False:
    raise Exception("User declined to delete folders - process terminated")

#delete output folder
_3DVisLabLib.DeleteFiles_RecreateFolder(Result_Output)

#get list of files in folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(Result_Input)
#filter for .jpg images
ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)

#empty folder warning
if len(ListAllImages)==0:
    raise Exception("CloudOCR perform OCR error - no images found in folder\n",Result_Input)

#loop through images, get OCR read from cloud service and save file to output folder
ListOCR_Reads=[]
for Index, Img_to_Ocr in enumerate(ListAllImages):
    print("File",str(Index),"/",str(len(ListAllImages)), "_Uploading to Cloud OCR",Img_to_Ocr)
    OCR_Result,CharsVConfidence=GoogleCloudOCR.PerformOCR(Img_to_Ocr,None)
    #create matching answer text file
    DelimitedImgFile=Img_to_Ocr.split("\\")
    DelimitedImgFile_LastElem=DelimitedImgFile[-1]
    ReplacedExtension=DelimitedImgFile_LastElem.replace(".jpg",".txt")
    AnswerFile=Result_Output+ "\\" + ReplacedExtension
    print("Saving answer to",AnswerFile)
    with open(AnswerFile, 'w',encoding="utf-8") as f:#encoding="utf-8" for unicode which is handled as standard in python 3
        f.write(OCR_Result)