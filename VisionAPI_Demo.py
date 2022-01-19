from google.cloud import vision
import io
import os
import _3DVisLabLib
import re

class CloudOCR():
    """Class to authenticate cloud service and perform OCR services."""
    def __init__(self):
        #Authenticate user - see notes 
        self.client = vision.ImageAnnotatorClient()
        #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Working\FindIMage_In_Dat\VisionAPIDemo\ocrtrial-338212-a4732d2e2a9c.json"
        #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Working\FindIMage_In_Dat\VisionAPIDemo\ServiceAccountToken.json"
        #set GOOGLE_APPLICATION_CREDENTIALS="C:\Working\FindIMage_In_Dat\VisionAPIDemo\ServiceAccountToken.json
    def PerformOCR(self,FilePath,ImageObject):
        """Pass in Filepath or Imageobject - currently imageobject is not tested"""
        if (ImageObject is None) and (FilePath is None):
            raise Exception("CloudOCR perform OCR error - please provide a filepath or an image object")

        if (ImageObject is not None) and (FilePath is not None):
            print("WARNING CloudOCR perform OCR, filepath and Image object provided - please use exclusive option")

        if FilePath is not None:
            print("Cloud OCR - loading file",FilePath)
            with io.open(FilePath, 'rb') as image_file:
                content = image_file.read()
        
        if ImageObject is not None:
            raise Exception("CloudOCR perform OCR error - ImageObject parameter WIP!!")
            content = ImageObject
        

        image = vision.Image(content=content)

        response = self.client.text_detection(image=image)
        texts = response.text_annotations
        OutTexts=[]
        StringOutput=""
        #TODO looks like first line is the entire read and the rest is the breakdown
        for text in texts:
            #print('\n"{}"'.format(text.description))
            OutTexts.append(text.description)
            StringOutput=StringOutput+str(text.description)
            vertices = (['({},{})'.format(vertex.x, vertex.y)for vertex in text.bounding_poly.vertices])

        joined_ReadString = " ".join(OutTexts)

        OutString3=""
        for Index, Line in enumerate(OutTexts):
            if Index>0:
                OutString3=OutString3+Line

        #print(OutTexts)
        #print('bounds: {}'.format(','.join(vertices)))

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(response.error.message))
        return OutString3


#instancing class will initialise Cloud service and attempt to authenticate agent
GoogleCloudOCR=CloudOCR()
Default_Input=str( r"C:\Working\FindIMage_In_Dat\OutputTestSNR\Brazil")
Default_Output=str( r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\CloudOCR")
#get input folder of images from extract S39 function - can be single images with embedded template answer or
#collimated images with answer text files with same filename but different extensions
Result_Input = input("Please enter images input folder - press enter to use default" + "   " + Default_Input)
if len(Result_Input)<2:
    Result_Input=Default_Input
Result_Output = input("Please enter answers output folder - press enter to use default" + "   " + Default_Output)
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
for Img_to_Ocr in ListAllImages:
    print("Uploading to Cloud",Img_to_Ocr)
    OCR_Result=GoogleCloudOCR.PerformOCR(Img_to_Ocr,None)
    # remove non alphanuemeric characters
    OCR_Result = re.sub(r'[^a-zA-Z0-9]', '', OCR_Result)
    #OCR_Result=OCR_Result.encode("utf-8")#non latin chars such as hindi can break code
    ListOCR_Reads.append(OCR_Result)
    #create matching answer text file
    DelimitedImgFile=Img_to_Ocr.split("\\")
    DelimitedImgFile_LastElem=DelimitedImgFile[-1]
    ReplacedExtension=DelimitedImgFile_LastElem.replace(".jpg",".txt")
    AnswerFile=Result_Output+ "\\" + ReplacedExtension
    print("Saving answer to",AnswerFile)
    with open(AnswerFile, 'w') as f:
        f.write(OCR_Result)


    #open text file
    #text_file = open(AnswerFile)
    #write string to file
    #n = text_file.write(OCR_Result)
    #close file
    #text_file.close()

    

#joined_ReadString=GoogleCloudOCR.PerformOCR(r"C:\Working\FindIMage_In_Dat\ForResearch\OCR_Testdata\Brazil\Collimated_NoProcessing\0.jpg",None)
#print(joined_ReadString)