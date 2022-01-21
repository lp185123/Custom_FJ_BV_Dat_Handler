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
            raise Exception("CloudOCR perform OCR error - ImageObject parameter WIP!! Google API only supports files at time of writing")
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

        #replace all non alphanumeric characters
        OutString3 = re.sub(r'[^a-zA-Z0-9]', '', OutString3)
        #ListOCR_Reads.append(OutString3)

        #print(OutTexts)
        #print('bounds: {}'.format(','.join(vertices)))

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(response.error.message))
        return OutString3




