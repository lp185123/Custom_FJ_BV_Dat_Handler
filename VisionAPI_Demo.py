import google
from google.cloud import vision
import google.auth.credentials
import io
import re
import os
print(google.api_core.__version__)
print(google.auth.credentials.__file__)
class CloudOCR():
    """Class to authenticate cloud service and perform OCR services."""
    def __init__(self):
        #Authenticate user - see notes 
        #https://www.youtube.com/watch?v=_24h-FQODqo good guidance - have to PIP install the google thing very specifically
        #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Working\FindIMage_In_Dat\VisionAPIDemo\ocrtrial-338212-a4732d2e2a9c.json"
        #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Working\FindIMage_In_Dat\VisionAPIDemo\ServiceAccountToken.json"
        self.client = vision.ImageAnnotatorClient()
        #set GOOGLE_APPLICATION_CREDENTIALS="C:\Working\FindIMage_In_Dat\VisionAPIDemo\ServiceAccountToken.json
        #pip install --upgrade google-analytics-data
        #pip install --upgrade google-auth
        print("Google Vision API initialised")# - this costs approx 3$ an hour (1$ per 1000 images) - same approx price as 125cc motorbike fuel (@60mph)")


    
    def PerformOCR(self,FilePath,ImageObject):
        """Pass in Filepath or Imageobject - currently imageobject is not tested"""
        if (ImageObject is None) and (FilePath is None):
            raise Exception("CloudOCR perform OCR error - please provide a filepath or an image object")

        if (ImageObject is not None) and (FilePath is not None):
            print("WARNING CloudOCR perform OCR, filepath and Image object provided - please use exclusive option")

        if FilePath is not None:
            #print("Cloud OCR - loading file",FilePath)
            with io.open(FilePath, 'rb') as image_file:
                content = image_file.read()
        
        if ImageObject is not None:
            raise Exception("CloudOCR perform OCR error - ImageObject parameter WIP!! Google API only supports files at time of writing")
            content = ImageObject
        
        #fakedict=dict()
        #fakedict["p"]="0.1"
        #fakedict["l"]="0.1"
        #fakedict["o"]="0.1"
        #fakedict["p"]="0.1"

        #return "plop",fakedict

        image = vision.Image(content=content)

        #response = self.client.text_detection(image=image,image_context={"language_hints": ["bn","en"]})
        #image_context={"language_hints": ["bn"]} #https://cloud.google.com/vision/docs/languages more langauge hints
        response = self.client.document_text_detection(image=image)#,image_context={"language_hints": ["bn"]})
        #response = self.client.text_detection(image=image,image_context={"language_hints": ["bn"]})
        #print("Google VIsion API using [document_text_detection], can swap mode to [text_detection] to improve results")
        UnicodeListSymbols=[]
        UnicodeCharVConfidence=dict()
        SymbolCounter=0
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                #print('\nBlock confidence: {}\n'.format(block.confidence))

                for paragraph in block.paragraphs:
                    #print('Paragraph confidence: {}'.format( paragraph.confidence))

                    for word in paragraph.words:
                        word_text = ''.join([
                            symbol.text for symbol in word.symbols
                        ])
                        #print('Word text: {} (confidence: {})'.format( word_text, word.confidence))

                        for symbol in word.symbols:
                            #print('\tSymbol: {} (confidence: {})'.format(symbol.text, symbol.confidence))
                            UnicodeCharVConfidence[SymbolCounter]=(symbol.text,symbol.confidence,symbol.bounding_box)
                            SymbolCounter=SymbolCounter+1
                            UnicodeListSymbols.append(symbol.text)

        word_text = ''.join(UnicodeListSymbols)


        
        #word_text=word_text.replace(" ","")
        # texts = response.text_annotations
        # OutTexts=[]
        # StringOutput=""
        # #TODO looks like first line is the entire read and the rest is the breakdown
        # for text in texts:
        #     #print('\n"{}"'.format(text.description))
        #     OutTexts.append(text.description)
        #     StringOutput=StringOutput+str(text.description)
        #     vertices = (['({},{})'.format(vertex.x, vertex.y)for vertex in text.bounding_poly.vertices])

        # joined_ReadString = " ".join(OutTexts)

        # OutString3=""
        # for Index, Line in enumerate(OutTexts):
        #     if Index>0:
        #         OutString3=OutString3+Line

        #replace all non alphanumeric characters
        #OutString3 = re.sub(r'[^a-zA-Z0-9]', '', OutString3)
        #ListOCR_Reads.append(OutString3)

        #print(OutTexts)
        #print('bounds: {}'.format(','.join(vertices)))

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(response.error.message))

        #return the characters found, and also a dictionary of each character vs confidence
        Cost=0.001
        #print("word_text",word_text)
        return word_text,UnicodeCharVConfidence



