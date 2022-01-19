from google.cloud import vision
import io
import os



class CloudOCR():
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
        #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Working\FindIMage_In_Dat\VisionAPIDemo\ocrtrial-338212-a4732d2e2a9c.json"
        #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Working\FindIMage_In_Dat\VisionAPIDemo\ServiceAccountToken.json"
        #set GOOGLE_APPLICATION_CREDENTIALS="C:\Working\FindIMage_In_Dat\VisionAPIDemo\ServiceAccountToken.json
    def PerformOCR(self):
        with io.open(r"C:\Working\FindIMage_In_Dat\VisionAPIDemo\Images\0.jpg", 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        response = self.client.text_detection(image=image)
        texts = response.text_annotations
        #print('Texts:')
        OutTexts=[]

        for text in texts:
            print('\n"{}"'.format(text.description))
        #    OutFile = open("myfile.txt","a")
        #    OutFile.write(text.description)
            OutTexts.append(text.description)
            vertices = (['({},{})'.format(vertex.x, vertex.y)for vertex in text.bounding_poly.vertices])

        print(OutTexts)
        print('bounds: {}'.format(','.join(vertices)))

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(response.error.message))



GoogleCloudOCR=CloudOCR()
GoogleCloudOCR.PerformOCR()