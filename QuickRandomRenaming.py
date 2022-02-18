import _3DVisLabLib
import shutil
import random

#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(r"E:\NCR\TestImages\LightSabreDuel")#E:\NCR\TestImages\LightSabreDuel\RandomOrder

#filter out non images
ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)

for image in ListAllImages:
    OutputFOlder=r"E:\NCR\TestImages\LightSabreDuel\RandomOrder" +"\\" + str(random.random()*1000).replace(".","") + ".jpg"
    shutil.copyfile(image, OutputFOlder)
