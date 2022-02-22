import _3DVisLabLib
import shutil
import random

#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(r"E:\NCR\TestImages\UK_Side_ALL")#E:\NCR\TestImages\LightSabreDuel\RandomOrder

#filter out non images
ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)

for index, image in enumerate(ListAllImages):
    OutputFOlder=r"E:\NCR\TestImages\UK_Side_ALL_Random" +"\\" + str(random.random()*1000).replace(".","") + ".jpg"
    shutil.copyfile(image, OutputFOlder)
    print(index,"/",len(ListAllImages))
