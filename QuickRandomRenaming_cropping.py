import _3DVisLabLib
import shutil
import random
import cv2

#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(r"E:\NCR\DELETE")#E:\NCR\TestImages\LightSabreDuel\RandomOrder

#filter out non images
ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)

Rename=False
Rando_crop=True
if Rename==True:
    for index, image in enumerate(ListAllImages):
        OutputFOlder=r"E:\NCR\TestImages\UK_Side_ALL_Random" +"\\" + str(random.random()*1000).replace(".","") + ".jpg"
        shutil.copyfile(image, OutputFOlder)
        print(index,"/",len(ListAllImages))


if Rando_crop==True:
    for index, image in enumerate(ListAllImages):
        print(index)
        OriginalImage_col = cv2.imread(image)
        #crop 4 subimages out of image
        CropSquareSide=120
        for Cropper in range (0,4):
            try:
                OutputFOlder=r"E:\NCR\DELETE_CROPPED" +"\\" + str(index) + "_" + str(Cropper)+ ".jpg"
                CropSquareY=random.randint(0,OriginalImage_col.shape[0]-CropSquareSide)
                CropSquareX=random.randint(0,OriginalImage_col.shape[1]-CropSquareSide)
                #roi = image[startY:endY, startX:endX]
                roi=OriginalImage_col[CropSquareY:CropSquareY+CropSquareSide,CropSquareX:CropSquareX+CropSquareSide,:]
                cv2.imwrite(OutputFOlder,roi)
            except:
                print("image broken")