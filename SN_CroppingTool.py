import _3DVisLabLib
import cv2
import numpy as np 

def normalize_2d(matrix):
    #check the matrix isnt all same number
    if matrix.min()==0 and matrix.max()==0:
        print("WARNING, matrix all zeros")
        return matrix
    if matrix.max()==matrix.min():
        print("WARNING: matrix homogenous")
        return np.ones(matrix.shape)
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))






BaseSNR_Folder = input("Please enter images folder: Default is C:\Working\FindIMage_In_Dat\Output")
if len(BaseSNR_Folder)==0:
    BaseSNR_Folder = r"C:\Working\FindIMage_In_Dat\Output"

OutputFolder=BaseSNR_Folder +"\\FilterOutput"
print("Creating output folder",OutputFolder)
if _3DVisLabLib.yesno("Delete output folder?"):
    _3DVisLabLib.DeleteFiles_RecreateFolder(OutputFolder)

FolderFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(BaseSNR_Folder)
#filter out non images
ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(FolderFiles)
print("ListAllImages",len(ListAllImages))
FilteredToImage=[]
for imgitem in ListAllImages:
    if  not "d8" in imgitem.lower():continue
    FilteredToImage.append(imgitem)
print("FilteredToImage",len(FilteredToImage))
#roll through all images and build up stacked image
BaseImage_np=None
kernel = np.ones((5,5),np.float32)/25#kernel size for smoothing - maybe make smoother as such small images
for Indexer,Image in enumerate(FilteredToImage):
    #try:
    #load image
    AddImage=cv2.imread(Image,cv2.IMREAD_GRAYSCALE)
    #invert
    #AddImage = np.invert(AddImage)
    #blur
    #AddImage = cv2.filter2D(AddImage,-1,kernel)
    #dilate
    #AddImage = cv2.dilate(AddImage, kernel, iterations=1)
    #convert to numpy array to stop opencv limiting ceiling
    AddImage_np=np.asarray(AddImage,dtype="int32")
    #try clipping maximum grayscale value so white doesnt saturate imag
    #AddImage_np=np.clip(AddImage_np,0,25)

    #set first image then skip to next iteration to don't double it up
    if BaseImage_np is None:
        BaseImage_np=AddImage_np
        continue

    #add images together
    if Indexer%1==0:#reduce image set to try and mitigate over blurring
        BaseImage_np=BaseImage_np+AddImage_np
        #print((BaseImage_np.max()))
    #except:
        #pass
#normalise to image grayscale range, as max range at moment will not be compatible with opencv or wont
#be interpreted correctly
BaseImage=normalize_2d(BaseImage_np)

_3DVisLabLib.ImageViewer_Quickv2_UserControl(BaseImage,0,True,False)
#display image, wait for user input keypress to determine manual scan parameters
FromTop=0
FromBase=0
FromLeft=0
FromRight=0
ImageToDisplay=BaseImage.copy()
SaveOutDictionary=dict()
while True:
    #general shape for reference ImageToDisplay[175,335] 0 is height 335 is width
    #ImageToDisplay=BaseImage.copy()
    ImageToDisplay=BaseImage.copy()#
    ImageToDisplay[:,:]=1
    ImageToDisplay[FromTop:(BaseImage.shape[0]-FromBase),FromLeft:(BaseImage.shape[1]-FromRight)]=BaseImage[FromTop:(BaseImage.shape[0]-FromBase),FromLeft:(BaseImage.shape[1]-FromRight)]


    User_keypress=_3DVisLabLib.ImageViewer_Quickv2_UserControl(ImageToDisplay,0,True,False)
    print(User_keypress)
    if User_keypress=="w":
        FromTop=FromTop+1
    if User_keypress=="s":
        FromBase=FromBase+1
    if User_keypress=="a":
        FromLeft=FromLeft+1
    if User_keypress=="d":
        FromRight=FromRight+1

    if User_keypress=="i":
        FromTop=FromTop-1
    if User_keypress=="k":
        FromBase=FromBase-1
    if User_keypress=="j":
        FromLeft=FromLeft-1
    if User_keypress=="l":
        FromRight=FromRight-1



    if User_keypress=="t":
        #roll through all images that have been averaged together for a quick preview
        SaveOutDictionary=dict()#clean this out
        for Indexer2,Image2 in enumerate(FilteredToImage):
            Display=cv2.imread(Image2,cv2.IMREAD_GRAYSCALE)
            ImageToDisplay=Display.copy()#
            ImageToDisplay[:,:]=255
            ImageToDisplay[FromTop:(BaseImage.shape[0]-FromBase),FromLeft:(BaseImage.shape[1]-FromRight)]=Display[FromTop:(BaseImage.shape[0]-FromBase),FromLeft:(BaseImage.shape[1]-FromRight)]
            _3DVisLabLib.ImageViewer_Quickv2_UserControl(ImageToDisplay,0,False,False)
            SaveOutDictionary[Image2]=ImageToDisplay
    
    
    if User_keypress=="x":
        for imgitem in SaveOutDictionary:
            
            FilenameSansPath=imgitem.split("\\")[-1]
            print("saving out",OutputFolder + "\\" + FilenameSansPath)
            cv2.imwrite(OutputFolder + "\\" + FilenameSansPath,SaveOutDictionary[imgitem])
        print("Finished saving")