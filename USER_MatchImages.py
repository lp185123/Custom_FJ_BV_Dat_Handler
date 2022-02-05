
import _3DVisLabLib
import json
import cv2
import random
from statistics import mean 
import copy

class MatchImagesObject():
    """Class to hold information for image sorting & match process"""
    def __init__(self):
        self.InputFolder=r"C:\Working\FindIMage_In_Dat\Output"
        self.Outputfolder=r"C:\Working\FindIMage_In_Dat\MatchImages"
        self.TraceExtractedImg_to_DatRecord="TraceImg_to_DatRecord.json"

        self.TraceExtractedImg_to_DatRecordObj=None
        
        self.ImagesInMem_to_Process=dict()
        self.ImagesInMem_InitPairs=dict()



    class FeatureMatch_Dict_Common:
    
        SIFT_default=dict(nfeatures=0,nOctaveLayers=3,contrastThreshold=0.04,edgeThreshold=10)
        
        ORB_default=dict(nfeatures=20000,scaleFactor=1.3,
                        nlevels=2,edgeThreshold=0,
                        firstLevel=0, WTA_K=2,
                        scoreType=0,patchSize=155)
        
        SIFT_Testing=dict(nfeatures=50,contrastThreshold=0.04,edgeThreshold=10)
        
        ORB_Testing=dict(nfeatures=20000,scaleFactor=1.02,
                        nlevels=4,edgeThreshold=0,
                        firstLevel=4, WTA_K=2,
                        scoreType=0,patchSize=100)


MatchImages=MatchImagesObject()

#delete all files in folder
if _3DVisLabLib.yesno("Delete all files in " + str(MatchImages.Outputfolder) + "?"):
    _3DVisLabLib.DeleteFiles_RecreateFolder(MatchImages.Outputfolder)

#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(MatchImages.InputFolder)
#filter out non images
ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)
#get object that links images to dat records
print("attempting to load image to dat record trace file",MatchImages.InputFolder + "\\" + MatchImages.TraceExtractedImg_to_DatRecord)
try:
    with open(MatchImages.InputFolder + "\\" + MatchImages.TraceExtractedImg_to_DatRecord) as json_file:
        MatchImages.TraceExtractedImg_to_DatRecordObj = json.load(json_file)
        print("json loaded succesfully")
        MatchImages.TraceExtractedImg_to_DatRecordObj
except Exception as e:
    print("JSON_Open error attempting to open json file " + str(MatchImages.InputFolder + "\\" + MatchImages.TraceExtractedImg_to_DatRecord) + " " + str(e))
    if _3DVisLabLib.yesno("Continue operation? No image to dat record trace will be possible so only image matching & sorting")==False:
        raise Exception("User declined to continue after JSOn image vs dat record file not found")

#reuse all feature matching code
def GetAverageOfMatches(List,Span):
    Spanlist=List[0:Span]
    Distances=[]
    for elem in Spanlist:
        Distances.append(elem.distance)
    return round(mean(Distances),2)

#load images into memory
for Index, ImagePath in enumerate(ListAllImages):
    print("Loading in image",ImagePath )
    Pod1Image = cv2.imread(ImagePath,0)
    Pod1Image_col = cv2.imread(ImagePath)
    MatchImages.ImagesInMem_to_Process[ImagePath]=(Pod1Image,Pod1Image_col)

def GetClosestImage(InputImgFilename, InputImageDict):
    #for each image in folder get set of feature matching points using orb/sift etc
    BestMatch=99999
    BestImage=None
    file1=None
    file2=None
#roll through all other images and get closest match
    for TestImage in InputImageDict:
        if TestImage==InputImgFilename:
            continue
        pts1,pts2,ORB_Report,ImageLog,ImageTextLog,UnsortedMatches=_3DVisLabLib.ORB_Feature_and_Match_SortImg(InputImageDict[InputImgFilename][0],InputImageDict[TestImage][0],MatchImages.FeatureMatch_Dict_Common.ORB_default,True)
        AverageMatchDistance=GetAverageOfMatches(UnsortedMatches,15)
        #print("average match distance=",GetAverageOfMatches(UnsortedMatches,4))
        #keep best match so far
        if BestMatch>AverageMatchDistance:
            BestMatch=AverageMatchDistance
            BestImage=copy.deepcopy(ImageLog[1])
            file1=TestImage
            file2=InputImgFilename
    if file1.lower()==file2.lower():
        raise Exception("Error - GetClosestImage - cannot match image with itself")
    return file1,file2,BestImage,BestMatch

index=0
#have to make top pairs folder first
_3DVisLabLib. MakeFolder(MatchImages.Outputfolder + "\\Pairs\\")
while len(MatchImages.ImagesInMem_to_Process)>0:
    index=index+1
    RandomImage=random.choice(list(MatchImages.ImagesInMem_to_Process.keys()))
    #print("starting with image",RandomImage)
    #_3DVisLabLib.ImageViewer_Quick_no_resize(MatchImages.ImagesInMem_to_Process[RandomImage][0],0,True,False)


    file1,file2,BestImage,BestMatch=GetClosestImage(RandomImage,MatchImages.ImagesInMem_to_Process)
    
    #if we have two files - copy them into processed/paired dictionary and remove from unprocessed
    if (file1 is not None) and (file2 is not None):
        TempFolder=MatchImages.Outputfolder + "\\Pairs\\"
        #_3DVisLabLib. MakeFolder(TempFolder)
        cv2.imwrite(TempFolder  + str(index) + "_File1_"+ str(BestMatch) +".jpg",MatchImages.ImagesInMem_to_Process[file1][0])
        cv2.imwrite(TempFolder+ str(index) + "_File2_"+ str(BestMatch) + ".jpg",MatchImages.ImagesInMem_to_Process[file2][0])
        del MatchImages.ImagesInMem_to_Process[file1]
        del MatchImages.ImagesInMem_to_Process[file2]
        print(file1,file2)
    else:
        print("No result!!")
    

    print(index,"/",len(MatchImages.ImagesInMem_to_Process))

    _3DVisLabLib.ImageViewer_Quick_no_resize(BestImage,0,False,False)


wwwwwwwwww

for Index, ImagePath in enumerate(ListAllImages):
    print("Loading in image",ImagePath )
    RandChoice=random.choice(ListAllImages)
    #dont check yourself
    if ListAllImages[Index]==RandChoice:
        continue
    #load in image
    #NOTE - loading image in twice so can use old code - if proven to work we can tailor code for single image
    Pod1Image = cv2.imread(ListAllImages[Index],0)
    Pod2Image = cv2.imread(RandChoice)#ListAllImages[Index+1],0)
    Pod1Image_col = cv2.imread(ListAllImages[Index])
    Pod2Image_col =cv2.imread(RandChoice)#ListAllImages[Index+1],0)
    
    #conduct feature matching
    print("Feature matching image")
    pts1,pts2,ORB_Report,ImageLog,ImageTextLog,UnsortedMatches=_3DVisLabLib.ORB_Feature_and_Match_SortImg(Pod1Image,Pod2Image,MatchImages.FeatureMatch_Dict_Common.ORB_default,True)
    AverageMatchDistance=GetAverageOfMatches(UnsortedMatches,4)
    #print("average match distance=",GetAverageOfMatches(UnsortedMatches,4))
    #keep best match so far
    if BestMatch>AverageMatchDistance:
        BestMatch=AverageMatchDistance
        BestImage=copy.deepcopy(ImageLog[1])
        file1=ListAllImages[Index]
        file2=RandChoice
    #pts1,pts2,ORB_Report,ImageLog,ImageTextLog=_3DVisLabLib.SIFT_Feature_and_Match(Pod1Image,Pod2Image,MatchImages.FeatureMatch_Dict_Common.SIFT_default,True)
    
    

#for indexer in range(0,len(ImageLog)):
print(file1,file2)
_3DVisLabLib.ImageViewer_Quick_no_resize(BestImage,0,True,False)
