
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
    Json_to_load=MatchImages.InputFolder + "\\" + MatchImages.TraceExtractedImg_to_DatRecord
    with open(Json_to_load) as json_file:
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
    keypoints,descriptor=_3DVisLabLib.OrbKeyPointsOnly(Pod1Image_col,MatchImages.FeatureMatch_Dict_Common.ORB_default)
    MatchImages.ImagesInMem_to_Process[ImagePath]=(Pod1Image,Pod1Image_col,keypoints,descriptor)

def GetClosestImage_KpsOnly(InputImgFilename, InputImageDict,averagingMatchpts):
    #for use with pre-existing keypoints and descriptors
    ImgVsMatch=dict()
    BestMatch=99999
    BestImage=None
    file1=None
    file2=None
#roll through all other images and get closest match
    for TestImage in InputImageDict:
        if TestImage==InputImgFilename:
            continue
        Keypoints1=MatchImages.ImagesInMem_to_Process[InputImgFilename][2]
        Descriptor1=MatchImages.ImagesInMem_to_Process[InputImgFilename][3]
        Keypoints2=MatchImages.ImagesInMem_to_Process[TestImage][2]
        Descriptor2=MatchImages.ImagesInMem_to_Process[TestImage][3]
        UnsortedMatches=_3DVisLabLib.Orb_FeatureMatch(Keypoints1,Descriptor1,Keypoints2,Descriptor2,999999)
        #pts1,pts2,ORB_Report,ImageLog,ImageTextLog,UnsortedMatches=_3DVisLabLib.ORB_Feature_and_Match_SortImg(InputImageDict[InputImgFilename][0],InputImageDict[TestImage][0],MatchImages.FeatureMatch_Dict_Common.ORB_default,True)
        AverageMatchDistance=GetAverageOfMatches(UnsortedMatches,averagingMatchpts)
        ImgVsMatch[TestImage]=AverageMatchDistance
        #print("average match distance=",GetAverageOfMatches(UnsortedMatches,4))
        #keep best match so far
        if BestMatch>AverageMatchDistance:
            BestMatch=AverageMatchDistance
            #BestImage=copy.deepcopy(ImageLog[1])
            file1=TestImage
            file2=InputImgFilename
    if file1.lower()==file2.lower():
        raise Exception("Error - GetClosestImage - cannot match image with itself")
    return file1,file2,BestImage,BestMatch,ImgVsMatch


def GetClosestImage(InputImgFilename, InputImageDict,averagingMatchpts):
    #for each image in folder get set of feature matching points using orb/sift etc
    ImgVsMatch=dict()
    BestMatch=99999
    BestImage=None
    file1=None
    file2=None
#roll through all other images and get closest match
    for TestImage in InputImageDict:
        if TestImage==InputImgFilename:
            continue
        pts1,pts2,ORB_Report,ImageLog,ImageTextLog,UnsortedMatches=_3DVisLabLib.ORB_Feature_and_Match_SortImg(InputImageDict[InputImgFilename][0],InputImageDict[TestImage][0],MatchImages.FeatureMatch_Dict_Common.ORB_default,True)
        AverageMatchDistance=GetAverageOfMatches(UnsortedMatches,averagingMatchpts)
        ImgVsMatch[TestImage]=AverageMatchDistance
        #print("average match distance=",GetAverageOfMatches(UnsortedMatches,4))
        #keep best match so far
        if BestMatch>AverageMatchDistance:
            BestMatch=AverageMatchDistance
            BestImage=copy.deepcopy(ImageLog[1])
            file1=TestImage
            file2=InputImgFilename
    if file1.lower()==file2.lower():
        raise Exception("Error - GetClosestImage - cannot match image with itself")
    return file1,file2,BestImage,BestMatch,ImgVsMatch

index=0
#have to make top pairs folder first
_3DVisLabLib. MakeFolder(MatchImages.Outputfolder + "\\Pairs\\")
while len(MatchImages.ImagesInMem_to_Process)>0:
    index=index+1
    RandomImage=random.choice(list(MatchImages.ImagesInMem_to_Process.keys()))
    #print("starting with image",RandomImage)
    #_3DVisLabLib.ImageViewer_Quick_no_resize(MatchImages.ImagesInMem_to_Process[RandomImage][0],0,True,False)


    file1,file2,BestImage,BestMatch,ImgVSMatch=GetClosestImage_KpsOnly(RandomImage,MatchImages.ImagesInMem_to_Process,15)
    


    #if we have two files - copy them into processed/paired dictionary and remove from unprocessed
    if (file1 is not None) and (file2 is not None):
        TempFolder=MatchImages.Outputfolder + "\\Pairs\\" + str(BestMatch) + "_" +str(index) 
        #_3DVisLabLib. MakeFolder(TempFolder)
        cv2.imwrite(TempFolder  + "_File1_"+ ".jpg",MatchImages.ImagesInMem_to_Process[file1][1])
        cv2.imwrite(TempFolder + "_File2_"+ ".jpg",MatchImages.ImagesInMem_to_Process[file2][1])
        if MatchImages.TraceExtractedImg_to_DatRecordObj is not None:
            with open(TempFolder + "_DatRecord.txt", 'w') as f:
                f.writelines(str(MatchImages.TraceExtractedImg_to_DatRecordObj[file1]))
                f.writelines(str(MatchImages.TraceExtractedImg_to_DatRecordObj[file2]))

            #with open(TempFolder + "_FirstImgMatches.txt", 'w') as f:
            #    for item in ImgVSMatch:
            #        f.writelines(str(ImgVSMatch[item])+ ",")
        #dvdvdvd
        del MatchImages.ImagesInMem_to_Process[file1]
        del MatchImages.ImagesInMem_to_Process[file2]
        print(file1,file2)
    else:
        print("No result!!")
    

    print(index,"/",len(MatchImages.ImagesInMem_to_Process))

    #_3DVisLabLib.ImageViewer_Quick_no_resize(BestImage,0,False,False)


