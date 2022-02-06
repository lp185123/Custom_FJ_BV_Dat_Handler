
from tkinter import Y
import _3DVisLabLib
import json
import cv2
import random
from statistics import mean 
import copy
import random
import time
import statistics
import scipy
import numpy as np
import gc
#gc.disable()

class MatchImagesObject():
    """Class to hold information for image sorting & match process"""
    def __init__(self):
        self.InputFolder=r"E:\NCR\TestImages\UK_SMall"
        self.Outputfolder=r"C:\Working\FindIMage_In_Dat\MatchImages"
        self.TraceExtractedImg_to_DatRecord="TraceImg_to_DatRecord.json"
        self.OutputPairs=self.Outputfolder + "\\Pairs\\"
        self.OutputDuplicates=self.Outputfolder + "\\Duplicates\\"
        self.TraceExtractedImg_to_DatRecordObj=None
        self.ImagesInMem_to_Process=dict()
        #self.ImagesInMem_to_Process_Orphans=dict()#cant deepcopy feature match keypoints
        self.ImagesInMem_Pairing=dict()
        self.ImagesInMem_Pairing_orphans=dict()
        self.GetDuplicates=False
        self.startTime =None
        self.Endtime =None
        

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

def GetAverageOfMatches(List,Span):
    Spanlist=List[0:Span]
    Distances=[]
    for elem in Spanlist:
        Distances.append(elem.distance)
    return round(mean(Distances),2)

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
                raise Exception("Error - GetClosestImage_KpsOnly - cannot match image with itself")
    return file1,file2,BestImage,BestMatch,ImgVsMatch

def ClosestList_images(InputImgDict_IndexID, InputImageDict_OfLists,averagingMatchpts,BaseImgDict,ImgPairings_keyDict):
    ImgVsMatch=dict()
    BestMatch=99999
    BestImage=None
    BestList=None
    BestID=None
    #roll through all other images and get closest match
    for TestListID in InputImageDict_OfLists:
        #dont test against self
        if InputImgDict_IndexID==TestListID:
            continue
        if TestListID not in ImgPairings_keyDict:
            continue
        InputList=InputImageDict_OfLists[InputImgDict_IndexID][0]
        TestList=InputImageDict_OfLists[TestListID][0]
        Keypoints1=BaseImgDict[InputList[0]][2]
        Descriptor1=BaseImgDict[InputList[0]][3]
        Keypoints2=BaseImgDict[TestList[0]][2]
        Descriptor2=BaseImgDict[TestList[0]][3]
        Histo1=BaseImgDict[InputList[0]][4]
        Histo2=BaseImgDict[TestList[0]][4]
        UnsortedMatches=_3DVisLabLib.Orb_FeatureMatch(Keypoints1,Descriptor1,Keypoints2,Descriptor2,99999)
        HistogramSimilarity=CompareHistograms(Histo1,Histo2)
        #pts1,pts2,ORB_Report,ImageLog,ImageTextLog,UnsortedMatches=_3DVisLabLib.ORB_Feature_and_Match_SortImg(InputImageDict[InputImgFilename][0],InputImageDict[TestImage][0],MatchImages.FeatureMatch_Dict_Common.ORB_default,True)
        AverageMatchDistance=GetAverageOfMatches(UnsortedMatches,averagingMatchpts)+HistogramSimilarity
        ImgVsMatch[TestList[0]]=AverageMatchDistance
        #print("average match distance=",GetAverageOfMatches(UnsortedMatches,4))
        #keep best match so far
        if BestMatch>AverageMatchDistance:
            BestMatch=AverageMatchDistance
            #BestImage=copy.deepcopy(ImageLog[1])
            BestList=TestList
            BestID=TestListID

    return BestList,BestID,BestMatch,ImgVsMatch

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


def CompareHistograms(histo_Img1,histo_Img2):
    def L2Norm(H1,H2):
        distance =0
        for i in range(len(H1)):
            distance += np.square(H1[i]-H2[i])
        return np.sqrt(distance)

    similarity_metric = L2Norm(histo_Img1,histo_Img2)
    return similarity_metric

def DoubleUp_ImgMatches():
    index=0
    List_BestMatches_QuickMetric=[]
    #create temp dictionary so we can remove keys as we test images
    ImgPairings_keyDict=dict()
    for Key in MatchImages.ImagesInMem_Pairing:
        ImgPairings_keyDict[Key]=Key
    while len(ImgPairings_keyDict)>0:
        print("length of list pairing dictionary",len(MatchImages.ImagesInMem_Pairing))
        index=index+1
        #print(index,"/",str(len(ImgPairings_keyDict)),"cycle", cycle)
        #go through pairs of images - each time in theory each value list length should double
        TestlistImages_index=random.choice(list(ImgPairings_keyDict.keys()))
        
        #find closest pair - check first instance of pair only - not sure if this is significant
        BestList,BestID,BestMatch,ImgVsMatch=ClosestList_images(TestlistImages_index,MatchImages.ImagesInMem_Pairing,15,MatchImages.ImagesInMem_to_Process,ImgPairings_keyDict)

        #print("BestMatch",BestMatch)

        if BestList is not None:
            
            #check if duplicate then save if so
            if BestMatch <0.01:
                print("Duplicate found")
                RandoNo=str(random.random()/100)
                cv2.imwrite(MatchImages.OutputDuplicates  + RandoNo+ "_File1_"+ ".jpg",MatchImages.ImagesInMem_to_Process[BestList[0]][1])
                cv2.imwrite(MatchImages.OutputDuplicates + RandoNo+ "_File2_"+ ".jpg",MatchImages.ImagesInMem_to_Process[MatchImages.ImagesInMem_Pairing[TestlistImages_index][0][0]][1])
            #if tracer file exists to trace images back to dat records save filenames y
                if MatchImages.TraceExtractedImg_to_DatRecordObj is not None:
                    with open(MatchImages.OutputDuplicates + RandoNo +"_DatRecord.txt", 'w') as f:
                        TempOut1=str(MatchImages.TraceExtractedImg_to_DatRecordObj[BestList[0]])
                        TempOut2=str(MatchImages.TraceExtractedImg_to_DatRecordObj[MatchImages.ImagesInMem_Pairing[TestlistImages_index][0][0]])
                        TempOut1=TempOut1.replace("\\","/")
                        TempOut2=TempOut2.replace("\\","/")
                        f.writelines(TempOut1)
                        f.writelines(TempOut2)
            else:
                #dont add duplicates or it can mess up the stats
                List_BestMatches_QuickMetric.append(BestMatch)
            #merge the two lists
            #rebuild the tuple of list of images and list of bestmatch scores
            Temp_ListMatchScores=MatchImages.ImagesInMem_Pairing[TestlistImages_index][1]
            Temp_ListMatchScores.append(BestMatch)
            MatchImages.ImagesInMem_Pairing[TestlistImages_index]=(MatchImages.ImagesInMem_Pairing[TestlistImages_index][0]+MatchImages.ImagesInMem_Pairing[BestID][0],Temp_ListMatchScores)
            #delete the test list as its now merged with input list
            del MatchImages.ImagesInMem_Pairing[BestID]
            
            #delete key from test dictionary
            del ImgPairings_keyDict[TestlistImages_index]
            del ImgPairings_keyDict[BestID]

            

        else:
            #add to orphans
            MatchImages.ImagesInMem_Pairing_orphans[TestlistImages_index]=ImgPairings_keyDict[TestlistImages_index]
            #no match
            del ImgPairings_keyDict[TestlistImages_index]
            
    return List_BestMatches_QuickMetric
#create resource manager
MatchImages=MatchImagesObject()

#delete all files in folder
print("Delete output folders:\n",MatchImages.Outputfolder)
if _3DVisLabLib.yesno("?"):
    _3DVisLabLib.DeleteFiles_RecreateFolder(MatchImages.Outputfolder)

#ask if user wants to check for duplicates
print("Get duplicates only?? - this will be in quadratic/O(n)^2 time (very long)!!!")
MatchImages.GetDuplicates= _3DVisLabLib.yesno("?")

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

MatchImages.startTime=time.time()
#load images into memory
for Index, ImagePath in enumerate(ListAllImages):
    print("Loading in image",ImagePath )
    Pod1Image = cv2.imread(ImagePath,0)
    Pod1Image_col = cv2.imread(ImagePath)
    #get feature match keypoints
    keypoints,descriptor=_3DVisLabLib.OrbKeyPointsOnly(Pod1Image_col,MatchImages.FeatureMatch_Dict_Common.ORB_default)
    #get histogram for comparing colours
    hist = cv2.calcHist([Pod1Image_col], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    #load into tuple for object
    MatchImages.ImagesInMem_to_Process[ImagePath]=(Pod1Image,Pod1Image_col,keypoints,descriptor,hist)

#make subfolders
_3DVisLabLib. MakeFolder(MatchImages.Outputfolder + "\\Pairs\\")
_3DVisLabLib. MakeFolder(MatchImages.Outputfolder + "\\Duplicates\\")
#MatchedImage,InputImage,BestImage,BestMatch,ImgVSMatch=GetClosestImage_KpsOnly(RandomImage,MatchImages.ImagesInMem_to_Process_copy,15)



for Index, img in enumerate(MatchImages.ImagesInMem_to_Process):
    MatchImages.ImagesInMem_Pairing[Index]=([img],[])


if MatchImages.GetDuplicates==True:
    Duplicates=0
    print("Getting duplicates - this will be in quadratic/O(n)^2 time (very long)!!!")
    #check each file against all other files
    index = 0
    ParsedDict=()
    while len( MatchImages.ImagesInMem_to_Process)>0:
        #get random image from those in memory
        External_MemImg=random.choice(list(MatchImages.ImagesInMem_to_Process.keys()))
        #test two images 
        MatchedImage,InputImage,BestImage,BestMatch,ImgVSMatch=GetClosestImage_KpsOnly(External_MemImg,MatchImages.ImagesInMem_to_Process,15)
        if MatchedImage is not None:
            #have a match - lets see if its a perfect match (duplicate)
            #in theory should be 0.0 as a score but potentially small variation due to .jpg compression?
            if BestMatch<0.01:
                cv2.imwrite(MatchImages.OutputDuplicates  + str(index)+ "_File1_"+ ".jpg",MatchImages.ImagesInMem_to_Process[MatchedImage][1])
                cv2.imwrite(MatchImages.OutputDuplicates + str(index)+ "_File2_"+ ".jpg",MatchImages.ImagesInMem_to_Process[InputImage][1])
                #if tracer file exists we can also save link to dat file and record
                if MatchImages.TraceExtractedImg_to_DatRecordObj is not None:
                        with open(MatchImages.OutputDuplicates + str(index) +"_DatRecord.txt", 'w') as f:
                            TempOut1=str(MatchImages.TraceExtractedImg_to_DatRecordObj[MatchedImage])
                            TempOut2=str(MatchImages.TraceExtractedImg_to_DatRecordObj[InputImage])
                            TempOut1=TempOut1.replace("\\","/")
                            TempOut2=TempOut2.replace("\\","/")
                            f.writelines(TempOut1)
                            f.writelines(TempOut2)
                Duplicates=Duplicates+1
        #delete key
        del MatchImages.ImagesInMem_to_Process[External_MemImg]
        #update user
        print(str(len(MatchImages.ImagesInMem_to_Process)))
        #index=index+len(MatchImages.ImagesInMem_to_Process)
        #print(str(index),"/",str(len(MatchImages.ImagesInMem_to_Process)**2))
    print(Duplicates,"Duplicates found and stored at",MatchImages.OutputDuplicates)
    MatchImages.Endtime= time.time()
    print("time taken (s):",round(MatchImages.Endtime -MatchImages.startTime ))
    exit()

Mean_Std_Per_cyclelist=[]
try:
    for looper in range (0,3):
        List_Bestmatches=DoubleUp_ImgMatches()
        LowMatches_to_remove=[]
        temp_stdev=statistics.pstdev(List_Bestmatches)
        temp_mean=sum(List_Bestmatches) / len(List_Bestmatches)
        Cutoff_Std_Deviation=temp_mean+(temp_stdev*1)#magic number until we put it in data object
        Mean_Std_Per_cyclelist.append((looper,temp_stdev,temp_mean,copy.deepcopy(Cutoff_Std_Deviation)))
        #here we can remove any matches that are N standard deviations away from the mean
        
        for ListIndex, ListOfImages in enumerate(MatchImages.ImagesInMem_Pairing):
            try:
                #will have a list of match distance per cycle - get the last one
                MatchDistance=(MatchImages.ImagesInMem_Pairing[ListOfImages][1][-1])
                #if distance is bigger than std deviation cut off - remove and place into orphans dictionary
                if MatchDistance>Mean_Std_Per_cyclelist[0][3]:#use first instance of std-deviation
                    LowMatches_to_remove.append(ListOfImages)
            except:
                print("error with ",ListOfImages,"in", "MatchImages.ImagesInMem_Pairing","moving to orphans")
                LowMatches_to_remove.append(ListOfImages)
        #remove high distance matches now
        print("bad matches found",len(LowMatches_to_remove))
        for badmatch in LowMatches_to_remove:
            MatchImages.ImagesInMem_Pairing_orphans[badmatch]=MatchImages.ImagesInMem_Pairing[badmatch]
            del MatchImages.ImagesInMem_Pairing[badmatch]

        plop=1
except Exception as e:
    print(e)
    raise Exception("Error with organise loop")

#lets write out pairing
for ListIndex, ListOfImages in enumerate(MatchImages.ImagesInMem_Pairing):
    for imgIndex, Images in enumerate (MatchImages.ImagesInMem_Pairing[ListOfImages][0]):
        MatchDistance=str(MatchImages.ImagesInMem_Pairing[ListOfImages][1])
        TempFfilename=MatchImages.OutputPairs  + "00" + str(ListIndex) + "_" +str(imgIndex) + "_mD" + MatchDistance + ".jpg"
        cv2.imwrite(TempFfilename,MatchImages.ImagesInMem_to_Process[Images][1])

MatchImages.Endtime= time.time()
print("Orphans",len(MatchImages.ImagesInMem_Pairing_orphans))
print("time taken (s):",round(MatchImages.Endtime- MatchImages.startTime ))
print(Mean_Std_Per_cyclelist)
