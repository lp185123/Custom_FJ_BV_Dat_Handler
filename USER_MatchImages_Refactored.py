
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
        #self.InputFolder=r"E:\NCR\SR_Generations\Sprint\Russia\DC\ForStd_Gen\ForGen"
        self.InputFolder=r"C:\Working\FindIMage_In_Dat\Output"
        self.Outputfolder=r"C:\Working\FindIMage_In_Dat\MatchImages"
        self.TraceExtractedImg_to_DatRecord="TraceImg_to_DatRecord.json"
        self.OutputPairs=self.Outputfolder + "\\Pairs\\"
        self.OutputDuplicates=self.Outputfolder + "\\Duplicates\\"
        self.Std_dv_Cutoff=0.5#set this to cut off how close notes can be in similarity
        self.TraceExtractedImg_to_DatRecordObj=None
        self.ImagesInMem_to_Process=dict()
        self.DuplicatesToCheck=dict()
        self.DuplicatesFound=[]
        self.Mean_Std_Per_cyclelist=None
        self.HistogramSelfSimilarityThreshold=0.005
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


    class ImageInfo():
        def __init__(self):
            self.Histogram=None
            self.ImageColour=None
            self.ImageAdjusted=None
            self.FM_Keypoints=None
            self.FM_Descriptors=None



def CompareHistograms(histo_Img1,histo_Img2):
    def L2Norm(H1,H2):
        distance =0
        for i in range(len(H1)):
            distance += np.square(H1[i]-H2[i])
        return np.sqrt(distance)

    similarity_metric = L2Norm(histo_Img1,histo_Img2)
    return similarity_metric


def main():
    #create resource manager
    MatchImages=MatchImagesObject()
    #delete all files in folder
    print("Delete output folders:\n",MatchImages.Outputfolder)
    if _3DVisLabLib.yesno("?"):
        _3DVisLabLib.DeleteFiles_RecreateFolder(MatchImages.Outputfolder)
        #make subfolders
        _3DVisLabLib. MakeFolder( MatchImages.OutputPairs)
        _3DVisLabLib. MakeFolder(MatchImages.OutputDuplicates)

    #ask if user wants to check for duplicates
    print("Get duplicates only?? - this will be in factorial time (very long)!!!")
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
    #start timer
    MatchImages.startTime=time.time()

            
    #populate images 
    #load images into memory
    for Index, ImagePath in enumerate(ListAllImages):
        print("Loading in image",ImagePath )
        ImageInfo=MatchImages.ImageInfo()

        Pod1Image = cv2.imread(ImagePath)
        Pod1Image_col = cv2.imread(ImagePath)
        Pod1Image_col_adjusted=Pod1Image_col[0:int(Pod1Image_col.shape[0]/2),0:int(Pod1Image_col.shape[1]),:]
        #get feature match keypoints
        keypoints,descriptor=_3DVisLabLib.OrbKeyPointsOnly(Pod1Image_col,MatchImages.FeatureMatch_Dict_Common.ORB_default)
        #get histogram for comparing colours
        hist = cv2.calcHist([Pod1Image_col], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        #load into tuple for object
        ImageInfo.Histogram=hist
        ImageInfo.ImageGrayscale=Pod1Image
        ImageInfo.ImageColour=Pod1Image_col
        ImageInfo.ImageAdjusted=Pod1Image_col_adjusted
        ImageInfo.FM_Keypoints=keypoints
        ImageInfo.FM_Descriptors=descriptor
        MatchImages.ImagesInMem_to_Process[ImagePath]=(ImageInfo)

    #build dictionary to remove items from
    for Index, img in enumerate(MatchImages.ImagesInMem_to_Process):
        MatchImages.DuplicatesToCheck[img]=img
          #self.DuplicatesToCheck=dict()
        #self.DuplicatesFound=[]
    if MatchImages.GetDuplicates==True:
        ScanIndex=0
        StartingLength_images=len(MatchImages.DuplicatesToCheck)
        DuplicatesFound=0
        while len(MatchImages.DuplicatesToCheck)>0:
            print("Image Set remaining:",len(MatchImages.DuplicatesToCheck),"/",len(MatchImages.ImagesInMem_to_Process))
            #get any image from set
            BaseImageKey=random.choice(list(MatchImages.DuplicatesToCheck.keys()))
            BaseImage=MatchImages.ImagesInMem_to_Process[BaseImageKey]
            for image in MatchImages.ImagesInMem_to_Process:
                ScanIndex=ScanIndex+1
                TestImage=MatchImages.ImagesInMem_to_Process[image]
                HistogramSimilarity=CompareHistograms(BaseImage.Histogram,TestImage.Histogram)
                if HistogramSimilarity<MatchImages.HistogramSelfSimilarityThreshold:
                    if BaseImageKey!=image:
                        print("Duplicate found!!",BaseImageKey,image)
                        DuplicatesFound=DuplicatesFound+1
                        cv2.imwrite(MatchImages.OutputDuplicates  + str(ScanIndex)+ "_File1_"+ str(HistogramSimilarity)+".jpg",MatchImages.ImagesInMem_to_Process[BaseImageKey].ImageColour)
                        cv2.imwrite(MatchImages.OutputDuplicates + str(ScanIndex)+ "_File2_"+ str(HistogramSimilarity)+".jpg",MatchImages.ImagesInMem_to_Process[image].ImageColour)
                        #if tracer file exists we can also save link to dat file and record
                        if MatchImages.TraceExtractedImg_to_DatRecordObj is not None:
                                Savefile_json=MatchImages.OutputDuplicates + str(ScanIndex) +"_DatRecord.txt"
                                DictOutputDetails=dict()
                                DictOutputDetails["BASEIMAGE"]=MatchImages.TraceExtractedImg_to_DatRecordObj[image]
                                DictOutputDetails["TESTEDIMAGE"]=MatchImages.TraceExtractedImg_to_DatRecordObj[BaseImageKey]
                                with open(Savefile_json, 'w') as outfile:
                                    json.dump(DictOutputDetails, outfile)
                        del MatchImages.DuplicatesToCheck[image]
            del MatchImages.DuplicatesToCheck[BaseImageKey]
        print(DuplicatesFound,"/",StartingLength_images," Duplicates found and stored at",MatchImages.OutputDuplicates)
        MatchImages.Endtime= time.time()
        print("time taken (hrs):",round((MatchImages.Endtime- MatchImages.startTime)/60/60 ),2)
        exit()

    #create indexed dictionary of images so we can start combining lists of images
    MatchImages.ImagesInMem_Pairing=dict()
    for Index, img in enumerate(MatchImages.ImagesInMem_to_Process):
        ImgCol_InfoSheet=ImgCol_InfoSheet_Class()
        ImgCol_InfoSheet.FirstImage=img
        MatchImages.ImagesInMem_Pairing[Index]=([img],ImgCol_InfoSheet)
    #match images loop
    #while True:
    #    pass

    #couple up images
    OutOfUse=0
    for looper in range (0,3):
        for BaseImageList in MatchImages.ImagesInMem_Pairing:
            print(OutOfUse,"removed from", len(MatchImages.ImagesInMem_Pairing),looper)
            #if list is inactive, skip
            if MatchImages.ImagesInMem_Pairing[BaseImageList][1].InUse==False:
                continue

            CheckImages_InfoSheet=CheckImages_Class()
            #get info for base image
            Base_Image_name=MatchImages.ImagesInMem_Pairing[BaseImageList][1].FirstImage
            Base_Image_Histo=MatchImages.ImagesInMem_to_Process[Base_Image_name].Histogram
            Base_Image_FMatches=MatchImages.ImagesInMem_to_Process[Base_Image_name].FM_Keypoints
            Base_Image_Descrips=MatchImages.ImagesInMem_to_Process[Base_Image_name].FM_Descriptors

            for TestImageList in MatchImages.ImagesInMem_Pairing:
                if MatchImages.ImagesInMem_Pairing[BaseImageList][1].InUse==False:
                    continue
                #check not testing itself
                if BaseImageList==TestImageList:
                    #set this to checked - warning will set base image to checked as well
                    continue#skip iteration
                #test images - this is where different strategies may come in
                #get first image, can also use the list for this
                #get info for test images
                Test_Image_name=MatchImages.ImagesInMem_Pairing[TestImageList][1].FirstImage
                Test_Image_Histo=MatchImages.ImagesInMem_to_Process[Test_Image_name].Histogram
                Test_Image_FMatches=MatchImages.ImagesInMem_to_Process[Test_Image_name].FM_Keypoints
                Test_Image_Descrips=MatchImages.ImagesInMem_to_Process[Test_Image_name].FM_Descriptors
                HistogramSimilarity=CompareHistograms(Base_Image_Histo,Test_Image_Histo)
                CheckImages_InfoSheet.AllHisto_results.append(HistogramSimilarity)
                if HistogramSimilarity<CheckImages_InfoSheet.BestMatch_Histo:
                    CheckImages_InfoSheet.BestMatch_Histo=HistogramSimilarity
                    CheckImages_InfoSheet.BestMatch_Histo_listIndex=TestImageList

            #after check all images, if a result then copy that list into the first list so combine the sets of images
            if len(CheckImages_InfoSheet.AllHisto_results)>0:
                #list of images now inactive as will be copied to another
                #[0] here is the list of images, while [1] is the info card
                #TempUpdatedList=MatchImages.ImagesInMem_Pairing[BaseImageList]
                #del MatchImages.ImagesInMem_Pairing[BaseImageList]
                MatchImages.ImagesInMem_Pairing[BaseImageList]=(MatchImages.ImagesInMem_Pairing[BaseImageList][0]+MatchImages.ImagesInMem_Pairing[CheckImages_InfoSheet.BestMatch_Histo_listIndex][0],MatchImages.ImagesInMem_Pairing[BaseImageList][1])
                MatchImages.ImagesInMem_Pairing[CheckImages_InfoSheet.BestMatch_Histo_listIndex][1].InUse=False
                OutOfUse=OutOfUse+1
                #MatchImages.ImagesInMem_Pairing[BaseImageList][0]=MatchImages.ImagesInMem_Pairing[BaseImageList][0]+MatchImages.ImagesInMem_Pairing[CheckImages_InfoSheet.BestMatch_Histo_listIndex][0]


    #lets write out pairing
    for ListIndex, ListOfImages in enumerate(MatchImages.ImagesInMem_Pairing):
        #make folder for each set of images
        if MatchImages.ImagesInMem_Pairing[ListOfImages][1].InUse==True:
            SetMatchImages_folder=MatchImages.OutputPairs +"\\" + str(ListIndex) +"\\"
            _3DVisLabLib. MakeFolder(SetMatchImages_folder)
            for imgIndex, Images in enumerate (MatchImages.ImagesInMem_Pairing[ListOfImages][0]):
                #MatchDistance=str(MatchImages.ImagesInMem_Pairing[ListOfImages][1])
                TempFfilename=SetMatchImages_folder  + "00" + str(ListIndex) + "_" +str(imgIndex)  + ".jpg"
                cv2.imwrite(TempFfilename,MatchImages.ImagesInMem_to_Process[Images].ImageColour)




class CheckImages_Class():
    def __init__(self):
        self.AllHisto_results=[]
        self.All_FM_results=[]
        self.MatchingImgLists=[]
        self.BestMatch_Histo=99999999
        self.BestMatch_Histo_listIndex=None


class ImgCol_InfoSheet_Class():
    def __init__(self):
        self.InUse=True
        self.Cycles=0
        self.AllImgs_Histo_Std=0
        self.AllImgs_Hist_mean=0
        self.AllImgs_FM_std=0
        self.AllImgs_FM_mean=0
        self.FirstImage=None





    
def CompareHistograms(histo_Img1,histo_Img2):
    def L2Norm(H1,H2):
        distance =0
        for i in range(len(H1)):
            distance += np.square(H1[i]-H2[i])
        return np.sqrt(distance)

    similarity_metric = L2Norm(histo_Img1,histo_Img2)
    return similarity_metric






if __name__ == "__main__":
    #entry point
    #try:
    
    main()
    #except Exception as e:
    #    ##note: cleaning up after exception should be to set os.chdir(anything else) or it will lock the folder
    #    print(e)
    #    # printing stack trace
    #    print("Press any key to continue")
    #    os.system('pause')