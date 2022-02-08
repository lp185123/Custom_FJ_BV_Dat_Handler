
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
                        cv2.imwrite(MatchImages.OutputDuplicates  + str(ScanIndex)+ "_File1_"+ str(HistogramSimilarity)+".jpg",MatchImages.ImagesInMem_to_Process[BaseImageKey].ImageColour)
                        cv2.imwrite(MatchImages.OutputDuplicates + str(ScanIndex)+ "_File2_"+ str(HistogramSimilarity)+".jpg",MatchImages.ImagesInMem_to_Process[image].ImageColour)
                        #if tracer file exists we can also save link to dat file and record
                        if MatchImages.TraceExtractedImg_to_DatRecordObj is not None:
                                Savefile_json=MatchImages.OutputDuplicates + str(ScanIndex) +"_DatRecord.txt"
                                DictOutputDetails=dict()
                                DictOutputDetails["BASEIMAGE"]=MatchImages.TraceExtractedImg_to_DatRecordObj[image]
                                DictOutputDetails["TESTEDIMAGE"]=MatchImages.TraceExtractedImg_to_DatRecordObj[BaseImageKey]
                                with open(Savefile_json, 'w') as outfile:
                                    json.dump(DictOutputDetails, outfile)#1 is the answer file
                        del MatchImages.DuplicatesToCheck[image]
                #print(HistogramSimilarity)
            del MatchImages.DuplicatesToCheck[BaseImageKey]



    
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