
from logging import raiseExceptions
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

import matplotlib
matplotlib.use('Agg')#can get "Tcl_AsyncDelete: async handler deleted by the wrong thread" crashes otherwise
import matplotlib.pyplot as plt




def PlotAndSave_2datas(Title,Filepath,Data1):
    
    #this causes crashes
    #save out plot of 1D data
    try:
        #Data1=np.random.rand(30,22)
        plt.pcolormesh((Data1), cmap = 'autumn')
        #plt.plot(Data1,Data2,'bo')#bo will draw dots instead of connected line
        plt.ylabel(Title)
        #plt.ylim([0, max(Data1)])
        #plt.ylim([0, max(Data2)])
        plt.savefig(Filepath)
        plt.cla()
        plt.clf()
        plt.close()
    except Exception as e:
        print("Error with matpyplot",e)

class MatchImagesObject():
    """Class to hold information for image sorting & match process"""
    def __init__(self):
        #self.InputFolder=r"E:\NCR\SR_Generations\Sprint\Russia\DC\ForStd_Gen\ForGen"
        self.InputFolder=r"E:\NCR\TestImages\UK\Output"
        self.Outputfolder=r"C:\Working\FindIMage_In_Dat\MatchImages"
        self.TraceExtractedImg_to_DatRecord="TraceImg_to_DatRecord.json"
        self.OutputPairs=self.Outputfolder + "\\Pairs\\"
        self.OutputDuplicates=self.Outputfolder + "\\Duplicates\\"
        self.AverageDistanceFM=20
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
            self.FourierTransform_mag=None
            self.ImageGrayscale=None
            
def PlotAndSave(Title,Filepath,Data,maximumvalue):
    
    #this causes crashes
    #save out plot of 1D data
    try:
        plt.plot(Data)
        plt.ylabel(Title)
        plt.ylim([0, maximumvalue])
        plt.savefig(Filepath)
        plt.cla()
        plt.clf()
        plt.close()
    except Exception as e:
        print("Error with matpyplot",e)


def GetAverageOfMatches(List,Span):
    Spanlist=List[0:Span]
    Distances=[]
    for elem in Spanlist:
        Distances.append(elem.distance)
    return round(mean(Distances),2)

def CompareHistograms(histo_Img1,histo_Img2):
    def L2Norm(H1,H2):
        distance =0
        for i in range(len(H1)):
            distance += np.square(H1[i]-H2[i])
        return np.sqrt(distance)

    similarity_metric = L2Norm(histo_Img1,histo_Img2)
    return similarity_metric

def normalize_2d(matrix):
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

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

        Pod1Image_Grayscale = cv2.imread(ImagePath,0)
        Pod1Image_col = cv2.imread(ImagePath)

        #Pod1Image_Grayscale=cv2.resize(Pod1Image_Grayscale.copy(),(100,100))
        #Pod1Image_col=cv2.resize(Pod1Image_col.copy(),(100,100))

        Pod1Image_col_adjusted=Pod1Image_col[0:int(Pod1Image_col.shape[0]/1.6),0:int(Pod1Image_col.shape[1]),:]
        Pod1Image_col=Pod1Image_col[0:int(Pod1Image_col.shape[0]/1.6),0:int(Pod1Image_col.shape[1]),:]
        #get feature match keypoints
        keypoints,descriptor=_3DVisLabLib.OrbKeyPointsOnly(Pod1Image_col_adjusted,MatchImages.FeatureMatch_Dict_Common.ORB_default)
        #get histogram for comparing colours
        hist = cv2.calcHist([Pod1Image_col], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        #get fourier transform
        #dft = cv2.dft(np.float32(Pod1Image_Grayscale),flags = cv2.DFT_COMPLEX_OUTPUT)
        #dft_shift = np.fft.fftshift(dft)
        #magnitude = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
        #product = 20*np.log(magnitude)
        f = np.fft.fft2(Pod1Image_Grayscale)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        #load into tuple for object
        ImageInfo.Histogram=hist
        ImageInfo.ImageGrayscale=Pod1Image_Grayscale
        ImageInfo.ImageColour=Pod1Image_col
        ImageInfo.ImageAdjusted=Pod1Image_col_adjusted
        ImageInfo.FM_Keypoints=keypoints
        ImageInfo.FM_Descriptors=descriptor
        ImageInfo.FourierTransform_mag=magnitude_spectrum
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



    HM_data_histo = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    HM_data_FM = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    HM_data_Both = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    HM_data_FourierDifference = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    #for looper in range (0,1):
    
    for BaseImageList in MatchImages.ImagesInMem_Pairing:
        
        #print(OutOfUse,"removed from", len(MatchImages.ImagesInMem_Pairing),looper)
        #if list is inactive, skip
        if MatchImages.ImagesInMem_Pairing[BaseImageList][1].InUse==False:
            pass
            #continue

        CheckImages_InfoSheet=CheckImages_Class()
        #get info for base image
        Base_Image_name=MatchImages.ImagesInMem_Pairing[BaseImageList][1].FirstImage
        Base_Image_Histo=MatchImages.ImagesInMem_to_Process[Base_Image_name].Histogram
        Base_Image_FMatches=MatchImages.ImagesInMem_to_Process[Base_Image_name].FM_Keypoints
        Base_Image_Descrips=MatchImages.ImagesInMem_to_Process[Base_Image_name].FM_Descriptors
        Base_Image_FourierMag=MatchImages.ImagesInMem_to_Process[Base_Image_name].FourierTransform_mag

        for TestImageList in MatchImages.ImagesInMem_Pairing:
            print(BaseImageList,TestImageList)
            if TestImageList<BaseImageList:
                #data is diagonally symmeterical
                continue
                
            if MatchImages.ImagesInMem_Pairing[TestImageList][1].InUse==False:
                pass
                #continue
            #check not testing itself
            if BaseImageList==TestImageList:
                pass
                #set this to checked - warning will set base image to checked as well
                #continue#skip iteration
            #test images - this is where different strategies may come in
            #get first image, can also use the list for this
            #get info for test images
            Test_Image_name=MatchImages.ImagesInMem_Pairing[TestImageList][1].FirstImage
            Test_Image_Histo=MatchImages.ImagesInMem_to_Process[Test_Image_name].Histogram
            Test_Image_FMatches=MatchImages.ImagesInMem_to_Process[Test_Image_name].FM_Keypoints
            Test_Image_Descrips=MatchImages.ImagesInMem_to_Process[Test_Image_name].FM_Descriptors
            Test_Image_FourierMag=MatchImages.ImagesInMem_to_Process[Test_Image_name].FourierTransform_mag

            HistogramSimilarity=CompareHistograms(Base_Image_Histo,Test_Image_Histo)
            CheckImages_InfoSheet.AllHisto_results.append(HistogramSimilarity)

            UnsortedMatches=_3DVisLabLib.Orb_FeatureMatch(Base_Image_FMatches,Base_Image_Descrips,Test_Image_FMatches,Test_Image_Descrips,99999)
            AverageMatchDistance=GetAverageOfMatches(UnsortedMatches,MatchImages.AverageDistanceFM)
            CheckImages_InfoSheet.All_FM_results.append(AverageMatchDistance)

            #get differnce between fourier magnitudes of image
            FourierDifference=(abs(Base_Image_FourierMag-Test_Image_FourierMag)).sum()

            HM_data_histo[BaseImageList,TestImageList]=HistogramSimilarity
            HM_data_FM[BaseImageList,TestImageList]=AverageMatchDistance
            HM_data_FourierDifference[BaseImageList,TestImageList]=FourierDifference
            #data is symmetrical - fill it in to help with visualisation
            HM_data_histo[TestImageList,BaseImageList]=HistogramSimilarity
            HM_data_FM[TestImageList,BaseImageList]=AverageMatchDistance
            HM_data_FourierDifference[TestImageList,BaseImageList]=FourierDifference
            

            if HistogramSimilarity<CheckImages_InfoSheet.BestMatch_Histo:
                CheckImages_InfoSheet.BestMatch_Histo=HistogramSimilarity
                CheckImages_InfoSheet.BestMatch_Histo_listIndex=TestImageList

            if AverageMatchDistance<CheckImages_InfoSheet.BestMatch_FeatureMatch:
                CheckImages_InfoSheet.BestMatch_FeatureMatch=AverageMatchDistance
                CheckImages_InfoSheet.BestMatch_FeatureMatch_listIndex=TestImageList

        #after check all images, if a result then copy that list into the first list so combine the sets of images
        if (len(CheckImages_InfoSheet.AllHisto_results)>0) and (True==False):
            #list of images now inactive as will be copied to another
            #[0] here is the list of images, while [1] is the info card
            #TempUpdatedList=MatchImages.ImagesInMem_Pairing[BaseImageList]
            #del MatchImages.ImagesInMem_Pairing[BaseImageList]
            MatchImages.ImagesInMem_Pairing[BaseImageList]=(MatchImages.ImagesInMem_Pairing[BaseImageList][0]+MatchImages.ImagesInMem_Pairing[CheckImages_InfoSheet.BestMatch_Histo_listIndex][0],MatchImages.ImagesInMem_Pairing[BaseImageList][1])
            MatchImages.ImagesInMem_Pairing[CheckImages_InfoSheet.BestMatch_Histo_listIndex][1].InUse=False
            OutOfUse=OutOfUse+1
            #MatchImages.ImagesInMem_Pairing[BaseImageList][0]=MatchImages.ImagesInMem_Pairing[BaseImageList][0]+MatchImages.ImagesInMem_Pairing[CheckImages_InfoSheet.BestMatch_Histo_listIndex][0]
        if len(CheckImages_InfoSheet.All_FM_results)>0 and (True==False):
            MatchImages.ImagesInMem_Pairing[BaseImageList]=(MatchImages.ImagesInMem_Pairing[BaseImageList][0]+MatchImages.ImagesInMem_Pairing[CheckImages_InfoSheet.BestMatch_FeatureMatch_listIndex][0],MatchImages.ImagesInMem_Pairing[BaseImageList][1])
            MatchImages.ImagesInMem_Pairing[CheckImages_InfoSheet.BestMatch_FeatureMatch_listIndex][1].InUse=False
            OutOfUse=OutOfUse+1


    BlankOut=HM_data_FourierDifference.max()
    #blank out the self test
    for item in MatchImages.ImagesInMem_Pairing:
        HM_data_FourierDifference[item,item]=BlankOut

    #normalise matrices
    HM_data_FM=normalize_2d(HM_data_FM)
    HM_data_histo=normalize_2d(HM_data_histo)
    HM_data_FourierDifference=normalize_2d(HM_data_FourierDifference)
    HM_data_Both=normalize_2d(HM_data_histo+HM_data_FM+HM_data_FourierDifference)
    #HM_data_Both=normalize_2d(HM_data_FourierDifference)

    #if have equal length for both results, asssume they are aligned - can examine response
    if len(CheckImages_InfoSheet.All_FM_results)==len(CheckImages_InfoSheet.AllHisto_results):
        FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_Both") +".jpg"
        PlotAndSave_2datas("HM_data_Both",FilePath,HM_data_Both)
        FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_FM") +".jpg"
        PlotAndSave_2datas("HM_data_FM",FilePath,HM_data_FM)
        FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_histo") +".jpg"
        PlotAndSave_2datas("HM_data_histo",FilePath,HM_data_histo)
        FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_FourierDifference") +".jpg"
        PlotAndSave_2datas("HM_data_FourierDifference",FilePath,HM_data_FourierDifference)

    
        #for every image or subsets of images, roll through heatmap finding nearest best match then
        #cross referencing it
        OrderedImages=dict()
        BaseImageList=random.choice(list(MatchImages.ImagesInMem_Pairing.keys()))

        #get minimum 
        result = np.where(HM_data_Both == np.amin(HM_data_Both))
        Element=random.choice(result[0])#incase we have two identical results


        BlankOut=HM_data_Both.max()+1
        #blank out the self test
        for item in MatchImages.ImagesInMem_Pairing:
            HM_data_Both[item,item]=BlankOut

        print(HM_data_Both)
        print("-----")
        BaseImageList=random.choice(list(MatchImages.ImagesInMem_Pairing.keys()))
        Counter=0
        MatchMetric_all=[]
        MatchMetric_Histo=[]
        MatchMetric_Fourier=[]
        MatchMetric_FM=[]
        while len(OrderedImages)+1<len(MatchImages.ImagesInMem_Pairing):

            Counter=Counter+1
            print("looking at row",BaseImageList,"for match for for")
            #HM_data_Both[BaseImageList,BaseImageList]=BlankOut
            Row=HM_data_Both[0:len(MatchImages.ImagesInMem_Pairing),BaseImageList]
            #print(Row)
            #get minimum value
            result = np.where(Row == np.amin(Row))
            Element=random.choice(result[0])#incase we have two identical results
            print("nearest matching is element",Element)
            print("nearest value",HM_data_Both[Element,BaseImageList])
            MatchMetric_all.append(HM_data_Both[Element,BaseImageList])
            MatchMetric_Histo.append(HM_data_histo[Element,BaseImageList])
            MatchMetric_Fourier.append(HM_data_FourierDifference[Element,BaseImageList])
            MatchMetric_FM.append(HM_data_FM[Element,BaseImageList])
            #add to output images
            

            for imgIndex, Images in enumerate (MatchImages.ImagesInMem_Pairing[Element][0]):
                #if len(Images)>1:
                    #raise Exception("too many images")
                SplitImagePath=Images.split("\\")[-1]
                FilePath=MatchImages.OutputPairs +"\\00" +str(Counter)+ "_ImgNo_" + str(BaseImageList) + "_" + SplitImagePath
                cv2.imwrite(FilePath,MatchImages.ImagesInMem_to_Process[Images].ImageColour)
                if Images in OrderedImages:
                    raise Exception("output images file already exists!!! logic error")
                else:
                    OrderedImages[Images]=BaseImageList
            #blank out element in both places
            HM_data_Both[0:len(MatchImages.ImagesInMem_Pairing),BaseImageList]=BlankOut
            HM_data_Both[BaseImageList,0:len(MatchImages.ImagesInMem_Pairing)]=BlankOut
            if Counter==1:
                HM_data_Both[0:len(MatchImages.ImagesInMem_Pairing),Element]=BlankOut
                HM_data_Both[Element,0:len(MatchImages.ImagesInMem_Pairing)]=BlankOut
            #baseimage should be an integer
            #work in columns to find nearest match, data should be mirrored diagonally to make it easier to visualise#
            
            
            #print(HM_data_Both)
            #print("-----")
            
            
            
            BaseImageList=Element

            
            
            
        PlotAndSave("MatchMetric_all",MatchImages.OutputPairs +"\\MatchMetric_all.jpg",MatchMetric_all,1)
        PlotAndSave("MatchMetric_Fourier",MatchImages.OutputPairs +"\\MatchMetric_Fourier.jpg",MatchMetric_Fourier,1)
        PlotAndSave("MatchMetric_FM",MatchImages.OutputPairs +"\\MatchMetric_FM.jpg",MatchMetric_FM,1)
        PlotAndSave("MatchMetric_Histo",MatchImages.OutputPairs +"\\MatchMetric_Histo.jpg",MatchMetric_Histo,1)


            

    MatchImages.Endtime= time.time()
    print("time taken (hrs):",round((MatchImages.Endtime- MatchImages.startTime)/60/60,2))
    exit()


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
        self.BestMatch_FeatureMatch=99999999
        self.BestMatch_FeatureMatch_listIndex=None


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