import datetime
#from cmath import nan
#from difflib import Match
#from logging import raiseExceptions
#from ssl import SSL_ERROR_SSL
from tkinter import Y
import enum
from cv2 import sqrt
import _3DVisLabLib
import json
import cv2
import random
from statistics import mean 
import copy
import random
import time
import multiprocessing
import statistics
import scipy
import math
import numpy as np
from scipy import stats
#from scipy.stats import skew
import gc
import MatchImages_lib
import copyreg#need this to pickle keypoints
#gc.disable()
import psutil
import os
#from sklearn.decomposition import PCA

#stuff for HOG
#from skimage.io import Ski_imread, ski_imshow
#from skimage.transform import ski_resize
#from skimage.feature import ski_hog
#from skimage import ski_exposure


import matplotlib
#matplotlib.use('Agg')#can get "Tcl_AsyncDelete: async handler deleted by the wrong thread" crashes otherwise
import matplotlib.pyplot as plt

class MatchImagesObject():

    def __init__(self):
        # USER OPTIONS
        ##################################################
        ##set input folder here
        ##################################################
        #self.InputFolder=r"C:\Working\TempImages\Faces\img_align_celeba"
        #self.InputFolder=r"C:\Working\TempImages\Flowers"
        #self.InputFolder = r"E:\NCR\TestImages\Furniture"
        #self.InputFolder=r"C:\Working\TempImages\TestMatches"
        #self.InputFolder=r"E:\NCR\TestImages\UK_SMall"
        #self.InputFolder=r"E:\NCR\TestImages\UK_1000"
        #self.InputFolder=r"C:\Working\TempImages\Furniture"
        #self.InputFolder=r"E:\NCR\TestImages\UK_Side_SMALL_15sets10"
        #self.InputFolder=r"E:\NCR\TestImages\UK_Side_SMALL_side_findmatchtest"
        #self.InputFolder=r"E:\NCR\TestImages\Randos"
        self.InputFolder=r"C:\Working\TempImages\butterflys"
        #self.InputFolder=r"E:\NCR\Currencies\Bangladesh_SR2800\Bangladesh\SR DC\MM8\05_1000(2020)"
        #self.InputFolder=r"C:\Working\TempImages\Food\images"
        ##################################################
        ##set subset of data - will select random images
        ##if cross checking for similarity will be in O (n/2) time complexity
        ##################################################
        self.SubSetOfData = int(9999999)  # subset of data
        ################################################################
        ##select what function will be used, which will load image,crop,resize
        ##etc for all analytical procesess
        ###################################################################
        # images have to be prepared in application specific ways - choose function here - don't leave the "()"!!!!
        #also hvae option to use stacked image function
        self.PrepareImagesFunction = MatchImages_lib.PrepareImageMetrics_MultipleImgs
        self.PreviewImagePrep=False#preview image processing functions
        ######################################################################
        ##turn on and off which analytics to use, some will add noise rather
        ##than useful metrics depending on images to analyse
        ######################################################################
        # what metrics to use
        self.Use__FeatureMatch = False  # match detailed areas of image - quite slow
        self.Use__histogram = True  # match how close the image colour distribution is - structure does not matter
        self.Use__FourierDifference = False  # only useful if subjects are perfectly aligned (like MM side) - otherwise will be noise
        self.Use__PhaseCorrelation = False  # not developed yet - can check 1d or 2d signal for X/Y movement (but not rotation).
        #in theory can convert for instance fourier magnitude image, polar unwrap it and check using phase correlation - but has to be proven
        self.Use__HOG_featureMatch = True  # dense feature match - good for small images - very effective for mm side
        self.Use__EigenVectorDotProd = False  #SQUARE IMAGE ONLY how close are principle components orientated- doesnt seem to work correctly yet - MUST BE SQUARE!
        self.Use__EigenValueDifference = False  #SQUARE IMAGE ONLY how close are principle component lengths for COLOUR - works pretty well - still needs a look at in how to package up matrix, and if using non -square do we use SVD instead?
        self.Use__FourierPowerDensity = True  # histogram of frequencies found in image - works very well
        self.Use__MacroStructure=False#very small image to compare macrostructure - experimental
        self.Use__StructuralPCA_dotProd=False#Principle component analysis on binarised image - a geometrical PCA
        self.Use__StructuralPCA_VectorValue = False  # for STRUCTURE Principle component analysis on binarised image - a geometrical PCA
        self.Use__TemplateMatching = False # match templates by sliding one over the other - can be different sizes so bear that in mind
        self.Use__HistogramStriping = True#to handle images with aspect ratio changes or scaling, uses a spherical mask
        self.Use__HistogramCentralis = True#create a histogram mask so just look at centre of image - not good for images with translation problem but may be effective for general image matching
        #self.Use__QuiltScan==True
        ######################################################
        ##set multi process behaviour - can force no threading if memory issues are encountered (imgs > 3000)
        #######################################################
        # set this to "1" to force inline processing, otherwise to limit cores set to the cores you wish to use then add one (as system will remove one for safety regardless)
        self.MemoryError_ReduceLoad = (True,11)  # fix memory errors (multiprocess makes copies of everything) (Activation,N+1 cores to use -EG use 4 cores = (True,5))
        self.BeastMode = False  # Beast mode will optimise processing and give speed boost - but won't be able to update user with estimated time left
        # self.OutputImageOrganisation=self.ProcessTerms.Sequential.value
        self.HyperThreading = True  # Experimental - hyperthreads are virtual cores so may not work for metrics like Feature Matching (except HOG)
        #but its not possible to predict how windows will distribute processes to which cores

        #***This option very slow still under development***#currently 150 images vs 12 input match takes 4min with on-fly but 12 seconds pre processed
        self.ProcessImagesOnFly=True#currently only works with MULTIPLE IMAGE function - if using a large amount of images
        #setting this to true will not load image processing details into memory but process on the fly, this
        #is much slower and much less efficient but is currently only solution to large datasets. Parallel processing may
        #mitigate some slowness
        
        #set buffer for memory that should remain - this is not definitive as its not possible to get an accurate size of the python process
        self.FreeMemoryBuffer_pc = 30  # how much memory % should be reserved while processing
        ##################################################
        ## input image matching mode - for ecah image in
        ## input image folder will try and find top 20
        ## matches from images in main input folder.
        ## will be in ON time complexity
        ##################################################
        #self.MatchFindFolder = r"C:\Working\TempImages\Faces\MatcherFolder"
        #self.MatchFindFolder = r"E:\NCR\TestImages\UK_Side_Small_15sets10_findmatch"
        self.MatchFindFolder = r"C:\Working\TempImages\MatchPerson2\cropped"
        self.MatchInputSet = False  # if a list of input images are provided the system will find similarities only with them, rather than
        # attempt to match every image sequentially.



        #END USER OPTIONS
        self.Outputfolder = r"E:\NCR\TestImages\MatchOutput"
        self.TraceExtractedImg_to_DatRecord = "TraceImg_to_DatRecord.json"
        self.OutputPairs = self.Outputfolder + "\\Pairs\\"
        self.OutputDuplicates = self.Outputfolder + "\\Duplicates\\"
        self.ImagesInMem_Pairing = dict()
        self.GetDuplicates = False
        self.startTime = None
        self.Endtime = None
        self.HM_data_MetricDistances = None
        self.HM_data_MetricDistances_auto = None
        self.DummyMinValue = -9999923
        self.DummyMaxValue=9999999999
        self.MetricDictionary = dict()
        self.TraceExtractedImg_to_DatRecordObj = None
        self.ImagesInMem_to_Process = dict()
        self.DuplicatesToCheck = dict()
        self.DuplicatesFound = []
        self.Mean_Std_Per_cyclelist = None
        self.HistogramSelfSimilarityThreshold = 0.005  # should be zero but incase there is image compression noise
        self.CurrentBaseImage = None
        self.List_ImagesToMatchFIlenames = dict()
        self.MetricsFunctions=dict()
        self.ImageProcessingFunction_SaveImages=False
        # populatemetricDictionary
        self.Metrics_dict = dict()
        self.Metrics_functions = dict()

        if self.Use__FeatureMatch: 
            self.Metrics_dict["HM_data_FM"] = None  # slow - maybe cant be hyperthreaded?
            self.Metrics_functions["HM_data_FM"] = None 
        if self.Use__histogram:
            self.Metrics_dict["HM_data_histo"] = None  # fast =no idea about hyperthreading
            self.Metrics_functions["HM_data_histo"] = None
        if self.Use__FourierDifference: 
            self.Metrics_dict["HM_data_FourierDifference"] = None  # fast -hyperthreading yes
            self.Metrics_functions["HM_data_FourierDifference"] = None
        if self.Use__PhaseCorrelation:
            self.Metrics_dict["HM_data_PhaseCorrelation"] = None  # -hyperthreading yes
            self.Metrics_functions["HM_data_PhaseCorrelation"] = None
        if self.Use__HOG_featureMatch: 
            self.Metrics_dict["HM_data_HOG_Dist"] = None  # slow -hyperthreading yes
            self.Metrics_functions["HM_data_HOG_Dist"] = None
        if self.Use__EigenVectorDotProd: 
            self.Metrics_dict["HM_data_EigenVectorDotProd"] = None  # fast
            self.Metrics_functions["HM_data_EigenVectorDotProd"] = None
        if self.Use__EigenValueDifference:
            self.Metrics_dict["HM_data_EigenValueDifference"] = None  # fast #-hyperthreading yes
            self.Metrics_functions["HM_data_EigenValueDifference"] = None
        if self.Use__FourierPowerDensity:
            self.Metrics_dict["HM_data_FourierPowerDensity"] = None  # fast #-hyperthreading yes
            self.Metrics_functions["HM_data_FourierPowerDensity"] = None
        if self.Use__MacroStructure: 
            self.Metrics_dict["HM_data_MacroStructure"] = None
            self.Metrics_functions["HM_data_MacroStructure"] = None
        if self.Use__StructuralPCA_dotProd: 
            self.Metrics_dict["HM_data_StructuralPCA_dotProd"] = None
            self.Metrics_functions["HM_data_StructuralPCA_dotProd"] = None
        if self.Use__StructuralPCA_VectorValue: 
            self.Metrics_dict["HM_data_StructuralPCA_VectorValue"] = None
            self.Metrics_functions["HM_data_StructuralPCA_VectorValue"] = None
        if self.Use__TemplateMatching: 
            self.Metrics_dict["HM_data_TemplateMatching"] = None
            self.Metrics_functions["HM_data_TemplateMatching"] = None
        if self.Use__HistogramStriping:
            self.Metrics_dict["HM_data_HistogramStriping"] = None
            self.Metrics_functions["HM_data_HistogramStriping"] = None
        if self.Use__HistogramCentralis:
            self.Metrics_dict["HM_data_HistogramCentralis"] = None
            self.Metrics_functions["HM_data_HistogramCentralis"] = None
        #if self.Use__QuiltScan: self.Metrics_dict["HM_data_QuiltScan"] = None

        for metrix in self.Metrics_dict:
            print("Using metrics",metrix)

    def ForceNormalise_forHOG(self,image1):
        #TODO cannot figure out why normal normalising makes this (visibly) normalised
        #while other methods leave it blown out - it may not be significant statistically but
        #it would help with visualising the process 
        image1_Norm = cv2.normalize(image1, image1,0, 255, cv2.NORM_MINMAX)
        image2_Norm = cv2.normalize(image1,image1, 0, 255, cv2.NORM_MINMAX)
        # black blank image, hieght of both images and width of widest image
        blank_image = np.zeros(shape=[image1.shape[0]+image1.shape[0], max(image1.shape[1],image1.shape[1]), image1.shape[2]], dtype=np.uint8)
        #drop in images
        blank_image[0:image1_Norm.shape[0],0:image1_Norm.shape[1],:]=image1_Norm[:,:,:]
        blank_image[image1_Norm.shape[0]:,0:image2_Norm.shape[1]:,:]=image2_Norm[:,:,:]
        blank_image=blank_image[0:image1_Norm.shape[0],0:image1_Norm.shape[1],:]
        #_3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(blank_image,(blank_image.shape[1]*3,blank_image.shape[0]*3)),0,True,True)
        return blank_image

    def StackTwoimages(self,image1,image2):
        #if grayscale-convert to colour
        if len(image1.shape)!=3:
            image1 = cv2.cvtColor(image1,cv2.COLOR_GRAY2RGB)
        if len(image2.shape)!=3:
            image2 = cv2.cvtColor(image2,cv2.COLOR_GRAY2RGB)
        image1_Norm = cv2.normalize(image1, image1,0, 255, cv2.NORM_MINMAX)
        image2_Norm = cv2.normalize(image2,image2, 0, 255, cv2.NORM_MINMAX)
        # black blank image, hieght of both images and width of widest image
        blank_image = np.zeros(shape=[image1.shape[0]+image2.shape[0], max(image1.shape[1],image2.shape[1]), image2.shape[2]], dtype=np.uint8)
        #drop in images
        blank_image[0:image1_Norm.shape[0],0:image1_Norm.shape[1],:]=image1_Norm[:,:,:]
        blank_image[image1_Norm.shape[0]:,0:image2_Norm.shape[1]:,:]=image2_Norm[:,:,:]
        #_3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(blank_image,(blank_image.shape[1]*3,blank_image.shape[0]*3)),0,True,True)
        return blank_image

    # def USERFunction_PrepareForHOG(self,image,TargetHeight_HOG=64,TargetWidth_HOG=128):

    #     #HOG expects 64 * 128
    #     #lets not chagne aspect ratio
    #     ImageHeight=image.shape[0]
    #     ImageWidth=image.shape[1]
    #     #set to target height then crop as needed
    #     Percent2match=TargetHeight_HOG/ImageHeight
    #     TargetWidth=round(ImageWidth*Percent2match)
    #     Resized=cv2.resize(image,(TargetWidth,TargetHeight_HOG))

    #     #create blank
    #     blank_image = np.zeros(shape=[TargetHeight_HOG, TargetWidth_HOG, image.shape[2]], dtype=np.uint8)

    #     #now crop to HOG shape
    #     HOG_specific_crop=Resized[:,0:TargetWidth_HOG,:]

    #     blank_image[0:HOG_specific_crop.shape[0],0:HOG_specific_crop.shape[1],:]=HOG_specific_crop

    #     #now need to flip on its side
    #     # rotate ccw
    #     Rotate=cv2.transpose(blank_image)
    #     Rotate=cv2.flip(Rotate,flipCode=0)
    #     return Rotate

    def USERFunction_CropForFM(self,image):
        return image
        #return image[:,0:151,:]
        #return image[0:int(image.shape[0]/1.1),0:int(image.shape[1]/1),:]
    def USERFunction_CropForHistogram(self,image):
        return image
        #return image[:,0:151,:]
    def USERFunction_OriginalImage(self,image):
        return image
        return cv2.resize(image.copy(),(500,250))

    
    def USERFunction_ResizePercent(self,image,Percent):
        Percent=Percent/100
        return cv2.resize(image.copy(),(int(image.shape[1]*Percent),int(image.shape[0]*Percent)))

    def USERFunction_Resize(self,image):

        #height/length
        #return image
        # if len(image.shape)!=3:
        # #return image
        #     return cv2.resize(image.copy(),(300,200))
        # else:
        #     return cv2.resize(image.copy(),(300,200))
        return image

        #usual for mm1 side
        #grayscale image
        if len(image.shape)!=3:
        #return image
            return image[0:62,0:130]
        else:
            return image[0:62,0:130,:]
            #image= image[303:800,400:1500,:]
            image=cv2.resize(image,(500,250))
            return image
        #this is pixels not percentage!
        #return image
        #return cv2.resize(image.copy(),(300,200))

    class ProcessTerms(enum.Enum):
        Sequential="Sequential"
        DoubleUp="DoubleUp"

    """Class to hold information for image sorting & match process"""

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
            self.is_ImageToMatch=False
            #self.Histogram=[None]
            self.OriginalImage=None
            self.ImageColour=None
            self.ImageAdjusted=None
            #self.FM_Keypoints=[None]
            #self.FM_Descriptors=[None]
            #self.FourierTransform_mag=[None]
            #self.ImageGrayscale=[None]
            #self.EigenValues=[None]
            #self.EigenVectors=[None]
            #self.PrincpleComponents=[None]
            #self.HOG_Mag=[None]
            #self.HOG_Angle=[None]
            #self.OPENCV_hog_descriptor=[None]
            #self.PhaseCorrelate_FourierMagImg=[None]
            self.DebugImage=None
            self.OriginalImageFilePath=None
            #self.PwrSpectralDensity=[None]
            #self.MacroStructure_img=[None]
            #self.PCA_Struct_EigenVecs=[None]
            #self.PCA_Struct_EigenVals=[None]
            self.ProcessImages_function=None#if this is not none - should be a function we can call to process images on the fly
            self.IsInError=False
            self.Metrics_functions=dict()

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

def PlotAndSaveHistogram(Title,Filepath,Data,maximumvalue,bins):
    #(skew(x))
    try:
        plt.hist(Data, bins = bins)
        #plt.ylabel(Title)
        plt.xlim([0, 2])
        plt.savefig(Filepath)
        plt.cla()
        plt.clf()
        plt.close()
    except Exception as e:
        print("Error with matpyplot",e)


def HM_data_MetricDistances():
    print("plop")


def PlotAndSave_2datas(Title, Filepath, Data1):
    # this causes crashes
    # save out plot of 1D data
    try:
        # Data1=np.random.rand(30,22)
        plt.pcolormesh((Data1), cmap='autumn')
        # plt.plot(Data1,Data2,'bo')#bo will draw dots instead of connected line
        plt.ylabel(Title)
        # plt.ylim([0, max(Data1)])
        # plt.ylim([0, max(Data2)])
        plt.savefig(Filepath)
        plt.cla()
        plt.clf()
        plt.close()
    except Exception as e:
        print("Error with matpyplot", e)


def GetAverageOfMatches(List,Span):
    Spanlist=List[0:Span]
    Distances=[]
    for elem in Spanlist:
        Distances.append(elem.distance)
    return round(mean(Distances),2)

def normalize_2d(matrix):
    #check the matrix isnt all same number
    if matrix.min()==0 and matrix.max()==0:
        print("WARNING, matrix all zeros")
        return matrix
    if matrix.max()==matrix.min():
        print("WARNING: matrix homogenous")
        return np.ones(matrix.shape)
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))



def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)




def main():
    #create resource manager
    PrepareMatchImages=MatchImagesObject()
    #delete all files in folder
    print("Delete output folders:\n",PrepareMatchImages.Outputfolder)
    if _3DVisLabLib.yesno("?"):
        _3DVisLabLib.DeleteFiles_RecreateFolder(PrepareMatchImages.Outputfolder)
        #make subfolders
        _3DVisLabLib. MakeFolder( PrepareMatchImages.OutputPairs)
        _3DVisLabLib. MakeFolder(PrepareMatchImages.OutputDuplicates)


    #ask if user wants to check for duplicates
    print("Get duplicates only?? - this will be in ~o^(2/N) complexity time (very long)!!!")
    PrepareMatchImages.GetDuplicates= _3DVisLabLib.yesno("?")


    #get object that links images to dat records
    print("attempting to load image to dat record trace file",PrepareMatchImages.InputFolder + "\\" + PrepareMatchImages.TraceExtractedImg_to_DatRecord)
    try:
        Json_to_load=PrepareMatchImages.InputFolder + "\\" + PrepareMatchImages.TraceExtractedImg_to_DatRecord
        with open(Json_to_load) as json_file:
            PrepareMatchImages.TraceExtractedImg_to_DatRecordObj = json.load(json_file)
            print("json loaded succesfully")
            PrepareMatchImages.TraceExtractedImg_to_DatRecordObj
    except Exception as e:
        print("JSON_Open error attempting to open json file " + str(PrepareMatchImages.InputFolder + "\\" + PrepareMatchImages.TraceExtractedImg_to_DatRecord) + " " + str(e))
        if _3DVisLabLib.yesno("Continue operation? No image to dat record trace will be possible so only image matching & sorting")==False:
            raise Exception("User declined to continue after JSOn image vs dat record file not found")





    print("filtering nested images from", PrepareMatchImages.InputFolder,"this can several minutes if N>10,000")
    #get all files in input folder
    InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(PrepareMatchImages.InputFolder)

    #filter out non images
    ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)

    #load images in random order for testing
    randomdict=dict()
    for Index, ImagePath in enumerate(ListAllImages):
        randomdict[ImagePath]=Index

    #user may have specified a subset of data
    RandomOrder=[]
    while (len(RandomOrder)<PrepareMatchImages.SubSetOfData) and (len(randomdict)>0):
        randomchoice_img=random.choice(list(randomdict.keys()))
        RandomOrder.append(randomchoice_img)
        del randomdict[randomchoice_img]
    print("User image reduction, operating on with",len(RandomOrder)," from ",len(ListAllImages),"images")



     #in this mode we have some input images (part of the input dataset) that we have to match other images to
    if PrepareMatchImages.MatchInputSet==True:
        print("Match finding mode")
        #get all files in input folder
        InputFiles_match=_3DVisLabLib.GetAllFilesInFolder_Recursive(PrepareMatchImages.MatchFindFolder)
        #filter out non images
        ListAllImages_match=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles_match)
        print(len(ListAllImages_match),"found in match finding folder")
        #check through main folder of images to see if already in there - if not - add - bear in mind may have different filenames
        PrepareMatchImages.List_ImagesToMatchFIlenames=dict()
        #for images we want to match - get their filenames only sans path and place into a dictionary vs full filename and path
        for FindMatchImage in ListAllImages_match:
            FileName=FindMatchImage.split("\\")[-1]
            PrepareMatchImages.List_ImagesToMatchFIlenames[FileName]=FindMatchImage
            RandomOrder.append(FindMatchImage)


        # #for each of these filenames without paths, check if exist in the main folder of images, if so - add them to list
        # for Image2Match in PrepareMatchImages.List_ImagesToMatchFIlenames:
        #     ImageFoundInDataset=False
        #     for FindMatchImage in RandomOrder:
        #         FileName=FindMatchImage.split("\\")[-1]
        #         if FileName==Image2Match:
        #             ImageFoundInDataset=True
        #             break
        #     if ImageFoundInDataset==False:
        #         RandomOrder.append(PrepareMatchImages.List_ImagesToMatchFIlenames[Image2Match])

                



    #start timer
    PrepareMatchImages.startTime=time.time()


    #populate images 
    #load images into memory

    
    #record images to allow us to debug code
    ImageReviewDict=dict()
    for Index, ImagePath in enumerate(RandomOrder):
        if Index%30==0: print("Image loadcheck",Index,"/",len(RandomOrder))


        #try:
        #test that image is valid first
        if MatchImages_lib.TestImage(ImagePath):
            
            ImageInfo=PrepareMatchImages.PrepareImagesFunction(PrepareMatchImages,ImagePath,Index,ImageReviewDict)

            #populate dictionary
            if ImageInfo is not None:
                PrepareMatchImages.ImagesInMem_to_Process[ImagePath]=(ImageInfo)

                #save out images for inspection
                if Index==0:
                    pass
                    #PrepareMatchImages.ImageProcessingFunction_SaveImages=True
                    #ImageInfo=PrepareMatchImages.PrepareImagesFunction(PrepareMatchImages,ImagePath,Index,ImageReviewDict)
                    #if ImageInfo.ProcessImages_function is not None:
                    #    PrepareMatchImages.ImageProcessingFunction_SaveImages=False


        #except:
         #   print("error with image, skipping",ImagePath)

    #need this to copy the keypoints for some reason - incompatible with pickle which means
    #any multiprocessing wont work either
    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)
    #need this test to see if we can pickle the object - if we cant then multiprocessing wont work
    MatchImages=copy.deepcopy(PrepareMatchImages)
    PrepareMatchImages=None#clean this out incase garbage collector doesnt 

    #build dictionary to remove items from
    for Index, img in enumerate(MatchImages.ImagesInMem_to_Process):
        MatchImages.DuplicatesToCheck[img]=img
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
                HistogramSimilarity=MatchImages_lib. CompareHistograms(BaseImage.Histogram,TestImage.Histogram)
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

    #in this mode we have some input images (part of the input dataset) that we have to match other images to
    if MatchImages.MatchInputSet==True:#List_ImagesToMatchFIlenames
        #create indexed dictionary of images so we can start combining lists of images
        MatchImages.ImagesInMem_Pairing=dict()
        #add images from list as first images so we can stop processing similarity
        FirstIndex=0
        for FirstIndex, Image2match in enumerate(MatchImages.List_ImagesToMatchFIlenames):
           ImgCol_InfoSheet=ImgCol_InfoSheet_Class()
           ImgCol_InfoSheet.FirstImage=MatchImages.List_ImagesToMatchFIlenames[Image2match]
           MatchImages.ImagesInMem_Pairing[FirstIndex]=([MatchImages.List_ImagesToMatchFIlenames[Image2match]],ImgCol_InfoSheet)
           if MatchImages.List_ImagesToMatchFIlenames[Image2match] not in MatchImages.ImagesInMem_to_Process:
               raise Exception("images to match not found in images in memory for processing object")
        for Index, img in enumerate(MatchImages.ImagesInMem_to_Process):
            # OktoAdd=True
            # if Index>FirstIndex:#add after the images to match
            #     for Image2Match in MatchImages.List_ImagesToMatchFIlenames:
            #         #dont add image to match twice
            #         if Image2Match.split("\\")[-1]==img.split("\\")[-1]:
            #             OktoAdd=False
            #     if OktoAdd==True:
            ImgCol_InfoSheet=ImgCol_InfoSheet_Class()
            ImgCol_InfoSheet.FirstImage=img
            MatchImages.ImagesInMem_Pairing[Index+FirstIndex+1]=([img],ImgCol_InfoSheet)
                            

    else:
        #create indexed dictionary of images so we can start combining lists of images
        MatchImages.ImagesInMem_Pairing=dict()
        for Index, img in enumerate(MatchImages.ImagesInMem_to_Process):
            ImgCol_InfoSheet=ImgCol_InfoSheet_Class()
            ImgCol_InfoSheet.FirstImage=img
            MatchImages.ImagesInMem_Pairing[Index]=([img],ImgCol_InfoSheet)




    #initialise metric matrices
    #previously would create an N by N matrix as expects to cross-check
    #all images - lets try and add logic so if we are testing a small
    #subset of images VS a large dataset (16 vs 100,000), we dont
    #run into memory problems with the default N by N matrix size
    if MatchImages.MatchInputSet==True:
        MetricsMatrices_height=len(MatchImages.List_ImagesToMatchFIlenames)
    else:
        MetricsMatrices_height=len(MatchImages.ImagesInMem_Pairing)
    
    MatchImages.HM_data_MetricDistances = np.zeros((MetricsMatrices_height,len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_MetricDistances_auto = np.zeros((MetricsMatrices_height,len(MatchImages.ImagesInMem_Pairing)))
    #initialise all metric matrices
    for MetricsMatrix in MatchImages.Metrics_dict:
        MatchImages.Metrics_dict[MetricsMatrix]=np.zeros((MetricsMatrices_height,len(MatchImages.ImagesInMem_Pairing)))


    #get info about cores for setting up multiprocess
    #WARNING might give weird results if inside a VM or a executable converted python 
    PhysicalCores=psutil.cpu_count(logical=False)#number of physical cores
    HyperThreadedCores = int(os.environ['NUMBER_OF_PROCESSORS'])#don't use these simulated cores
    #conditional user options
    if MatchImages.HyperThreading==True:
        print("Hyperthreading cores will be used (User option)")
        #user wants to use virtual cores - this might not be compatible with all similarity metrics
        Cores_Available=HyperThreadedCores
    else:
        print("Physical cores only will be used (User option)")
        #user just wants physical cores - we still dont know how windows distributes cores so may be virtual anyway
        Cores_Available=PhysicalCores
    #final core count available
    CoresTouse=1
    #user may have restricted performance to overcome memory errors or to leave system capacity for other tasks
    if MatchImages.MemoryError_ReduceLoad[0]==True and Cores_Available>1:
        CoresTouse=min(Cores_Available,MatchImages.MemoryError_ReduceLoad[1])#if user has over-specified cores restrict to cores available
        print("THROTTLING BY USER - Memory protection: restricting cores to", CoresTouse, "or less, user option MemoryError_ReduceLoad")
    else:
        CoresTouse=Cores_Available
    #if no restriction by user , leave a core anyway
    processes=max(CoresTouse-1,1)#rule is thumb is to use number of logical cores minus 1, but always make sure this number >0. Its not a good idea to blast CPU at 100% as this can reduce performance as OS tries to balance the load
    
    #find how much memory single process uses (windows)
    Currentprocess = psutil.Process(os.getpid())
    SingleProcess_Memory=Currentprocess.memory_percent()
    SystemMemoryUsed=psutil.virtual_memory().percent
    FreeMemoryBuffer_pc=MatchImages.FreeMemoryBuffer_pc#arbitrary free memory to leave
    MaxPossibleProcesses=max(math.floor((100-FreeMemoryBuffer_pc-SystemMemoryUsed)/SingleProcess_Memory),0)#can't be lower than zero
    print(MaxPossibleProcesses,"parallel processes possible at system capacity (leaving",FreeMemoryBuffer_pc,"% memory free)")
    print("Each process will use ~ ",round(psutil.virtual_memory().total*SingleProcess_Memory/100000000000,1),"gb")#convert bytes to gb
    #cannot proceed if we can't use even one core
    if processes<1 or MaxPossibleProcesses<1:
        print(("Multiprocess Configuration error! Less than 1 process possible - memory or logic error"))
        processes=1
        MaxPossibleProcesses=1
        print("Forcing processors =1, may cause memory error")
        #raise Exception("Multiprocess Configuration error! Less than 1 process possible - memory or logic error")
    #check system has enough memory - if not restrict cores used
    if processes>MaxPossibleProcesses:
        print("WARNING!! possible memory overflow - restricting number of processes from",processes,"to",MaxPossibleProcesses)
        processes=MaxPossibleProcesses
    #user option for process boost

    if MatchImages.MatchInputSet==True:
        ThreadsNeeded=len(MatchImages.List_ImagesToMatchFIlenames)
        chunksize=1#in this mode we are generally running a low # of threads VS a large amount of images
        
    else:
        ThreadsNeeded=len(MatchImages.ImagesInMem_Pairing)
        chunksize=processes*3#arbitrary setting to be able to get time feedback and not clog up system - but each reset of tasks can have memory overhead


    #user option to speed up process by stacking tasks with no breaks - warning - might need maxpossibletask settings in the pool 
    if MatchImages.BeastMode==True:
        print("WARNING! Beast mode active - this optimisation prohibits any timing feedback")
        chunksize=int(ThreadsNeeded/processes)

    #how many jobs do we build up to pass off to the multiprocess pool, in this case in theory each core gets 3 stacked tasks
    ProcessesPerCycle=processes*chunksize
    


    
    #experimental with optimisation for multiprocessing
    #currently multiprocesses have staggered finish due to allotment of jobs
    #in this manner we wont have a situation where the first thread has the bigger range of the factorial jobs
    #and all processes more or less have some workload - this can be done more systematically but this
    #gets us most the way there without fiddly code
    print("Optimising parallel process load balance for",processes,"cores")

    #different strategies if we are matching images from input folder - dont need to load balance as will have much smaller subset of
    #images to test
    if MatchImages.MatchInputSet==False:
        SacrificialDictionary=copy.deepcopy(MatchImages.ImagesInMem_Pairing)
        ImagesInMem_Pairing_ForThreading=dict()
        while len(SacrificialDictionary)>0:
            RandomItem=random.choice(list(SacrificialDictionary.keys()))
            ImagesInMem_Pairing_ForThreading[RandomItem]=MatchImages.ImagesInMem_Pairing[RandomItem]
            del SacrificialDictionary[RandomItem]
    else:#
        ImagesInMem_Pairing_ForThreading=dict()
        for Indexer, Image2match in enumerate((MatchImages.List_ImagesToMatchFIlenames)):
            ImagesInMem_Pairing_ForThreading[Indexer]=MatchImages.ImagesInMem_Pairing[Indexer]
            
    
    #check that cores dont have more than 1 task stack in this mode- otherwise the process time will
    #be multiplied as excess tasks will run after the parallel processes
    if MatchImages.MatchInputSet==True:
        if MatchImages.SubSetOfData>1000:
            #if MatchImages.ProcessImagesOnFly==True
            if chunksize>1:
                print("\n\n\n")
                print("large amount of images to check for image matcher - please ensure IMAGE TO MATCH N does not exceed CORES N")
                print("As need one core per image to run through all test images, and if cannot do in parallel will multiply time taken by excess N")
                print("Information will follow to assess parallel configuration")
                result=input("any key to confirm warning")
    
    pool = multiprocessing.Pool(processes=processes)
    listJobs=[]
    CompletedProcesses=0
    #start timer and time metrics
    listTimings=[]
    listCounts=[]
    listAvgTime=[]
    ProcessOnly_start = time.perf_counter()
    t1_start = time.perf_counter()
    if processes>1:
        print("[Multiprocess start]","Taskstack per core:",chunksize,"  Taskpool size:",ProcessesPerCycle,"  Physical cores used:",processes,"   Image Threads:",ThreadsNeeded)
        for Index, BaseImageList in enumerate(ImagesInMem_Pairing_ForThreading):
            

            listJobs.append((MatchImages,BaseImageList))

            if (Index%ProcessesPerCycle==0 and Index!=0) or Index==len(ImagesInMem_Pairing_ForThreading)-1:#before was matches.imagepairinginmemory incase this breaks
                #maxtasksperchild can help with memory problems
                ReturnList=(pool.imap_unordered(MatchImages_lib.ProcessSimilarity,listJobs,chunksize=chunksize))
                #populate output metric comparison matrices
                for ReturnObjects in ReturnList:
                    if ReturnObjects is not None:
                        Rtrn_CurrentBaseImage=ReturnObjects["BASEIMAGE"]
                        for returnItem in MatchImages.Metrics_dict:
                            if returnItem in ReturnObjects:
                                MatchImages.Metrics_dict[returnItem][Rtrn_CurrentBaseImage,:]=ReturnObjects[returnItem]
                            else:
                                print("No match for",returnItem)

                CompletedProcesses=CompletedProcesses+len(listJobs)
                print("Jobs done:", CompletedProcesses, "/", len(MatchImages.ImagesInMem_Pairing))
                try:
                    #get timings
                    listTimings.append(round(time.perf_counter()-t1_start,2))
                    listCounts.append(len(listJobs))
                    listAvgTime.append(round(listTimings[-1]/listCounts[-1],3))
                    #start timer again
                    t1_start = time.perf_counter()

                    #clear list of jobs
                    listJobs=[]
                
                    if len(listTimings)>3:
                        Diffy=listTimings[-1]-listTimings[1]
                        Diffx=len(listTimings)-1
                        m=Diffy/Diffx
                        _C=listTimings[1]-(m*1)
                        #playing around with time estimation
                        TaskLots=(len(MatchImages.ImagesInMem_Pairing)/ProcessesPerCycle)
                        TimeEstimate_sum=[]
                        TimeEstimateLeft_sum=[]
                        TimeEstimate_sum.append(listTimings[0])#first one is always non uniform
                        CurrentPosition=round(Index/ProcessesPerCycle)
                        for counter in range (1,int(TaskLots)):
                            _Y=(m*counter)+_C
                            if counter<CurrentPosition:
                                TimeEstimate_sum.append(listTimings[counter])
                            if counter>=CurrentPosition:
                                TimeEstimate_sum.append(round(_Y,2))
                                TimeEstimateLeft_sum.append(round(_Y,2))
                        # if len(listTimings)<7:
                        #     print("(more samples needed) Estimated total time=",str(datetime.timedelta(seconds=sum(TimeEstimate_sum))))
                        #     print("(more samples needed) Estimated Time left=",str(datetime.timedelta(seconds=sum(TimeEstimateLeft_sum))))
                        # else:
                        #     print("Estimated total time=",str(datetime.timedelta(seconds=sum(TimeEstimate_sum))))
                        #     print("Estimated Time left=",str(datetime.timedelta(seconds=sum(TimeEstimateLeft_sum))))
                        #second time estimate
                        slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(listTimings[1:-1])),listTimings[1:-1])
                        TimeEstimate_sum=[]
                        TimeEstimateLeft_sum=[]
                        TimeEstimate_sum.append(listTimings[0])#first one is always non uniform
                        for counter in range (1,int(TaskLots)):
                            _Y=(slope*counter)+intercept
                            if counter<CurrentPosition:
                                TimeEstimate_sum.append(listTimings[counter])
                            if counter>=CurrentPosition:
                                TimeEstimate_sum.append(round(_Y,2))
                                TimeEstimateLeft_sum.append(round(_Y,2))
                        if len(listTimings)<7:
                            print("(More time samples needed) Estimated total time=",str(datetime.timedelta(seconds=sum(TimeEstimate_sum))))
                            print("(More time samples needed) Estimated Time left=",str(datetime.timedelta(seconds=sum(TimeEstimateLeft_sum))))
                        else:
                            print("Estimated total time=",str(datetime.timedelta(seconds=sum(TimeEstimate_sum))))
                            print("Estimated Time left=",str(datetime.timedelta(seconds=sum(TimeEstimateLeft_sum))))
                except:
                    print("Problem with time estimate code")



    if processes==1:
        print("[singleprocess start] inline process started:"),
        for Index, BaseImageList in enumerate(ImagesInMem_Pairing_ForThreading):
            print("starting",Index,"/",len(ImagesInMem_Pairing_ForThreading))
            ReturnObjects=MatchImages_lib.ProcessSimilarity((MatchImages,BaseImageList))
            if ReturnObjects is not None:
                Rtrn_CurrentBaseImage=ReturnObjects["BASEIMAGE"]
                for returnItem in MatchImages.Metrics_dict:
                    if returnItem in ReturnObjects:
                        MatchImages.Metrics_dict[returnItem][Rtrn_CurrentBaseImage,:]=ReturnObjects[returnItem]
                    else:
                        print("No match for",returnItem)

    
    print("Total process only time:",str(datetime.timedelta(seconds= time.perf_counter()-ProcessOnly_start)))
    print("Conforming data..")



    #create diagonally symetrical matrix
    for BaseImageList in MatchImages.ImagesInMem_Pairing:
        for testImageList in MatchImages.ImagesInMem_Pairing:
            #can either be checking each image against all others, or only a smaller subset of images against a main set
            if MatchImages.MatchInputSet == True:
                #dont need to make symettrical as the matrix will not be square to cope with large dataset
                #if (testImageList > len(MatchImages.List_ImagesToMatchFIlenames)) and (BaseImageList > len(MatchImages.List_ImagesToMatchFIlenames)):
                continue
            if testImageList<BaseImageList:
                for MatchMetric in MatchImages.Metrics_dict:
                    MatchImages.Metrics_dict[MatchMetric][BaseImageList,testImageList]=MatchImages.Metrics_dict[MatchMetric][testImageList,BaseImageList]

    #fix any bad numbers
    if MatchImages.MatchInputSet == False:
        #dealing with an N by N matrix
        for BaseImageList in MatchImages.ImagesInMem_Pairing:
            for testImageList in MatchImages.ImagesInMem_Pairing:
                for MatchMetric in MatchImages.Metrics_dict:
                    if math.isnan( MatchImages.Metrics_dict[MatchMetric][BaseImageList,testImageList]):
                        MatchImages.Metrics_dict[MatchMetric][BaseImageList,testImageList]=MatchImages.DummyMinValue
                        print("Bad value found, ",MatchMetric)
        #dealing with a rectangular matrix with a small height but potentially very long width

    if MatchImages.MatchInputSet == True:
        for OuterIndex in range(MatchImages.HM_data_MetricDistances.shape[0]):
            for InnerIndex in range(MatchImages.HM_data_MetricDistances.shape[1]):
                for MatchMetric in MatchImages.Metrics_dict:
                    if math.isnan( MatchImages.Metrics_dict[MatchMetric][OuterIndex,InnerIndex]):
                        MatchImages.Metrics_dict[MatchMetric][OuterIndex,InnerIndex]=MatchImages.DummyMinValue
                        print("Bad value found, ",MatchMetric)
                  



#normalize
    for MatchMetric in MatchImages.Metrics_dict:
        MatchImages.Metrics_dict[MatchMetric]=normalize_2d(np.where(MatchImages.Metrics_dict[MatchMetric]==MatchImages.DummyMinValue, MatchImages.Metrics_dict[MatchMetric].max()+1, MatchImages.Metrics_dict[MatchMetric]))

    if MatchImages.MatchInputSet == False:
        for BaseImageList in MatchImages.ImagesInMem_Pairing:
            for TestImageList in MatchImages.ImagesInMem_Pairing:
                if TestImageList<BaseImageList:
                    #data is diagonally symmetrical
                    continue
                #special case if matching input images - dont need to complete similarity matrix
                if MatchImages.MatchInputSet == True:
                    if (testImageList > len(MatchImages.List_ImagesToMatchFIlenames)) and (BaseImageList > len(MatchImages.List_ImagesToMatchFIlenames)):
                        continue
                #create total of all metrics
                BuildUpTotals=[]
                for MatchMetric in MatchImages.Metrics_dict:
                    #if math.isnan(MatchImages.Metrics_dict[MatchMetric][BaseImageList,TestImageList]):
                    BuildUpTotals.append(MatchImages.Metrics_dict[MatchMetric][BaseImageList,TestImageList])
                    #else:
                    #    BuildUpTotals.append(MatchImages.Metrics_dict[MatchMetric][BaseImageList,TestImageList])
                SquaredSum=0
                for Total in BuildUpTotals:
                    SquaredSum=SquaredSum+Total**2
                MatchImages.HM_data_MetricDistances[TestImageList,BaseImageList]=MatchImages.HM_data_MetricDistances[TestImageList,BaseImageList]+math.sqrt(SquaredSum)
                #mirror data for visualisation
                MatchImages.HM_data_MetricDistances[BaseImageList,TestImageList]=MatchImages.HM_data_MetricDistances[TestImageList,BaseImageList]
    else:

        for OuterIndex in range(MatchImages.HM_data_MetricDistances.shape[0]):
            for InnerIndex in range(MatchImages.HM_data_MetricDistances.shape[1]):
                #create total of all metrics
                BuildUpTotals=[]
                #for MatchMetric in MatchImages.Metrics_dict:
                for MatchMetric in MatchImages.Metrics_dict:
                    #if math.isnan(MatchImages.Metrics_dict[MatchMetric][BaseImageList,TestImageList]):
                    BuildUpTotals.append(MatchImages.Metrics_dict[MatchMetric][OuterIndex,InnerIndex])
                    #else:
                    #    BuildUpTotals.append(MatchImages.Metrics_dict[MatchMetric][BaseImageList,TestImageList])
                SquaredSum=0
                for Total in BuildUpTotals:
                    SquaredSum=SquaredSum+Total**2
                MatchImages.HM_data_MetricDistances[OuterIndex,InnerIndex]=MatchImages.HM_data_MetricDistances[OuterIndex,InnerIndex]+math.sqrt(SquaredSum)



    #normalise final data
    MatchImages.HM_data_MetricDistances=normalize_2d(MatchImages.HM_data_MetricDistances)


    #save data out as pickle/data

    
    MatchImages_lib.PrintResults(MatchImages,PlotAndSave_2datas,PlotAndSave)

    #match images to set of input images
    if MatchImages.MatchInputSet==True:
        MatchImages_lib.MatchImagestoInputImages(MatchImages,PlotAndSave_2datas,PlotAndSave)
        exit()
    else:
        #sequential matching
        MatchImages_lib.SequentialMatchingPerImage(MatchImages,PlotAndSave_2datas,PlotAndSave)

        #pairwise matching
        MatchImages_lib.PairWise_Matching(MatchImages,PlotAndSave_2datas,PlotAndSave,ImgCol_InfoSheet_Class)

        



class CheckImages_Class():
    def __init__(self):
        self.AllHisto_results=[]
        self.All_FM_results=[]
        self.All_EigenDotProd_result=[]
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
        self.StatsOfList=[]



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