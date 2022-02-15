
from logging import raiseExceptions
from ssl import SSL_ERROR_SSL
from tkinter import Y

from cv2 import sqrt
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
import math
import numpy as np
from scipy.stats import skew
import gc
import MatchImages_lib
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

    def USERFunction_CropForFM(self,image):
        #return image
        return image[0:106,0:189,:]
        #return image[0:int(image.shape[0]/1.1),0:int(image.shape[1]/1),:]
    def USERFunction_CropForHistogram(self,image):
        return image
        return image[80:150,60:140,:]
    def USERFunction_Resize(self,image):
        return image
        #this is pixels not percentage!
        #return image
        return cv2.resize(image.copy(),(200,200))

    """Class to hold information for image sorting & match process"""
    def __init__(self):
        self.InputFolder=r"E:\NCR\TestImages\TinyTest"
        self.Outputfolder=r"C:\Working\FindIMage_In_Dat\MatchImages"
        self.TraceExtractedImg_to_DatRecord="TraceImg_to_DatRecord.json"
        self.OutputPairs=self.Outputfolder + "\\Pairs\\"
        self.OutputDuplicates=self.Outputfolder + "\\Duplicates\\"
        self.AverageDistanceFM=100
        #self.Std_dv_Cutoff=0.5#set this to cut off how close notes can be in similarity
        self.TraceExtractedImg_to_DatRecordObj=None
        self.ImagesInMem_to_Process=dict()
        self.DuplicatesToCheck=dict()
        self.DuplicatesFound=[]
        self.Mean_Std_Per_cyclelist=None
        self.HistogramSelfSimilarityThreshold=0.005
        self.SubSetOfData=int(100)#subset of data
        #self.ImagesInMem_to_Process_Orphans=dict()#cant deepcopy feature match keypoints
        self.ImagesInMem_Pairing=dict()
        self.ImagesInMem_Pairing_orphans=dict()
        self.GetDuplicates=False
        self.startTime =None
        self.Endtime =None
        self.HM_data_MetricDistances=None
        self.HM_data_FM=None
        self.HM_data_histo=None
        self.HM_data_FourierDifference=None
        self.HM_data_EigenVectorDotProd=None
        self.HM_data_All=None
        
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
            self.EigenValues=None
            self.EigenVectors=None
            self.PrincpleComponents=None
            
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

    #load images in random order for testing
    randomdict=dict()
    for Index, ImagePath in enumerate(ListAllImages):
        randomdict[ImagePath]=Index

    #user may have specified a subset of data
    RandomOrder=[]
    while (len(RandomOrder)<MatchImages.SubSetOfData) and (len(randomdict)>0):
        randomchoice_img=random.choice(list(randomdict.keys()))
        RandomOrder.append(randomchoice_img)
        del randomdict[randomchoice_img]
        #if user is reducing data
        #for Subdivide in range (2,MatchImages.ReduceData):
        #    if len(randomdict)>0:
        #        randomchoice_img=random.choice(list(randomdict.keys()))
        #        del randomdict[randomchoice_img]
    print("User image reduction, operating on with",len(RandomOrder),"/",len(ListAllImages),"images")
    #populate images 
    #load images into memory
    for Index, ImagePath in enumerate(RandomOrder):
        print("Loading in image",ImagePath )
        ImageInfo=MatchImages.ImageInfo()

        Pod1Image_Grayscale = cv2.imread(ImagePath, cv2.IMREAD_GRAYSCALE)
        Pod1Image_col = cv2.imread(ImagePath)

        Pod1Image_Grayscale=MatchImages.USERFunction_Resize(Pod1Image_Grayscale)
        Pod1Image_col=MatchImages.USERFunction_Resize(Pod1Image_col)

        Pod1Image_col_cropped_FM=MatchImages.USERFunction_CropForFM(Pod1Image_col)
        Pod1Image_col_cropped_Histogram=MatchImages.USERFunction_CropForHistogram(Pod1Image_col)

        Pod1Image_col_cropped_PCA=Pod1Image_col_cropped_FM.copy()
        #get feature match keypoints
        keypoints,descriptor=_3DVisLabLib.OrbKeyPointsOnly(Pod1Image_col_cropped_FM,MatchImages.FeatureMatch_Dict_Common.ORB_default)
        #keypoints,descriptor=_3DVisLabLib.AkazeKeyPointsOnly(Pod1Image_col_cropped_FM)

        #get histogram for comparing colours
        hist = cv2.calcHist([Pod1Image_col_cropped_Histogram], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if Index==0:
            #on first loop show image to user
            DrawnKeypoints=_3DVisLabLib.draw_keypoints_v2(Pod1Image_col_cropped_FM.copy(),keypoints)
            print("original image")
            _3DVisLabLib.ImageViewer_Quick_no_resize(Pod1Image_col,0,True,True)
            print("cropped for feature matching")
            _3DVisLabLib.ImageViewer_Quick_no_resize(Pod1Image_col_cropped_FM,0,True,True)
            print("feature match sample")
            _3DVisLabLib.ImageViewer_Quick_no_resize(DrawnKeypoints,0,True,True)
            print("Crop for histogram")
            _3DVisLabLib.ImageViewer_Quick_no_resize(Pod1Image_col_cropped_Histogram,0,True,True)

        #get fourier transform
        #dft = cv2.dft(np.float32(Pod1Image_Grayscale),flags = cv2.DFT_COMPLEX_OUTPUT)
        #dft_shift = np.fft.fftshift(dft)
        #magnitude = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
        #product = 20*np.log(magnitude)
        f = np.fft.fft2(Pod1Image_Grayscale)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        #get eigenvectors and values for image
        #have to use grayscale at the moment
        #convert using opencv converter to comply with example code
        #https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html
        #ImageforPCA=cv2.cvtColor(Pod1Image_col, cv2.COLOR_BGR2GRAY)
        # Convert image to binary
        #_, bw = cv2.threshold(ImageforPCA, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #adaptive threshold
        #bw = cv2.adaptiveThreshold(ImageforPCA,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # Find all the contours in the thresholded image
        # _, contours = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # for i, c in enumerate(contours):
        #     # Calculate the area of each contour
        #     area = cv2.contourArea(c)
        #     # Ignore contours that are too small or too large
        #     if area < 1e2 or 1e5 < area:
        #         continue
        #     # Draw each contour only for visualisation purposes
        #     cv2.drawContours(ImageforPCA, contours, i, (0, 0, 255), 2)
        #     # Find the orientation of each shape
        #     _3DVisLabLib.Get_PCA_getOrientation(c, ImageforPCA)

        #_3DVisLabLib.ImageViewer_Quick_no_resize(bw,0,True,True)
        # Find the orientation of each shape
        #_3DVisLabLib.Get_PCA_getOrientation(bw, ImageforPCA)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(ImageforPCA,0,True,True)
        #https://towardsdatascience.com/principal-component-analysis-in-depth-understanding-through-image-visualization-892922f77d9f
        #in_matrix = None 
        #vec = Pod1Image_Grayscale.reshape(Pod1Image_Grayscale.shape[0] * Pod1Image_Grayscale.shape[1])
        #in_matrix = vec
        #can also stack images if we want
        #n_matrix = np.vstack((in_matrix, vec))'
        #mean, eigenvectors = cv2.PCACompute(in_matrix, np.mean(in_matrix, axis=0).reshape(1,-1))
        #np.set_printoptions(precision=3)
        #cov = np.cov(MB_matrix.transpose())

        #https://towardsdatascience.com/principal-component-analysis-in-depth-understanding-through-image-visualization-892922f77d9f
        #n_bands=Pod1Image_col.shape[2]-1
        # 3 dimensional dummy array with zeros

        #multi channel PCA
        PCA_image=Pod1Image_col_cropped_PCA
        MB_img = np.zeros((PCA_image.shape[0],PCA_image.shape[1],PCA_image.shape[2]))
        # stacking up images (channels?) into the array
        #this is unncessary but leave here in case we want to do something later on by laying up images
        for i in range(PCA_image.shape[2]):
            MB_img[:,:,i] =PCA_image[:,:,i]

        # Convert 2d band array in 1-d to make them as feature vectors and Standardization
        MB_matrix = np.zeros((MB_img[:,:,0].size,PCA_image.shape[2]))

        for i in range(PCA_image.shape[2]):
            MB_array = MB_img[:,:,i].flatten()  # covert 2d to 1d array 
            MB_arrayStd = (MB_array - MB_array.mean())/MB_array.std()  
            MB_matrix[:,i] = MB_arrayStd

        np.set_printoptions(precision=3)
        cov = np.cov(MB_matrix.transpose())
        # Eigen Values
        EigVal,EigVec = np.linalg.eig(cov)
        # Ordering Eigen values and vectors
        order = EigVal.argsort()[::-1]
        EigVal = EigVal[order]
        EigVec = EigVec[:,order]
        #Projecting data on Eigen vector directions resulting to Principal Components 
        PC = np.matmul(MB_matrix,EigVec)   #cross product


        #save PCA into image info object
        ImageInfo.EigenValues=EigVal
        print((EigVec[0],EigVec[1]))
        ImageInfo.EigenVectors=EigVec
        ImageInfo.PrincpleComponents=PC




        #load into tuple for object
        ImageInfo.Histogram=hist
        ImageInfo.ImageGrayscale=Pod1Image_Grayscale
        ImageInfo.ImageColour=Pod1Image_col
        ImageInfo.ImageAdjusted=Pod1Image_col_cropped_FM
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


    OutOfUse=0


    #initialise metric matrices
    MatchImages.HM_data_histo = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_FM = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_All = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_FourierDifference = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_MetricDistances = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_EigenVectorDotProd = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))


    for BaseImageList in MatchImages.ImagesInMem_Pairing:
        print(BaseImageList,"/",len(MatchImages.ImagesInMem_Pairing))
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
        Base_Image_FM=MatchImages.ImagesInMem_to_Process[Base_Image_name].ImageAdjusted
        Base_Image_EigenVectors=MatchImages.ImagesInMem_to_Process[Base_Image_name].EigenVectors
        Base_Image_EigenValues=MatchImages.ImagesInMem_to_Process[Base_Image_name].EigenValues
        for TestImageList in MatchImages.ImagesInMem_Pairing:
            #if TestImageList%300==0:
            #    print(BaseImageList,"/",len(MatchImages.ImagesInMem_Pairing),"--testimg--",TestImageList)
            if TestImageList<BaseImageList:
                #data is diagonally symmetrical
                continue
            if MatchImages.ImagesInMem_Pairing[TestImageList][1].InUse==False:
                pass
                #continue
            #check not testing itself
            if BaseImageList==TestImageList:
                plop=1
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
            Test_Image_FM=MatchImages.ImagesInMem_to_Process[Test_Image_name].ImageAdjusted
            Test_Image_EigenVectors=MatchImages.ImagesInMem_to_Process[Test_Image_name].EigenVectors
            Test_Image_EigenValues=MatchImages.ImagesInMem_to_Process[Test_Image_name].EigenValues

            #eigenvector metric
            #get dot product of top eigenvector (should be sorted for most significant set to [0])
            EigenDotProduct=1-(Base_Image_EigenVectors[0] @ Test_Image_EigenVectors[0])
            
            EigenValue_diff=abs((Base_Image_EigenValues[0] )-(Test_Image_EigenValues[0] ))
            #print(EigenValue_diff)
            CheckImages_InfoSheet.All_EigenDotProd_result.append(EigenValue_diff)

            #histogram metric
            HistogramSimilarity=CompareHistograms(Base_Image_Histo,Test_Image_Histo)
            CheckImages_InfoSheet.AllHisto_results.append(HistogramSimilarity)

            #feature match metric
            MatchedPoints,OutputImage,PointsA,PointsB,FinalMatchMetric=_3DVisLabLib.Orb_FeatureMatch(Base_Image_FMatches,Base_Image_Descrips,Test_Image_FMatches,Test_Image_Descrips,99999,Base_Image_FM,Test_Image_FM,0.2)
            AverageMatchDistance=FinalMatchMetric#smaller the better
            #_3DVisLabLib.ImageViewer_Quick_no_resize(OutputImage,0,True,False)
            CheckImages_InfoSheet.All_FM_results.append(AverageMatchDistance)

            #fourier difference metric
            #get differnce between fourier magnitudes of image
            #not the best solution as fourier magnitude will rotate with image 
            #generally this performs well on its own as matches similar notes with similar skew
            FourierDifference=(abs(Base_Image_FourierMag-Test_Image_FourierMag)).sum()

            #populate output metric comparison matrices
            MatchImages.HM_data_histo[BaseImageList,TestImageList]=HistogramSimilarity
            MatchImages.HM_data_FM[BaseImageList,TestImageList]=AverageMatchDistance
            MatchImages.HM_data_FourierDifference[BaseImageList,TestImageList]=FourierDifference
            MatchImages.HM_data_EigenVectorDotProd[BaseImageList,TestImageList]=EigenValue_diff
            #data is symmetrical - fill it in to help with visualisation
            MatchImages.HM_data_histo[TestImageList,BaseImageList]=HistogramSimilarity
            MatchImages.HM_data_FM[TestImageList,BaseImageList]=AverageMatchDistance
            MatchImages.HM_data_FourierDifference[TestImageList,BaseImageList]=FourierDifference
            MatchImages.HM_data_EigenVectorDotProd[TestImageList,BaseImageList]=EigenValue_diff
            
            #old code for sorting images in place
            if HistogramSimilarity<CheckImages_InfoSheet.BestMatch_Histo:
                CheckImages_InfoSheet.BestMatch_Histo=HistogramSimilarity
                CheckImages_InfoSheet.BestMatch_Histo_listIndex=TestImageList

            if AverageMatchDistance<CheckImages_InfoSheet.BestMatch_FeatureMatch:
                CheckImages_InfoSheet.BestMatch_FeatureMatch=AverageMatchDistance
                CheckImages_InfoSheet.BestMatch_FeatureMatch_listIndex=TestImageList

        #after check all images, if a result then copy that list into the first list so combine the sets of images
        #THIS HAS BEEN DISABLED!
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



    
    #sort out results and populate final metric
    MatchImages.HM_data_FM=normalize_2d(MatchImages.HM_data_FM)
    MatchImages.HM_data_histo=normalize_2d(MatchImages.HM_data_histo)
    MatchImages.HM_data_FourierDifference=normalize_2d(MatchImages.HM_data_FourierDifference)
    MatchImages.HM_data_EigenVectorDotProd=normalize_2d(MatchImages.HM_data_EigenVectorDotProd)
    
    for BaseImageList in MatchImages.ImagesInMem_Pairing:
        for TestImageList in MatchImages.ImagesInMem_Pairing:
            if TestImageList<BaseImageList:
                #data is diagonally symmetrical
                continue
            EigenVectorDotProd=0#MatchImages.HM_data_EigenVectorDotProd[BaseImageList,TestImageList]
            HistogramSimilarity=MatchImages.HM_data_histo[BaseImageList,TestImageList]
            AverageMatchDistance=0#MatchImages.HM_data_FM[BaseImageList,TestImageList]
            FourierDifference=MatchImages.HM_data_FourierDifference[BaseImageList,TestImageList]


            #experiment with metric distance
            MatchImages.HM_data_MetricDistances[BaseImageList,TestImageList]=math.sqrt((HistogramSimilarity**2)+(AverageMatchDistance**2)+(FourierDifference**2)+(EigenVectorDotProd**2))
            #mirror data for visualisation
            MatchImages.HM_data_MetricDistances[TestImageList,BaseImageList]=MatchImages.HM_data_MetricDistances[BaseImageList,TestImageList]
            if TestImageList==BaseImageList:
                plop=1
    MatchImages.HM_data_MetricDistances=normalize_2d(MatchImages.HM_data_MetricDistances)
    #HM_data_All=normalize_2d(MatchImages.HM_data_histo+MatchImages.HM_data_FM+MatchImages.HM_data_FourierDifference)
   



    MatchImages_lib.PrintResults(MatchImages,CheckImages_InfoSheet,PlotAndSave_2datas,PlotAndSave)




    #pairwise matching
    MatchImages_lib.PairWise_Matching(MatchImages,CheckImages_InfoSheet,PlotAndSave_2datas,PlotAndSave,ImgCol_InfoSheet_Class)







    #sequential matching
    MatchImages_lib.SequentialMatchingPerImage(MatchImages,CheckImages_InfoSheet,PlotAndSave_2datas,PlotAndSave)

    SSL_ERROR_SSL



    #HM_data_All=MatchImages.HM_data_FM
    HM_data_All_Copy=copy.deepcopy(HM_data_All)

    #debug final data
    HM_data_All=MatchImages.HM_data_MetricDistances

    

    #if have equal length for both results, asssume they are aligned - can examine response
    if len(CheckImages_InfoSheet.All_FM_results)==len(CheckImages_InfoSheet.AllHisto_results):
        FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_All") +".jpg"
        PlotAndSave_2datas("HM_data_All",FilePath,HM_data_All)
        FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_FM") +".jpg"
        PlotAndSave_2datas("HM_data_FM",FilePath,MatchImages.HM_data_FM)
        FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_histo") +".jpg"
        PlotAndSave_2datas("HM_data_histo",FilePath,MatchImages.HM_data_histo)
        FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_FourierDifference") +".jpg"
        PlotAndSave_2datas("HM_data_FourierDifference",FilePath,MatchImages.HM_data_FourierDifference)
        FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("MatchImages.HM_data_MetricDistances") +".jpg"
        PlotAndSave_2datas("MatchImages.HM_data_MetricDistances",FilePath,MatchImages.HM_data_MetricDistances)

    
        #for every image or subsets of images, roll through heatmap finding nearest best match then
        #cross referencing it
        OrderedImages=dict()
        #BaseImageList=random.choice(list(MatchImages.ImagesInMem_Pairing.keys()))

        #get minimum 
        #result = np.where(HM_data_All == np.amin(HM_data_All))
        #Element=random.choice(result[0])#incase we have two identical results


        #blank out the self test
        BlankOut=HM_data_All.max()*2.00000#should be "2" if normalised
        for item in MatchImages.ImagesInMem_Pairing:
            HM_data_All[item,item]=BlankOut

        #print(HM_data_All)
        #print("-----")
        BaseImageList=0#random.choice(list(MatchImages.ImagesInMem_Pairing.keys()))
        Counter=0
        MatchMetric_all=[]
        MatchMetric_Histo=[]
        MatchMetric_Fourier=[]
        MatchMetric_FM=[]
        while len(OrderedImages)+1<len(MatchImages.ImagesInMem_Pairing):#+1 is a fudge or it crashes out with duplicate image bug - cant figure this out 
            Counter=Counter+1
            #FilePath=MatchImages.OutputPairs +"\\00" + str(Counter) +  str(OutOfUse) +("HM_data_All") +".jpg"
            #PlotAndSave_2datas("HM_data_All",FilePath,normalize_2d(HM_data_All))
            
            #print("looking at row",BaseImageList,"for match for for")
            #HM_data_All[BaseImageList,BaseImageList]=BlankOut
            Row=HM_data_All[0:len(MatchImages.ImagesInMem_Pairing),BaseImageList]
            #print(Row)
            #get minimum value
            result = np.where(Row == np.amin(Row))
            #print("REsult",Row)
            Element=random.choice(result[0])#incase we have two identical results
            #print("nearest matching is element",Element)
            #print("nearest value",HM_data_All[Element,BaseImageList])
            MatchMetric_all.append(HM_data_All[Element,BaseImageList])
            MatchMetric_Histo.append(MatchImages.HM_data_histo[Element,BaseImageList])
            MatchMetric_Fourier.append(MatchImages.HM_data_FourierDifference[Element,BaseImageList])
            MatchMetric_FM.append(MatchImages.HM_data_FM[Element,BaseImageList])
            #add to output images
            

            for imgIndex, Images in enumerate (MatchImages.ImagesInMem_Pairing[Element][0]):
                #if len(Images)>1:y
                    #raise Exception("too many images")
                SplitImagePath=Images.split("\\")[-1]
                FilePath=MatchImages.OutputPairs +"\\00" +str(Counter)+ "_ImgNo_" + str(BaseImageList) + "_score_" + str(round(HM_data_All[Element,BaseImageList],3))+ "_" + SplitImagePath
                cv2.imwrite(FilePath,MatchImages.ImagesInMem_to_Process[Images].ImageColour)
                if Images in OrderedImages:
                    raise Exception("output images file already exists!!! logic error " + FilePath)
                else:
                    OrderedImages[Images]=BaseImageList
            #now print out histogram with skew?
            #FilePath=MatchImages.OutputPairs +"\\00" +str(Counter)+ "_ImgNo_" + str(BaseImageList) + "_" + str(round(HM_data_All[Element,BaseImageList],3))+ "_HISTO_" + SplitImagePath
            #PlotAndSaveHistogram("self similar histogram",FilePath,HM_data_All_Copy[0:len(MatchImages.ImagesInMem_Pairing),BaseImageList],0,30)


            #blank out element in All places
            HM_data_All[0:len(MatchImages.ImagesInMem_Pairing),BaseImageList]=BlankOut
            HM_data_All[BaseImageList,0:len(MatchImages.ImagesInMem_Pairing)]=BlankOut
            #if Counter==1:
            #    HM_data_All[0:len(MatchImages.ImagesInMem_Pairing),Element]=BlankOut
            #    HM_data_All[Element,0:len(MatchImages.ImagesInMem_Pairing)]=BlankOut
            #baseimage should be an integer
            #work in columns to find nearest match, data should be mirrored diagonally to make it easier to visualise#
            
            #move to next element
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