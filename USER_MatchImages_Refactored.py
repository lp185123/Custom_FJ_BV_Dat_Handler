
from logging import raiseExceptions
from ssl import SSL_ERROR_SSL
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
import statistics
import scipy
import math
import numpy as np
from scipy.stats import skew
import gc
import MatchImages_lib
#gc.disable()


#stuff for HOG
#from skimage.io import Ski_imread, ski_imshow
#from skimage.transform import ski_resize
#from skimage.feature import ski_hog
#from skimage import ski_exposure


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
        image1_Norm = cv2.normalize(image1, image1,0, 255, cv2.NORM_MINMAX)
        image2_Norm = cv2.normalize(image2,image2, 0, 255, cv2.NORM_MINMAX)
        # black blank image, hieght of both images and width of widest image
        blank_image = np.zeros(shape=[image1.shape[0]+image2.shape[0], max(image1.shape[1],image2.shape[1]), image2.shape[2]], dtype=np.uint8)
        #drop in images
        blank_image[0:image1_Norm.shape[0],0:image1_Norm.shape[1],:]=image1_Norm[:,:,:]
        blank_image[image1_Norm.shape[0]:,0:image2_Norm.shape[1]:,:]=image2_Norm[:,:,:]
        #_3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(blank_image,(blank_image.shape[1]*3,blank_image.shape[0]*3)),0,True,True)
        return blank_image

    def USERFunction_PrepareForHOG(self,image,TargetHeight_HOG=64,TargetWidth_HOG=128):
        #HOG expects 64 * 128
        #lets not chagne aspect ratio
        ImageHeight=image.shape[0]
        ImageWidth=image.shape[1]
        #set to target height then crop as needed
        Percent2match=TargetHeight_HOG/ImageHeight
        TargetWidth=round(ImageWidth*Percent2match)
        Resized=cv2.resize(image,(TargetWidth,TargetHeight_HOG))

        #create blank
        blank_image = np.zeros(shape=[TargetHeight_HOG, TargetWidth_HOG, image.shape[2]], dtype=np.uint8)

        #now crop to HOG shape
        HOG_specific_crop=Resized[:,0:TargetWidth_HOG,:]

        blank_image[0:HOG_specific_crop.shape[0],0:HOG_specific_crop.shape[1],:]=HOG_specific_crop

        #now need to flip on its side
        # rotate ccw
        Rotate=cv2.transpose(blank_image)
        Rotate=cv2.flip(Rotate,flipCode=0)
        return Rotate

    def USERFunction_CropForFM(self,image):
        return image
        #return image[:,0:151,:]
        #return image[0:int(image.shape[0]/1.1),0:int(image.shape[1]/1),:]
    def USERFunction_CropForHistogram(self,image):
        return image
        #return image[:,0:151,:]
    def USERFunction_Resize(self,image):
        #return image
        #this is pixels not percentage!
        #return image
        return cv2.resize(image.copy(),(400,300))

    class ProcessTerms(enum.Enum):
        Sequential="Sequential"
        DoubleUp="DoubleUp"

    """Class to hold information for image sorting & match process"""
    def __init__(self):
        self.InputFolder=r"E:\NCR\TestImages\UK_Side_ALL"
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
        self.SubSetOfData=int(99)#subset of data
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
        self.HM_data_HOG_Dist=None
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
            self.OriginalImage=None
            self.ImageColour=None
            self.ImageAdjusted=None
            self.FM_Keypoints=None
            self.FM_Descriptors=None
            self.FourierTransform_mag=None
            self.ImageGrayscale=None
            self.EigenValues=None
            self.EigenVectors=None
            self.PrincpleComponents=None
            self.HOG_Mag=None
            self.HOG_Angle=None
            self.OPENCV_hog_descriptor=None
            
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

def Get_PCA_(InputImage):
    #https://towardsdatascience.com/principal-component-analysis-in-depth-understanding-through-image-visualization-892922f77d9f
    #n_bands=Pod1Image_col.shape[2]-1
    # 3 dimensional dummy array with zeros
    #multi channel PCA
    PCA_image=InputImage
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
    return PC,EigVal,EigVec

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

def GetHOG_featureVector(image):
    img = np.float32(image) / 255.0
    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    # Python Calculate gradient magnitude and direction ( in degrees )
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    Return_norm = cv2.normalize(mag, mag,0, 255, cv2.NORM_MINMAX)
    return Return_norm, angle


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

    #HOG detector - might want to put this in a function
    # Create HOG Descriptor object
    HOG_extrator = cv2.HOGDescriptor()
    for Index, ImagePath in enumerate(RandomOrder):
        print("Loading in image",ImagePath )
        #create class object for each image
        ImageInfo=MatchImages.ImageInfo()

        #load in original image
        OriginalImage_GrayScale = cv2.imread(ImagePath, cv2.IMREAD_GRAYSCALE)
        OriginalImage_col = cv2.imread(ImagePath)

        #create resized versions
        GrayScale_Resized=MatchImages.USERFunction_Resize(OriginalImage_GrayScale)
        Colour_Resized=MatchImages.USERFunction_Resize(OriginalImage_col)

        #create version for feature matching
        Image_For_FM=MatchImages.USERFunction_CropForFM(Colour_Resized)

        #create version for histogram matching
        Image_For_Histogram=MatchImages.USERFunction_CropForHistogram(Colour_Resized)
        #get histogram for comparing colours
        hist = cv2.calcHist([Image_For_Histogram], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        #prepare image for HOG matching
        #image needs to be a particular size for HOG matching
        For_HOG_FeatureMatch=MatchImages.USERFunction_PrepareForHOG(Colour_Resized.copy())
        #this step is for visualisation
        HOG_mag,HOG_angle=GetHOG_featureVector(For_HOG_FeatureMatch)
        #ensure input image is correct dimensions for HOG function
        if For_HOG_FeatureMatch.shape[0]!=128 and For_HOG_FeatureMatch.shape[0]!=64:
            raise Exception("Image not correct size for HOG (128 * 64)")
        else:
            #get histogram for HOG used for comparison during match matrix
            OPENCV_hog_descriptor=HOG_extrator.compute(For_HOG_FeatureMatch)
        
        #create extended image for feature matching to experiment with
        HOG_Style_Gradient,HOG_angle_temp=GetHOG_featureVector(Image_For_FM.copy())
        #try and force a normalised visulation for the magnitude image 
        #normal normalisation doesnt seem to work - do not know why yet
        StackedColour_AndGradient_img=MatchImages.StackTwoimages(Image_For_FM,HOG_Style_Gradient)


        #force image normalisation - not sure why this doesnt work with normal normalisation
        GradientImage=MatchImages.StackTwoimages(HOG_Style_Gradient,HOG_Style_Gradient)

        #for principle component analysis
        Image_For_PCA=GradientImage.copy()
        PC,EigVal,EigVec=Get_PCA_(Image_For_PCA)
        

        #get feature match keypoints
        keypoints,descriptor=_3DVisLabLib.OrbKeyPointsOnly(StackedColour_AndGradient_img,MatchImages.FeatureMatch_Dict_Common.ORB_default)

        #get fourier transform
        #TODO don't we want fourier cofficients here?
        f = np.fft.fft2(OriginalImage_GrayScale)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))#magnitude is what we will use to compare

        

        if Index==0:
            #on first loop show image to user
            DrawnKeypoints=_3DVisLabLib.draw_keypoints_v2(StackedColour_AndGradient_img.copy(),keypoints)
            print("original image")
            _3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(Colour_Resized,(Colour_Resized.shape[1]*3,Colour_Resized.shape[0]*3)),0,True,True)
            print("feature match sample")
            _3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(DrawnKeypoints,(DrawnKeypoints.shape[1]*2,DrawnKeypoints.shape[0]*2)),0,True,True)
            print("Crop for histogram")
            _3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(Image_For_Histogram,(Image_For_Histogram.shape[1]*3,Image_For_Histogram.shape[0]*3)),0,True,True)
            print("Crop for PCA")
            _3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(Image_For_PCA,(Image_For_PCA.shape[1]*3,Image_For_PCA.shape[0]*3)),0,True,True)
            print("Prepare for HOG features")
            _3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(For_HOG_FeatureMatch,(For_HOG_FeatureMatch.shape[1]*3,For_HOG_FeatureMatch.shape[0]*3)),0,True,True)
            print("HOG features")
            _3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(HOG_mag,(HOG_mag.shape[1]*3,HOG_mag.shape[0]*3)),0,True,True)
            

        #get fourier transform
        #dft = cv2.dft(np.float32(OriginalImage_GrayScale),flags = cv2.DFT_COMPLEX_OUTPUT)
        #dft_shift = np.fft.fftshift(dft)
        #magnitude = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
        #product = 20*np.log(magnitude)
        


        #magnitude_spectrum_normed=MatchImages.ForceNormalise_forHOG(magnitude_spectrum)
        #print("forced magnitude_spectrum_normed")
        #_3DVisLabLib.ImageViewer_Quick_no_resize(magnitude_spectrum_normed,0,True,True)


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
        #vec = OriginalImage_GrayScale.reshape(OriginalImage_GrayScale.shape[0] * OriginalImage_GrayScale.shape[1])
        #in_matrix = vec
        #can also stack images if we want
        #n_matrix = np.vstack((in_matrix, vec))'
        #mean, eigenvectors = cv2.PCACompute(in_matrix, np.mean(in_matrix, axis=0).reshape(1,-1))
        #np.set_printoptions(precision=3)
        #cov = np.cov(MB_matrix.transpose())

        


        #save PCA into image info object


        #load into image object
        ImageInfo.EigenValues=EigVal
        ImageInfo.EigenVectors=EigVec
        ImageInfo.PrincpleComponents=PC
        ImageInfo.Histogram=hist
        ImageInfo.OriginalImage=OriginalImage_col
        ImageInfo.ImageGrayscale=OriginalImage_GrayScale
        ImageInfo.ImageColour=Colour_Resized
        ImageInfo.ImageAdjusted=Image_For_FM
        ImageInfo.FM_Keypoints=keypoints
        ImageInfo.FM_Descriptors=descriptor
        ImageInfo.FourierTransform_mag=magnitude_spectrum
        ImageInfo.HOG_Mag=HOG_mag
        ImageInfo.HOG_Angle=HOG_angle
        ImageInfo.OPENCV_hog_descriptor=OPENCV_hog_descriptor
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

    #initialise metric matrices
    MatchImages.HM_data_histo = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_FM = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_All = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_FourierDifference = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_MetricDistances = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_EigenVectorDotProd = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))
    MatchImages.HM_data_HOG_Dist = np.zeros((len(MatchImages.ImagesInMem_Pairing),len(MatchImages.ImagesInMem_Pairing)))


    for BaseImageList in MatchImages.ImagesInMem_Pairing:
        print(BaseImageList,"/",len(MatchImages.ImagesInMem_Pairing))

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
        Base_Image_HOG_Descriptor=MatchImages.ImagesInMem_to_Process[Base_Image_name].OPENCV_hog_descriptor
        for TestImageList in MatchImages.ImagesInMem_Pairing:
            if TestImageList<BaseImageList:
                #data is diagonally symmetrical
                continue
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
            Test_Image_HOG_Descriptor=MatchImages.ImagesInMem_to_Process[Test_Image_name].OPENCV_hog_descriptor



            #eigenvector metric
            #get dot product of top eigenvector (should be sorted for most significant set to [0])
            EigenDotProduct=round((Base_Image_EigenVectors[0] @ Test_Image_EigenVectors[0]),5)
            EigenValue_diff=abs((Base_Image_EigenValues[0] )-(Test_Image_EigenValues[0] ))
            #print(EigenValue_diff)
            #CheckImages_InfoSheet.All_EigenDotProd_result.append(EigenValue_diff)



            #histogram metric
            HistogramSimilarity=CompareHistograms(Base_Image_Histo,Test_Image_Histo)
            #CheckImages_InfoSheet.AllHisto_results.append(HistogramSimilarity)



            #feature match metric
            try:
                MatchedPoints,OutputImage,PointsA,PointsB,FinalMatchMetric=_3DVisLabLib.Orb_FeatureMatch(Base_Image_FMatches,Base_Image_Descrips,Test_Image_FMatches,Test_Image_Descrips,99999,Base_Image_FM,Test_Image_FM,0.75)
                AverageMatchDistance=FinalMatchMetric#smaller the better
                #print("Feature match",FinalMatchMetric,len(Base_Image_FMatches),len(Test_Image_FMatches))
            except:
                print("ERROR with feature match",len(Base_Image_FMatches),len(Test_Image_FMatches))
                #watch out this might not be a valid maximum!!
                AverageMatchDistance=-99999
            #CheckImages_InfoSheet.All_FM_results.append(AverageMatchDistance)




            HOG_distance=CompareHistograms(Base_Image_HOG_Descriptor, Test_Image_HOG_Descriptor)
            #fourier difference metric
            #get differnce between fourier magnitudes of image
            #not the best solution as fourier magnitude will rotate with image 
            #generally this performs well on its own as matches similar notes with similar skew
            FourierDifference=(abs(Base_Image_FourierMag-Test_Image_FourierMag)).sum()


            StackTwoimages=MatchImages.StackTwoimages(Base_Image_FM,Test_Image_FM)
            #_3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(StackTwoimages,(StackTwoimages.shape[1]*1,StackTwoimages.shape[0]*1)),0,True,True)
            #populate output metric comparison matrices
            MatchImages.HM_data_histo[BaseImageList,TestImageList]=HistogramSimilarity
            MatchImages.HM_data_FM[BaseImageList,TestImageList]=AverageMatchDistance
            MatchImages.HM_data_FourierDifference[BaseImageList,TestImageList]=FourierDifference
            MatchImages.HM_data_EigenVectorDotProd[BaseImageList,TestImageList]=EigenValue_diff
            MatchImages.HM_data_HOG_Dist[BaseImageList,TestImageList]=HOG_distance
            #data is symmetrical - fill it in to help with visualisation
            MatchImages.HM_data_histo[TestImageList,BaseImageList]=HistogramSimilarity
            MatchImages.HM_data_FM[TestImageList,BaseImageList]=AverageMatchDistance
            MatchImages.HM_data_FourierDifference[TestImageList,BaseImageList]=FourierDifference
            MatchImages.HM_data_EigenVectorDotProd[TestImageList,BaseImageList]=EigenValue_diff
            MatchImages.HM_data_HOG_Dist[TestImageList,BaseImageList]=HOG_distance
            

    #sort out results and populate final metric
    
    
    #we have to repair placeholder for no data by maxing it out over valid max, but just enough so 
    #we can still use the visualisations easily without them being oversaturated with large dynamic rane
    MatchImages.HM_data_FM=normalize_2d(np.where(MatchImages.HM_data_FM==-99999, MatchImages.HM_data_FM.max()+1, MatchImages.HM_data_FM))
    MatchImages.HM_data_histo=normalize_2d(MatchImages.HM_data_histo)
    MatchImages.HM_data_FourierDifference=normalize_2d(MatchImages.HM_data_FourierDifference)
    MatchImages.HM_data_EigenVectorDotProd=normalize_2d(MatchImages.HM_data_EigenVectorDotProd)
    MatchImages.HM_data_HOG_Dist=normalize_2d(MatchImages.HM_data_HOG_Dist)
    
    for BaseImageList in MatchImages.ImagesInMem_Pairing:
        for TestImageList in MatchImages.ImagesInMem_Pairing:
            if TestImageList<BaseImageList:
                #data is diagonally symmetrical
                continue
            EigenVectorDotProd=0#MatchImages.HM_data_EigenVectorDotProd[BaseImageList,TestImageList]
            HistogramSimilarity=0#MatchImages.HM_data_histo[BaseImageList,TestImageList]
            AverageMatchDistance=0#MatchImages.HM_data_FM[BaseImageList,TestImageList]
            FourierDifference=0#MatchImages.HM_data_FourierDifference[BaseImageList,TestImageList]
            HOG_Distance=MatchImages.HM_data_HOG_Dist[BaseImageList,TestImageList]

            #experiment with metric distance
            MatchImages.HM_data_MetricDistances[BaseImageList,TestImageList]=math.sqrt((HistogramSimilarity**2)+(AverageMatchDistance**2)+(FourierDifference**2)+(EigenVectorDotProd**2)+HOG_Distance**2)
            #mirror data for visualisation
            MatchImages.HM_data_MetricDistances[TestImageList,BaseImageList]=MatchImages.HM_data_MetricDistances[BaseImageList,TestImageList]
            if TestImageList==BaseImageList:
                plop=1
    MatchImages.HM_data_MetricDistances=normalize_2d(MatchImages.HM_data_MetricDistances)
    #HM_data_All=normalize_2d(MatchImages.HM_data_histo+MatchImages.HM_data_FM+MatchImages.HM_data_FourierDifference)
   



    MatchImages_lib.PrintResults(MatchImages,CheckImages_InfoSheet,PlotAndSave_2datas,PlotAndSave)


    #sequential matching
    MatchImages_lib.SequentialMatchingPerImage(MatchImages,CheckImages_InfoSheet,PlotAndSave_2datas,PlotAndSave)

    #pairwise matching
    #MatchImages_lib.PairWise_Matching(MatchImages,CheckImages_InfoSheet,PlotAndSave_2datas,PlotAndSave,ImgCol_InfoSheet_Class)

    



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