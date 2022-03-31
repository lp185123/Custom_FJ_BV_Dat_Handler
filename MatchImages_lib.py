import copy
from difflib import Match
import numpy as np
import time
import random
import cv2
import _3DVisLabLib
import scipy.stats as stats
import statistics
import matplotlib.pyplot as pl
from statistics import mean 
import math
import shutil


# Create HOG Descriptor object outside of loops
HOG_extrator = cv2.HOGDescriptor()
HistogramCentralis_mask=None
HistogramStripulus_Centralis=None

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

def USERFunction_PrepareForHOG(image,TargetHeight_HOG=64,TargetWidth_HOG=128):
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

def GetFFT_OfImage(InputGraySCaleImg,CropPercent_100,Blur):
    if len(InputGraySCaleImg.shape)==3:
        raise Exception("GetFFT_OfImage image not grayscale")

    f = np.fft.fft2(InputGraySCaleImg)
    fshift = np.fft.fftshift(f)
    FFT_magnitude_spectrum = 20*np.log(np.abs(fshift))#magnitude is what we will use to compare

    #most of fourier is just noise - lets crop it
    CropRange=CropPercent_100/100#0.80#1.0=100%
    RangeX=int(FFT_magnitude_spectrum.shape[0]*CropRange)
    RangeY=int(FFT_magnitude_spectrum.shape[1]*CropRange)
    BufferX=int((FFT_magnitude_spectrum.shape[0]-RangeX)/2)
    BufferY=int((FFT_magnitude_spectrum.shape[1]-RangeY)/2)
    FFT_magnitude_spectrum=FFT_magnitude_spectrum[BufferX:-BufferX,BufferY:-BufferY]
    
    if Blur==True:
        #a lot of noise - lets see if we can remove noise
        KernelSize=3
        kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing - maybe make smoother as such small images
        dst = cv2.filter2D(FFT_magnitude_spectrum,-1,kernel)
        FFT_magnitude_spectrum_visualise=cv2.convertScaleAbs(dst)
    
    return FFT_magnitude_spectrum

def PCA_Structure(Image):
    #if colour image,convert to gray
    if len(Image.shape) == 3:
        grayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    else:
        grayImage=Image
    #binarise image and use for PCA of structure
    img_blur = cv2.medianBlur(grayImage, 5)
    Binarised_Otsu,image_result  = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #_3DVisLabLib.ImageViewer_Quick_no_resize(image_result,0,True,True)
    count = np.count_nonzero(image_result == 255)
    #create numpy array for each white or black pixel depending on what we will use
    data_pts = np.zeros((count, 2), dtype=np.float64)
    #create array
    Datapt_index=0
    for _X in range(image_result.shape[0]):
        for _Y in range(image_result.shape[1]):
            if image_result[_X,_Y]>0:
                data_pts[Datapt_index][0]=_X
                data_pts[Datapt_index][1]= _Y
                Datapt_index=Datapt_index+1
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    angle_list=[]
    eigenvectors_list=[]
    eigenvalues_list=[]
    for EigenIndex, EigenResult in enumerate(eigenvectors):
        cntr = (int(mean[0, 0]), int(mean[0, 1]))
        angle = math.degrees(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))  # orientation in degrees
        angle_list.append(angle)
        eigenvectors_list.append(eigenvectors[EigenIndex])
        eigenvalues_list.append(eigenvalues[EigenIndex])
    return image_result,angle_list,eigenvectors_list,eigenvalues_list

def PrepareImageMetrics_FacesRandomObjects(PrepareMatchImages,ImagePath,Index,ImageReviewDict):
    #create class object for each image
    ImageInfo=PrepareMatchImages.ImageInfo()

    #load in original image
    OriginalImage_col = cv2.imread(ImagePath)
    #bad code to convert gray to colour but still 3 channels
    if len(OriginalImage_col.shape)!=3:
        OriginalImage_col=cv2.cvtColor(OriginalImage_col,cv2.COLOR_GRAY2RGB)
        OriginalImage_GrayScale=OriginalImage_col.copy()
    else:
        #load as grayscale but still with 3 channels to be compatible with functions
        #very lazy code 
        OriginalImage_GrayScale_temp = cv2.cvtColor(OriginalImage_col, cv2.COLOR_BGR2GRAY)
        OriginalImage_GrayScale=OriginalImage_col.copy()
        OriginalImage_GrayScale[:,:,0]=OriginalImage_GrayScale_temp
        OriginalImage_GrayScale[:,:,1]=OriginalImage_GrayScale_temp
        OriginalImage_GrayScale[:,:,2]=OriginalImage_GrayScale_temp
        #OriginalImage_GrayScale = cv2.cvtColor(OriginalImage_col, cv2.COLOR_BGR2GRAY)

    #OriginalImage_col=OriginalImage_GrayScale.copy()

    InputImage=PrepareMatchImages.USERFunction_OriginalImage(OriginalImage_col)
    if Index<3: ImageReviewDict["OriginalImage_GrayScale"]=OriginalImage_GrayScale
    if Index<3: ImageReviewDict["OriginalImage_col"]=OriginalImage_col
    if Index<3: ImageReviewDict["InputImage"]=InputImage

    #create resized versions
    GrayScale_Resized=PrepareMatchImages.USERFunction_ResizePercent(OriginalImage_GrayScale,100)
    Colour_Resized=PrepareMatchImages.USERFunction_ResizePercent(OriginalImage_col,100)



    #same size as face images
    FaceImageSize=(178,218)
    GrayScale_Resized=cv2.resize(GrayScale_Resized,(FaceImageSize[0] ,FaceImageSize[1]))
    Colour_Resized=cv2.resize(Colour_Resized,(FaceImageSize[0] ,FaceImageSize[1]))
    



    
    
    #create quilt/list of successively crops
    CropSequence_resizeRatio=Resize_toPixel_keepRatio(OriginalImage_col,FaceImageSize[1],FaceImageSize[0])
    CropSequence_resize=PrepareMatchImages.USERFunction_ResizePercent(CropSequence_resizeRatio,75)
    CropSequence_blur = cv2.blur(src=CropSequence_resize, ksize=(7, 7))
    #special case - don't crop the main set of images, but crop the incoming faces
    #have to be same size for cross correlation to work
    
    if ImagePath in PrepareMatchImages.List_ImagesToMatchFIlenames.values():
        MultiCropQuilt,MultiCropList=CreateCropInMatrixOfImage(CropSequence_blur,90,85,10,False)
    else:
        MultiCropQuilt,MultiCropList=CreateCropInMatrixOfImage(CropSequence_blur,100,75,10,False)
    #prepare for cross correlation
    if len(MultiCropQuilt.shape)==3:
        MultiCropQuilt_gray=cv2.cvtColor(MultiCropQuilt, cv2.COLOR_BGR2GRAY)
        MultiCropList_gray=[]
        #for Img in MultiCropList:
        #    MultiCropList_gray.append(cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY))
    #special case - don't crop the main set of images, but crop the incoming faces
    #MultiCropQuilt_gray = cv2.GaussianBlur(MultiCropQuilt_gray, (3,3), 0)
    #MultiCropQuilt_gray = cv2.Laplacian(MultiCropQuilt_gray,cv2.CV_64F)
    
    PhaseCorrelate_Std = np.float32(MultiCropQuilt_gray)

    #_3DVisLabLib.ImageViewer_Quick_no_resize(MultiCropQuilt_gray,0,True,True)
    if Index < 3: ImageReviewDict["PhaseCorrelate quilt visualise"] = MultiCropQuilt_gray


        



    #special case - don't crop the main set of images, but crop the incoming faces
    if ImagePath in PrepareMatchImages.List_ImagesToMatchFIlenames.values():
        #if faces - crop
        GrayScale_Resized=PrepareMatchImages.USERFunction_Crop_Pixels(GrayScale_Resized,(44,184),(30,145))#Y range then X range
        Colour_Resized=PrepareMatchImages.USERFunction_Crop_Pixels(Colour_Resized,(44,184),(30,145))#Y range then X range
        if Index<3: ImageReviewDict["Colour_Resized_FACE"]=Colour_Resized

    else:
        #otherwise, just resize - might want to keep aspect ratio somehow
        #dummy image is so we can paste in the above
        DummyImage = PrepareMatchImages.USERFunction_Crop_Pixels(Colour_Resized, (44, 184),(30,145))  # Y range then X range
        GrayScale_Resized=cv2.resize(GrayScale_Resized,(DummyImage.shape[1] ,DummyImage.shape[0]))
        Colour_Resized = cv2.resize(Colour_Resized, (DummyImage.shape[1], DummyImage.shape[0]))
        if Index < 3: ImageReviewDict["Colour_Resized_ ITEM"] = Colour_Resized

    Colour_Resized_blurred = cv2.blur(src=Colour_Resized, ksize=(15, 15))
    GrayScale_Resized_blurred = cv2.blur(src=GrayScale_Resized, ksize=(15, 15))

    #get small image to experiment with macro structure matching
    MacroStructure_img = cv2.blur(src=Colour_Resized, ksize=(20, 20))
    #MacroStructure_img = cv2.resize(Colour_Resized,(8,8))
    if Index < 3: ImageReviewDict["MacroStructure_img"] =  cv2.resize(MacroStructure_img,(270,270))#blow it up so we can see the preview- will be tiny otherwise

    #get PCA for structure
    image_result,angle_list,eigenvectors_list,eigenvalues_list=PCA_Structure(GrayScale_Resized_blurred)
    if Index < 3: ImageReviewDict["get PCA for structure"] = image_result
    PCA_Struct_EigenVecs=eigenvectors_list
    PCA_Struct_EigenVals=eigenvalues_list


    #create version for feature matching
    Image_For_FM=Colour_Resized.copy()#PrepareMatchImages.USERFunction_CropForFM(CoFlour_Resized)
    if Index<3: ImageReviewDict["Image_For_FM"]=Image_For_FM

    #create version for histogram matching
    Image_For_Histogram=Colour_Resized.copy()#PrepareMatchImages.USERFunction_CropForHistogram(Colour_Resized)
    Image_For_Histogram=PrepareMatchImages.USERFunction_ResizePercent(Image_For_Histogram,40)
    if Index<3: ImageReviewDict["Image_For_Histogram"]=Image_For_Histogram
    #get histogram for comparing colours
    hist = cv2.calcHist([Image_For_Histogram], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    #prepare image for HOG matching
    #image needs to be a particular size for HOG matching
    For_HOG_FeatureMatch=USERFunction_PrepareForHOG(Colour_Resized.copy())
    if Index<3: ImageReviewDict["For_HOG_FeatureMatch"]=For_HOG_FeatureMatch
    #this step is for visualisation
    HOG_mag,HOG_angle=GetHOG_featureVector(For_HOG_FeatureMatch)
    if Index<3: ImageReviewDict["HOG_mag visualised"]=cv2.convertScaleAbs(HOG_mag)
    #ensure input image is correct dimensions for HOG function
    if For_HOG_FeatureMatch.shape[0]!=128 and For_HOG_FeatureMatch.shape[0]!=64:
        raise Exception("Image not correct size for HOG (128 * 64)")
    else:
        #get histogram for HOG used for comparison during match matrix
        OPENCV_hog_descriptor=HOG_extrator.compute(For_HOG_FeatureMatch)
    
    #create extended image for feature matching to experiment with
    HOG_Style_Gradient_unnormalised,HOG_angle_temp=GetHOG_featureVector(Image_For_FM.copy())
    if Index<3: ImageReviewDict["HOG_Style_Gradient_unnormalised"]=HOG_Style_Gradient_unnormalised
    if Index<3: ImageReviewDict["HOG_Style_Gradient_visualised"]=cv2.convertScaleAbs(HOG_Style_Gradient_unnormalised)
    #try and force a normalised visulation for the magnitude image 
    #normal normalisation doesnt seem to work - do not know why yet
    StackedColour_AndGradient_img=PrepareMatchImages.StackTwoimages(Image_For_FM,HOG_Style_Gradient_unnormalised)
    if Index<3: ImageReviewDict["StackedColour_AndGradient_img"]=StackedColour_AndGradient_img

    #force image normalisation - not sure why this doesnt work with normal normalisation
    GradientImage=HOG_Style_Gradient_unnormalised.copy()#PrepareMatchImages.ForceNormalise_forHOG(HOG_Style_Gradient_unnormalised)
    if Index<3: ImageReviewDict["GradientImage visualise"]=cv2.convertScaleAbs(GradientImage)
    GradientImage_gray = cv2.cvtColor(GradientImage, cv2.COLOR_BGR2GRAY)
    if Index<3: ImageReviewDict["GradientImage_gray visualise"]=cv2.convertScaleAbs(GradientImage_gray)

    #for principle component analysis
    #has to have shape=3 (colour image)
    #Y range then X range
    #Image_For_PCA=PrepareMatchImages.USERFunction_Crop_Pixels(OriginalImage_col,(42,177),(59,119))
    Image_For_PCA=PrepareMatchImages.USERFunction_ResizePercent(Colour_Resized_blurred,90)
    PC,EigVal,EigVec=Get_PCA_(Image_For_PCA)
    if Index<3: ImageReviewDict["Image_For_PCA"]=Image_For_PCA

    #get feature match keypoints
    keypoints,descriptor=_3DVisLabLib.OrbKeyPointsOnly(StackedColour_AndGradient_img,PrepareMatchImages.FeatureMatch_Dict_Common.ORB_default)

    #get fourier transform
    #TODO don't we want fourier cofficients here?
    Image_fourier=cv2.cvtColor(Image_For_PCA,cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(Image_fourier)
    fshift = np.fft.fftshift(f)
    FFT_magnitude_spectrum = 20*np.log(np.abs(fshift))#magnitude is what we will use to compare

    #most of fourier is just noise - lets crop it
    CropRange=0.80#1.0=100%
    RangeX=int(FFT_magnitude_spectrum.shape[0]*CropRange)
    RangeY=int(FFT_magnitude_spectrum.shape[1]*CropRange)
    BufferX=int((FFT_magnitude_spectrum.shape[0]-RangeX)/2)
    BufferY=int((FFT_magnitude_spectrum.shape[1]-RangeY)/2)
    FFT_magnitude_spectrum=FFT_magnitude_spectrum[BufferX:-BufferX,BufferY:-BufferY]
    FFT_magnitude_spectrum_visualise=cv2.convertScaleAbs(FFT_magnitude_spectrum)
    if Index<3: ImageReviewDict["FFT_magnitude_spectrum_visualise"]=FFT_magnitude_spectrum_visualise
    #PowerSpectralDensity=10*np.log10(abs(fshift).^2)
    

    #get power spectral density 1d array
    PwrSpectralDensity=cv2.cvtColor(Colour_Resized, cv2.COLOR_BGR2GRAY)
    PwrSpectralDensity = GetPwrSpcDensity(PwrSpectralDensity)
    #if Index<3: ImageReviewDict["PwrSpectralDensity_visualise"]=PwrSpectralDensity

    #get a version of the Fourier magnitude that will work with the opencv phase correlation function to get similarity metric
    #this works with the fourier as it is positioned in the centre of the image - this wouldnt work well with
    #images that have translation and rotation differences
    #PhaseCorrelate_FourierMagImg=GetPhaseCorrelationReadyImage(FFT_magnitude_spectrum)
    #PhaseCorrelate_FourierMagImg = np.float32(PhaseCorrelate_FourierMagImg)
    #if Index < 3: ImageReviewDict["PhaseCorrelate_FourierMagImg visualise"] = cv2.convertScaleAbs(PhaseCorrelate_FourierMagImg)


    # PhaseCorrelate_Std=cv2.resize(Colour_Resized_blurred,(40,40))#PrepareMatchImages.USERFunction_ResizePercent(GrayScale_Resized,40)
    # #if we are using an image we hvae to convert to float
    # #can only do grayscale images
    # if len(PhaseCorrelate_Std.shape)==3:
    #     PhaseCorrelate_Std=cv2.cvtColor(PhaseCorrelate_Std, cv2.COLOR_BGR2GRAY)
    # PhaseCorrelate_Std = np.float32(PhaseCorrelate_Std)
    # #PhaseCorrelate_Std=GetPhaseCorrelationReadyImage(PhaseCorrelate_Std)
    # if Index<3: ImageReviewDict["PhaseCorrelate_Std visualise"]=cv2.resize(cv2.convertScaleAbs(PhaseCorrelate_Std),(300,300))






    if PrepareMatchImages.PreviewImagePrep==True and Index<1:
        #on first loop show image to user
        FM_DrawnKeypoints=_3DVisLabLib.draw_keypoints_v2(StackedColour_AndGradient_img.copy(),keypoints)
        ImageReviewDict["FM_DrawnKeypoints"]=FM_DrawnKeypoints

        for imagereviewimg in ImageReviewDict:
            Img=ImageReviewDict[imagereviewimg]
            print(imagereviewimg)
            _3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(Img,(Img.shape[1]*1,Img.shape[0]*1)),0,True,True)


    #load into image object
    #WARNING this is easy to overpopulate and block out the RAM, then PC will do pagefile stuff on the harddrive and reduce performance
    ImageInfo.EigenValues=[EigVal]
    ImageInfo.EigenVectors=[EigVec]
    ImageInfo.PrincpleComponents=[None]#PC
    ImageInfo.Histogram=[hist]
    ImageInfo.OriginalImage=[None]#InputImage#OriginalImage_col
    ImageInfo.ImageGrayscale=[None]#OriginalImage_GrayScale
    ImageInfo.ImageColour=[None]#Colour_Resized
    ImageInfo.ImageAdjusted=[None]#Image_For_FM
    ImageInfo.FM_Keypoints=[None]#keypoints
    ImageInfo.FM_Descriptors=[None]#descriptor
    ImageInfo.FourierTransform_mag=[FFT_magnitude_spectrum]
    ImageInfo.HOG_Mag=[None]#HOG_mag
    ImageInfo.HOG_Angle=[None]#HOG_angle
    ImageInfo.OPENCV_hog_descriptor=[OPENCV_hog_descriptor]
    ImageInfo.PhaseCorrelate_FourierMagImg=[PhaseCorrelate_Std]
    ImageInfo.DebugImage=[None]#DebugImage
    ImageInfo.OriginalImageFilePath=ImagePath
    ImageInfo.PwrSpectralDensity=[PwrSpectralDensity]
    ImageInfo.MacroStructure_img=[MacroStructure_img]
    ImageInfo.PCA_Struct_EigenVecs = [PCA_Struct_EigenVecs]
    ImageInfo.PCA_Struct_EigenVals = [PCA_Struct_EigenVals]
    return ImageInfo

def _Crop_Pixels(image,CropRangeY,CropRangeX):
    if len(image.shape)!=3:
        return image[CropRangeY[0]:CropRangeY[1],CropRangeX[0]:CropRangeX[1]]
    else:
        return image[CropRangeY[0]:CropRangeY[1],CropRangeX[0]:CropRangeX[1],:]


def PrepareImageMetrics_Faces(PrepareMatchImages,ImagePath,Index,ImageReviewDict):
    #create class object for each image
    ImageInfo=PrepareMatchImages.ImageInfo()

    #load in original image
    OriginalImage_col = cv2.imread(ImagePath)
    #bad code to convert gray to colour but still 3 channels
    if len(OriginalImage_col.shape)!=3:
        OriginalImage_col=cv2.cvtColor(OriginalImage_col,cv2.COLOR_GRAY2RGB)
        OriginalImage_GrayScale=OriginalImage_col.copy()
    else:
        #load as grayscale but still with 3 channels to be compatible with functions
        #very lazy code 
        OriginalImage_GrayScale_temp = cv2.cvtColor(OriginalImage_col, cv2.COLOR_BGR2GRAY)
        OriginalImage_GrayScale=OriginalImage_col.copy()
        OriginalImage_GrayScale[:,:,0]=OriginalImage_GrayScale_temp
        OriginalImage_GrayScale[:,:,1]=OriginalImage_GrayScale_temp
        OriginalImage_GrayScale[:,:,2]=OriginalImage_GrayScale_temp

    OriginalImage_col=OriginalImage_GrayScale.copy()

    InputImage=PrepareMatchImages.USERFunction_OriginalImage(OriginalImage_col)
    if Index<3: ImageReviewDict["OriginalImage_GrayScale"]=OriginalImage_GrayScale
    if Index<3: ImageReviewDict["OriginalImage_col"]=OriginalImage_col
    if Index<3: ImageReviewDict["InputImage"]=InputImage

    #create resized versions
    GrayScale_Resized=PrepareMatchImages.USERFunction_ResizePercent(OriginalImage_GrayScale,100)
    Colour_Resized=PrepareMatchImages.USERFunction_ResizePercent(OriginalImage_col,100)

    GrayScale_Resized=PrepareMatchImages.USERFunction_Crop_Pixels(GrayScale_Resized,(99,165),(49,118))#Y range then X range
    Colour_Resized=PrepareMatchImages.USERFunction_Crop_Pixels(Colour_Resized,(99,165),(49,118))#Y range then X range
    if Index<3: ImageReviewDict["Colour_Resized"]=Colour_Resized

    #get small image to experiment with macro structure matching
    temp_MacroStructure_img=PrepareMatchImages.USERFunction_Crop_Pixels(OriginalImage_col,(42,177),(59,119))
    MacroStructure_img = cv2.resize(temp_MacroStructure_img,(7,7))
    if Index < 3: ImageReviewDict["MacroStructure_img"] =  cv2.resize(MacroStructure_img,(270,270))#blow it up so we can see the preview- will be tiny otherwise

    # get PCA for structure
    temp_PCA=PrepareMatchImages.USERFunction_Crop_Pixels(OriginalImage_col,(42,177),(59,119))
    image_result, angle_list, eigenvectors_list, eigenvalues_list = PCA_Structure(temp_PCA)
    if Index < 3: ImageReviewDict["get PCA for structure"] = image_result
    PCA_Struct_EigenVecs = eigenvectors_list
    PCA_Struct_EigenVals = eigenvalues_list

    #create version for feature matching
    Image_For_FM=Colour_Resized.copy()#PrepareMatchImages.USERFunction_CropForFM(CoFlour_Resized)
    if Index<3: ImageReviewDict["Image_For_FM"]=Image_For_FM

    #create version for histogram matching
    Image_For_Histogram=Colour_Resized.copy()#PrepareMatchImages.USERFunction_CropForHistogram(Colour_Resized)
    Image_For_Histogram=PrepareMatchImages.USERFunction_ResizePercent(Image_For_Histogram,100)
    if Index<3: ImageReviewDict["Image_For_Histogram"]=Image_For_Histogram
    #get histogram for comparing colours
    hist = cv2.calcHist([Image_For_Histogram], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    #prepare image for HOG matching
    #image needs to be a particular size for HOG matching
    For_HOG_FeatureMatch=USERFunction_PrepareForHOG(Colour_Resized.copy())
    if Index<3: ImageReviewDict["For_HOG_FeatureMatch"]=For_HOG_FeatureMatch
    #this step is for visualisation
    HOG_mag,HOG_angle=GetHOG_featureVector(For_HOG_FeatureMatch)
    if Index<3: ImageReviewDict["HOG_mag visualised"]=cv2.convertScaleAbs(HOG_mag)
    #ensure input image is correct dimensions for HOG function
    if For_HOG_FeatureMatch.shape[0]!=128 and For_HOG_FeatureMatch.shape[0]!=64:
        raise Exception("Image not correct size for HOG (128 * 64)")
    else:
        #get histogram for HOG used for comparison during match matrix
        OPENCV_hog_descriptor=HOG_extrator.compute(For_HOG_FeatureMatch)
    
    #create extended image for feature matching to experiment with
    HOG_Style_Gradient_unnormalised,HOG_angle_temp=GetHOG_featureVector(Image_For_FM.copy())
    if Index<3: ImageReviewDict["HOG_Style_Gradient_unnormalised"]=HOG_Style_Gradient_unnormalised
    if Index<3: ImageReviewDict["HOG_Style_Gradient_visualised"]=cv2.convertScaleAbs(HOG_Style_Gradient_unnormalised)
    #try and force a normalised visulation for the magnitude image 
    #normal normalisation doesnt seem to work - do not know why yet
    StackedColour_AndGradient_img=PrepareMatchImages.StackTwoimages(Image_For_FM,HOG_Style_Gradient_unnormalised)
    if Index<3: ImageReviewDict["StackedColour_AndGradient_img"]=StackedColour_AndGradient_img

    #force image normalisation - not sure why this doesnt work with normal normalisation
    GradientImage=HOG_Style_Gradient_unnormalised.copy()#PrepareMatchImages.ForceNormalise_forHOG(HOG_Style_Gradient_unnormalised)
    if Index<3: ImageReviewDict["GradientImage visualise"]=cv2.convertScaleAbs(GradientImage)
    GradientImage_gray = cv2.cvtColor(GradientImage, cv2.COLOR_BGR2GRAY)
    if Index<3: ImageReviewDict["GradientImage_gray visualise"]=cv2.convertScaleAbs(GradientImage_gray)

    #for principle component analysis
    #has to have shape=3 (colour image)
    #Y range then X range
    Image_For_PCA=PrepareMatchImages.USERFunction_Crop_Pixels(OriginalImage_col,(42,177),(59,119))
    Image_For_PCA=PrepareMatchImages.USERFunction_ResizePercent(Image_For_PCA,90)
    PC,EigVal,EigVec=Get_PCA_(Image_For_PCA)
    if Index<3: ImageReviewDict["Image_For_PCA"]=Image_For_PCA

    #get feature match keypoints
    keypoints,descriptor=_3DVisLabLib.OrbKeyPointsOnly(StackedColour_AndGradient_img,PrepareMatchImages.FeatureMatch_Dict_Common.ORB_default)

    #get fourier transform
    #TODO don't we want fourier cofficients here?
    Image_fourier=cv2.cvtColor(Image_For_PCA,cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(Image_fourier)
    fshift = np.fft.fftshift(f)
    FFT_magnitude_spectrum = 20*np.log(np.abs(fshift))#magnitude is what we will use to compare

    #most of fourier is just noise - lets crop it
    CropRange=0.80#1.0=100%
    RangeX=int(FFT_magnitude_spectrum.shape[0]*CropRange)
    RangeY=int(FFT_magnitude_spectrum.shape[1]*CropRange)
    BufferX=int((FFT_magnitude_spectrum.shape[0]-RangeX)/2)
    BufferY=int((FFT_magnitude_spectrum.shape[1]-RangeY)/2)
    FFT_magnitude_spectrum=FFT_magnitude_spectrum[BufferX:-BufferX,BufferY:-BufferY]
    FFT_magnitude_spectrum_visualise=cv2.convertScaleAbs(FFT_magnitude_spectrum)
    if Index<3: ImageReviewDict["FFT_magnitude_spectrum_visualise"]=FFT_magnitude_spectrum_visualise
    #PowerSpectralDensity=10*np.log10(abs(fshift).^2)
    

    #get power spectral density 1d array

    PwrSpectralDensity= GetPwrSpcDensity(Image_For_PCA)
    #if Index<3: ImageReviewDict["PwrSpectralDensity_visualise"]=PwrSpectralDensity

    #get a version of the Fourier magnitude that will work with the opencv phase correlation function to get similarity metric
    #this works with the fourier as it is positioned in the centre of the image - this wouldnt work well with
    #images that have translation and rotation differences
    PhaseCorrelate_FourierMagImg=GetPhaseCorrelationReadyImage(FFT_magnitude_spectrum)
    if Index<3: ImageReviewDict["PhaseCorrelate_FourierMagImg visualise"]=cv2.convertScaleAbs(PhaseCorrelate_FourierMagImg)


    #PhaseCorrelate_Std=PrepareMatchImages.USERFunction_ResizePercent(GrayScale_Resized,40)
    #if we are using an image we hvae to convert to float
    #PhaseCorrelate_Std = PhaseCorrelate_Std.astype("float32")
    #PhaseCorrelate_Std=GetPhaseCorrelationReadyImage(PhaseCorrelate_Std)
    #if Index<3: ImageReviewDict["PhaseCorrelate_Std visualise"]=cv2.convertScaleAbs(PhaseCorrelate_Std)




    PhaseCorrelate_Std=cv2.resize(Colour_Resized,(25,25))#PrepareMatchImages.USERFunction_ResizePercent(GrayScale_Resized,40)
    #if we are using an image we hvae to convert to float
    #can only do grayscale images
    if len(PhaseCorrelate_Std.shape)==3:
        PhaseCorrelate_Std=cv2.cvtColor(PhaseCorrelate_Std, cv2.COLOR_BGR2GRAY)
    PhaseCorrelate_Std = np.float32(PhaseCorrelate_Std)
    #PhaseCorrelate_Std=GetPhaseCorrelationReadyImage(PhaseCorrelate_Std)
    if Index<3: ImageReviewDict["PhaseCorrelate_Std visualise"]=cv2.resize(cv2.convertScaleAbs(PhaseCorrelate_Std),(300,300))




    if PrepareMatchImages.PreviewImagePrep==True and Index<1:
        #on first loop show image to user
        FM_DrawnKeypoints=_3DVisLabLib.draw_keypoints_v2(StackedColour_AndGradient_img.copy(),keypoints)
        ImageReviewDict["FM_DrawnKeypoints"]=FM_DrawnKeypoints


        for imagereviewimg in ImageReviewDict:
            Img=ImageReviewDict[imagereviewimg]
            print(imagereviewimg)
            _3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(Img,(Img.shape[1]*1,Img.shape[0]*1)),0,True,True)
        



    #load into image object
    #WARNING this is easy to overpopulate and block out the RAM, then PC will do pagefile stuff on the harddrive and reduce performance
    ImageInfo.EigenValues=[EigVal]
    ImageInfo.EigenVectors=[EigVec]
    ImageInfo.PrincpleComponents=[None]#PC
    ImageInfo.Histogram=[hist]
    ImageInfo.OriginalImage=[None]#InputImage#OriginalImage_col
    ImageInfo.ImageGrayscale=[None]#OriginalImage_GrayScale
    ImageInfo.ImageColour=[None]#Colour_Resized
    ImageInfo.ImageAdjusted=[None]#Image_For_FM
    ImageInfo.FM_Keypoints=[None]#keypoints
    ImageInfo.FM_Descriptors=[None]#descriptor
    ImageInfo.FourierTransform_mag=[FFT_magnitude_spectrum]
    ImageInfo.HOG_Mag=[None]#HOG_mag
    ImageInfo.HOG_Angle=[None]#HOG_angle
    ImageInfo.OPENCV_hog_descriptor=[OPENCV_hog_descriptor]
    ImageInfo.PhaseCorrelate_FourierMagImg=[PhaseCorrelate_Std]
    ImageInfo.DebugImage=[None]#DebugImage
    ImageInfo.OriginalImageFilePath=ImagePath
    ImageInfo.PwrSpectralDensity=[PwrSpectralDensity]
    ImageInfo.MacroStructure_img=[MacroStructure_img]
    ImageInfo.PCA_Struct_EigenVecs = [PCA_Struct_EigenVecs]
    ImageInfo.PCA_Struct_EigenVals = [PCA_Struct_EigenVals]

    return ImageInfo

def GetHistogram_ColorNormalised(Image):
    raise Exception("GetHistogram_ColorNormalised Not developed yet - still too slow creating 2d histogram ")
    hsv = cv2.cvtColor(Image,cv2.COLOR_BGR2HSV)#colour 2d histogram needs conveted to hsv
    # channels = [0,1] because we need to process both H and S plane.
    # bins = [180,256] 180 for H plane and 256 for S plane.
    # range = [0,180,0,256] Hue value lies between 0 and 180 & Saturation lies between 0 and 256.
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
    
def Histogram_Stripes(Image,StripsPerDim,BinsPerChannel,Mask):
    #generates horizontal and vertical histogram stripes which may be compared to
    #overcome skewed and aspect ratio'd images
    Height=Image.shape[0]
    Width=Image.shape[1]
    Histogram_List=[]

    HeightStrips=math.floor(Height/StripsPerDim)
    WidthStrips=math.floor(Width/StripsPerDim)

    for WidthStripeIndex in range (0,Width,WidthStrips):
        #Image_for_slice=Image[:,WidthStripeIndex:WidthStripeIndex+WidthStrips]
        hist = cv2.calcHist([Image], [0, 1, 2], Mask, [BinsPerChannel, BinsPerChannel, BinsPerChannel],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        Histogram_List.append(hist)
    for HeightStripeIndex in range (0,Height,HeightStrips):
        #Image_for_slice=Image[HeightStripeIndex:HeightStripeIndex+HeightStrips,:]
        hist = cv2.calcHist([Image], [0, 1, 2], Mask, [BinsPerChannel, BinsPerChannel, BinsPerChannel],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        Histogram_List.append(hist)

    return Histogram_List

def StackedImg_Generator(ImageInfo_ref,IsTestImage,Metrics_dict):
    #get stack of images and associated analysis
    #IsTestImage set to TRUE will stop the program generating stacked images that wont be used (only need one reference)
    #colour and grayscale images, already resized 

    #don't write into object
    ImageInfo=copy.deepcopy(ImageInfo_ref)

    #clean out dictionary of images
    ImageInfo.Metrics_functions=dict()
    #populate according to metrics to be used
    #create as empty lists so we can append to them
    for MetricsItem in Metrics_dict:
        ImageInfo.Metrics_functions[MetricsItem]=[]

    #load in original image
    OriginalImage_col = cv2.imread(ImageInfo.OriginalImageFilePath)

    if OriginalImage_col is None:
        ImageInfo.IsInError==True
        print("Empty Image",ImageInfo.OriginalImageFilePath)
        return ImageInfo

    #bad code to convert gray to colour but still 3 channels
    if len(OriginalImage_col.shape)!=3:
        OriginalImage_col=cv2.cvtColor(OriginalImage_col,cv2.COLOR_GRAY2RGB)


    #even if just one image its still in list format to keep logic common
    #ImageColour = Resize_toPixel_keepRatio(OriginalImage_col, 120, 120)
    #NOTES
    #ImageColour=_Crop_Pixels(OriginalImage_col,(0,63),(0,130))#Y range then X range
    #FACES
    #ImageColour=_Crop_Pixels(OriginalImage_col,(35,190),(25,160))##Y range then X range

    #faces vs eveything
    # if IsTestImage==True:
    #     ImageColour=_Crop_Pixels(OriginalImage_col,(35,190),(25,160))##Y range then X range
    #     ImageColour= cv2.resize(ImageColour, (120, 120))
    # else:
    #     ImageColour= cv2.resize(OriginalImage_col, (120, 120))

    ImageColour= cv2.resize(OriginalImage_col, (120, 120))

    #ImageGrayscale=Resize_toPixel_keepRatio(ImageInfo.ImageGrayscale[0], 120, 120)

    Canvas,List_CroppingImg=CreateCropInMatrixOfImage(ImageColour,100,100,0,False)
    KernelSize=5
    kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing

    
    #_3DVisLabLib.ImageViewer_Quick_no_resize(Canvas,0,True,True)

    #HISTOGRAM OF ORIENTATED GRADIENTS FEATURE MATCH
    if "HM_data_HOG_Dist" in Metrics_dict:
        #don't need to do at every scale - its scale invarient
        For_HOG_FeatureMatch=USERFunction_PrepareForHOG(ImageColour)
        #ensure input image is correct dimensions for HOG function
        if For_HOG_FeatureMatch.shape[0]!=128 and For_HOG_FeatureMatch.shape[0]!=64:
            raise Exception("Image not correct size for HOG (128 * 64)")
        else:
            #get histogram for HOG used for comparison during match matrix
            OPENCV_hog_descriptor=HOG_extrator.compute(For_HOG_FeatureMatch)
        ImageInfo.Metrics_functions["HM_data_HOG_Dist"].append(OPENCV_hog_descriptor)

    
    #FEATURE MATCHER
    if "HM_data_FM" in Metrics_dict:
        keypoints,descriptor=_3DVisLabLib.OrbKeyPointsOnly(ImageColour,FeatureMatch_Dict_Common.ORB_default)
        ImageInfo.Metrics_functions["HM_data_FM"].append((keypoints,descriptor))


    #get various crops into image
    #we have to load the list up properly - they should come in here as "[NONE]"

    for Img_colour in List_CroppingImg:
        #get grayscale
        Img_grayscale=cv2.cvtColor(Img_colour, cv2.COLOR_BGR2GRAY)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(Img_colour,0,True,True)
        if "HM_data_HistogramCentralis" in Metrics_dict:
            #create a masked histogram with just central position
            global HistogramCentralis_mask
            if HistogramCentralis_mask is None:#check global object is populated with mask or not
                HistogramCentralis_mask = np.zeros((Img_colour.shape[0],Img_colour.shape[1],3), np.uint8)
                #get radius
                Diameter=min((HistogramCentralis_mask.shape[0]),(HistogramCentralis_mask.shape[1]))
                Radius=int((Diameter/2)*1.0)#percentage of smallest dimension (1=100%)
                cv2.circle(HistogramCentralis_mask,(int(HistogramCentralis_mask.shape[1]/2),int(HistogramCentralis_mask.shape[0]/2)), Radius, (255,255,255), -1)
                HistogramCentralis_mask = cv2.cvtColor(HistogramCentralis_mask, cv2.COLOR_BGR2GRAY)
                #_3DVisLabLib.ImageViewer_Quick_no_resize(HistogramCentralis_mask,0,True,True)
            hist = cv2.calcHist([Img_colour], [0, 1, 2], HistogramCentralis_mask, [8, 8, 8],[0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            ImageInfo.Metrics_functions["HM_data_HistogramCentralis"].append(hist)
            
    
        #HISTOGRAM
        if "HM_data_histo" in Metrics_dict:
            hist = cv2.calcHist([Img_colour], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
            #hsv conversion very slow
            #hsv = cv2.cvtColor(Img_colour,cv2.COLOR_BGR2HSV)#colour 2d histogram needs conveted to hsv
            # channels = [0,1] because we need to process both H and S plane.
            # bins = [180,256] 180 for H plane and 256 for S plane.
            # range = [0,180,0,256] Hue value lies between 0 and 180 & Saturation lies between 0 and 256.
            #hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            ImageInfo.Metrics_functions["HM_data_histo"].append(hist)
            
        #MACROSTRUCTURE
        if "HM_data_MacroStructure" in Metrics_dict:
        #get small image to experiment with macro structure matching
            
            MacroStructure_img = cv2.resize(Img_colour, (19, 19))
            KernelSize=3
            kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing
            MacroStructure_img = cv2.filter2D(MacroStructure_img,-1,kernel)
            #MacroStructure_img=MacroStructure_img/np.sqrt(np.sum(MacroStructure_img**2))
            ImageInfo.Metrics_functions["HM_data_MacroStructure"].append(MacroStructure_img)
            #_3DVisLabLib.ImageViewer_Quick_no_resize(MacroStructure_img,0,True,True)


        #HISTOGRAM STRIPING - vertical and horizontal histogram 
        if "HM_data_HistogramStriping" in Metrics_dict:
            global HistogramStripulus_Centralis
            if HistogramStripulus_Centralis is None:#check global object is populated with mask or not
               HistogramStripulus_Centralis = np.zeros((Img_colour.shape[0],Img_colour.shape[1],3), np.uint8)
               #get radius
               Diameter=min((HistogramStripulus_Centralis.shape[0]),(HistogramStripulus_Centralis.shape[1]))
               Radius=int((Diameter/2)*1)#percentage of smallest dimension (1=100%)
               cv2.circle(HistogramStripulus_Centralis,(int(HistogramStripulus_Centralis.shape[1]/2),int(HistogramStripulus_Centralis.shape[0]/2)), Radius, (255,255,255), -1)
               HistogramStripulus_Centralis = cv2.cvtColor(HistogramStripulus_Centralis, cv2.COLOR_BGR2GRAY)
            #MacroStructure_img = cv2.resize(Img_colour, (25, 25))
            #_3DVisLabLib.ImageViewer_Quick_no_resize(HistogramStripulus_Centralis,0,True,True)
            HistoStripes = cv2.filter2D(Img_colour,-1,kernel)
            HistoStripes=Histogram_Stripes(HistoStripes,9,16,HistogramStripulus_Centralis)
            ImageInfo.Metrics_functions["HM_data_HistogramStriping"].append(HistoStripes)

        #POWER SPECTRAL DENSITY
        if "HM_data_FourierPowerDensity" in Metrics_dict:
            PwrSpectralDensity= GetPwrSpcDensity(Img_colour)
            ImageInfo.Metrics_functions["HM_data_FourierPowerDensity"].append(PwrSpectralDensity)

        #PRINCPLE COMPONENT ANALYSIS:STRUCTURE
        if ("HM_data_StructuralPCA_dotProd" in Metrics_dict) or ("HM_data_StructuralPCA_VectorValue" in Metrics_dict):
            image_result,angle_list,eigenvectors_list,eigenvalues_list=PCA_Structure(Img_grayscale)
            if "HM_data_StructuralPCA_dotProd" in Metrics_dict:
                ImageInfo.Metrics_functions["HM_data_StructuralPCA_dotProd"].append(eigenvectors_list)
            if "HM_data_StructuralPCA_VectorValue" in Metrics_dict:
                ImageInfo.Metrics_functions["HM_data_StructuralPCA_VectorValue"].append(eigenvalues_list)

        #FOURIER MAGNITUDE DIFFERENCE
        if "HM_data_FourierDifference" in Metrics_dict:
            #have one image we are testing the sequenced images against
            ImageInfo.Metrics_functions["HM_data_FourierDifference"].append(GetFFT_OfImage(Img_grayscale,80,True))

        #PHASE CORRELATION IMAGE
        if "HM_data_PhaseCorrelation" in Metrics_dict:
            if len(Img_grayscale.shape)==3:
                Img_grayscale=cv2.cvtColor(Img_grayscale, cv2.COLOR_BGR2GRAY)
            
            Img_grayscale = cv2.filter2D(Img_grayscale,-1,kernel)#smoothing might help with cross correllation
            PhaseCorrelate_Std = np.float32(Img_grayscale)
            ImageInfo.Metrics_functions["HM_data_PhaseCorrelation"].append(PhaseCorrelate_Std)


        if "HM_data_EigenValueDifference" in Metrics_dict or "HM_data_EigenVectorDotProd" in Metrics_dict:
            #has to be colour image
            PC,EigVal,EigVec=Get_PCA_(Img_colour)
            if "HM_data_EigenVectorDotProd" in Metrics_dict:
                ImageInfo.Metrics_functions["HM_data_EigenVectorDotProd"].append(EigVec)
            if "HM_data_EigenValueDifference" in Metrics_dict:
                ImageInfo.Metrics_functions["HM_data_EigenValueDifference"].append(EigVal)

        if "HM_data_TemplateMatching" in Metrics_dict:
            ImageInfo.Metrics_functions["HM_data_TemplateMatching"].append(Img_colour)

        #if in match image mode (folder of images to match - its not necessary to calculate sequential images as only first will be used)
        if IsTestImage==True:
            break


    return ImageInfo

def TestImage(InputImage):
    try:
            
        OriginalImage_col = cv2.imread(InputImage)
        if OriginalImage_col is None:
            print("Invalid as Image, ignoring: ",InputImage)
            return False
        #try and force an exception if a bad image
        TestX=OriginalImage_col.shape[0]
        TestY=OriginalImage_col.shape[1]
        if TestX<1 or TestY<1:
            print("Invalid Image(too small), ignoring: ",InputImage)
            return False
    except:
        print("Bad Image, ignoring: ",InputImage)
        return False
    
    return True

def PrepareImageMetrics_MultipleImgs(PrepareMatchImages,ImagePath,Index,ImageReviewDict):
    #load image into memory - do image processing on demand
    ImageInfo=PrepareMatchImages.ImageInfo()

    #only need to populate the filename 
    ImageInfo.OriginalImageFilePath=ImagePath
    
    #treat this image differently potentially if its the one we want to match
    if ImagePath in PrepareMatchImages.List_ImagesToMatchFIlenames.values():
        ImageInfo.is_ImageToMatch=True
    else:
        ImageInfo.is_ImageToMatch=False

    #user option to process items on the fly or not
    if PrepareMatchImages.ProcessImagesOnFly==True:
        ImageInfo.ProcessImages_function=StackedImg_Generator
    else:
        #set function to none so similarity processor knows data exists for image
        ImageInfo.ProcessImages_function=None
        #function to generate stack of images and populated into input object
        ImageInfo=StackedImg_Generator(ImageInfo,False,PrepareMatchImages.Metrics_dict)

    return ImageInfo


def PrepareImageMetrics_NotesSide(PrepareMatchImages,ImagePath,Index,ImageReviewDict):
    #create class object for each image
    ImageInfo=PrepareMatchImages.ImageInfo()

    #load in original image
    OriginalImage_col = cv2.imread(ImagePath)
    #bad code to convert gray to colour but still 3 channels
    if len(OriginalImage_col.shape)!=3:
        OriginalImage_col=cv2.cvtColor(OriginalImage_col,cv2.COLOR_GRAY2RGB)
        OriginalImage_GrayScale=OriginalImage_col.copy()
    else:
        #load as grayscale but still with 3 channels to be compatible with functions
        #very lazy code 
        OriginalImage_GrayScale_temp = cv2.cvtColor(OriginalImage_col, cv2.COLOR_BGR2GRAY)
        OriginalImage_GrayScale=OriginalImage_col.copy()
        OriginalImage_GrayScale[:,:,0]=OriginalImage_GrayScale_temp
        OriginalImage_GrayScale[:,:,1]=OriginalImage_GrayScale_temp
        OriginalImage_GrayScale[:,:,2]=OriginalImage_GrayScale_temp

   

    InputImage=PrepareMatchImages.USERFunction_OriginalImage(OriginalImage_col)
    if Index<3: ImageReviewDict["OriginalImage_GrayScale"]=OriginalImage_GrayScale
    if Index<3: ImageReviewDict["OriginalImage_col"]=OriginalImage_col
    if Index<3: ImageReviewDict["InputImage"]=InputImage

    #create resized versions
    GrayScale_Resized=PrepareMatchImages.USERFunction_ResizePercent(OriginalImage_GrayScale,100)
    Colour_Resized=PrepareMatchImages.USERFunction_ResizePercent(OriginalImage_col,100)
    GrayScale_Resized=PrepareMatchImages.USERFunction_Crop_Pixels(GrayScale_Resized,(0,65),(0,120))#Y range then X range
    Colour_Resized=PrepareMatchImages.USERFunction_Crop_Pixels(Colour_Resized,(0,65),(0,120))#Y range then X range
    if Index<3: ImageReviewDict["Colour_Resized cropped"]=Colour_Resized


    #get PCA for structure
    image_result,angle_list,eigenvectors_list,eigenvalues_list=PCA_Structure(GrayScale_Resized)
    if Index < 3: ImageReviewDict["get PCA for structure"] = image_result
    PCA_Struct_EigenVecs=eigenvectors_list
    PCA_Struct_EigenVals=eigenvalues_list

    #create version for feature matching
    Image_For_FM=Colour_Resized.copy()#PrepareMatchImages.USERFunction_CropForFM(CoFlour_Resized)
    if Index<3: ImageReviewDict["Image_For_FM"]=Image_For_FM

    # get small image to experiment with macro structure matching
    MacroStructure_img = cv2.resize(Colour_Resized, (7, 7))
    if Index < 3: ImageReviewDict["MacroStructure_img"] = cv2.resize(MacroStructure_img, (270, 270))  # blow it up so we can see the preview- will be tiny otherwise

    #create version for histogram matching
    Image_For_Histogram=Colour_Resized.copy()#PrepareMatchImages.USERFunction_CropForHistogram(Colour_Resized)
    Image_For_Histogram=PrepareMatchImages.USERFunction_ResizePercent(Image_For_Histogram,50)
    if Index<3: ImageReviewDict["Image_For_Histogram"]=Image_For_Histogram
    #get histogram for comparing colours
    hist = cv2.calcHist([Image_For_Histogram], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    #prepare image for HOG matching
    #image needs to be a particular size for HOG matching
    For_HOG_FeatureMatch=USERFunction_PrepareForHOG(Colour_Resized.copy())
    if Index<3: ImageReviewDict["For_HOG_FeatureMatch"]=For_HOG_FeatureMatch
    #this step is for visualisation
    HOG_mag,HOG_angle=GetHOG_featureVector(For_HOG_FeatureMatch)
    if Index<3: ImageReviewDict["HOG_mag visualised"]=cv2.convertScaleAbs(HOG_mag)
    #ensure input image is correct dimensions for HOG function
    if For_HOG_FeatureMatch.shape[0]!=128 and For_HOG_FeatureMatch.shape[0]!=64:
        raise Exception("Image not correct size for HOG (128 * 64)")
    else:
        #get histogram for HOG used for comparison during match matrix
        OPENCV_hog_descriptor=HOG_extrator.compute(For_HOG_FeatureMatch)
    
    #create extended image for feature matching to experiment with
    HOG_Style_Gradient_unnormalised,HOG_angle_temp=GetHOG_featureVector(Image_For_FM.copy())
    if Index<3: ImageReviewDict["HOG_Style_Gradient_unnormalised"]=HOG_Style_Gradient_unnormalised
    if Index<3: ImageReviewDict["HOG_Style_Gradient_visualised"]=cv2.convertScaleAbs(HOG_Style_Gradient_unnormalised)
    #try and force a normalised visulation for the magnitude image 
    #normal normalisation doesnt seem to work - do not know why yet
    StackedColour_AndGradient_img=PrepareMatchImages.StackTwoimages(Image_For_FM,HOG_Style_Gradient_unnormalised)
    if Index<3: ImageReviewDict["StackedColour_AndGradient_img"]=StackedColour_AndGradient_img

    #force image normalisation - not sure why this doesnt work with normal normalisation
    GradientImage=HOG_Style_Gradient_unnormalised.copy()#PrepareMatchImages.ForceNormalise_forHOG(HOG_Style_Gradient_unnormalised)
    if Index<3: ImageReviewDict["GradientImage visualise"]=cv2.convertScaleAbs(GradientImage)
    GradientImage_gray = cv2.cvtColor(GradientImage, cv2.COLOR_BGR2GRAY)
    if Index<3: ImageReviewDict["GradientImage_gray visualise"]=cv2.convertScaleAbs(GradientImage_gray)

    #for principle component analysis
    #has to have shape=3 (colour image)
    Image_For_PCA=PrepareMatchImages.USERFunction_Crop_Pixels(Colour_Resized,(0,65),(0,65))
    Image_For_PCA=PrepareMatchImages.USERFunction_ResizePercent(Image_For_PCA,90)
    PC,EigVal,EigVec=Get_PCA_(Image_For_PCA)
    if Index<3: ImageReviewDict["Image_For_PCA"]=Image_For_PCA

    #get feature match keypoints
    keypoints,descriptor=_3DVisLabLib.OrbKeyPointsOnly(StackedColour_AndGradient_img,PrepareMatchImages.FeatureMatch_Dict_Common.ORB_default)

    #get fourier transform
    #TODO don't we want fourier cofficients here?
    f = np.fft.fft2(GradientImage_gray)
    fshift = np.fft.fftshift(f)
    FFT_magnitude_spectrum = 20*np.log(np.abs(fshift))#magnitude is what we will use to compare

    #most of fourier is just noise - lets crop it
    CropRange=0.80#1.0=100%
    RangeX=int(FFT_magnitude_spectrum.shape[0]*CropRange)
    RangeY=int(FFT_magnitude_spectrum.shape[1]*CropRange)
    BufferX=int((FFT_magnitude_spectrum.shape[0]-RangeX)/2)
    BufferY=int((FFT_magnitude_spectrum.shape[1]-RangeY)/2)
    FFT_magnitude_spectrum=FFT_magnitude_spectrum[BufferX:-BufferX,BufferY:-BufferY]
    
    #a lot of noise - lets see if we can remove noise
    KernelSize=5
    kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing - maybe make smoother as such small images
    dst = cv2.filter2D(FFT_magnitude_spectrum,-1,kernel)
    FFT_magnitude_spectrum_visualise=cv2.convertScaleAbs(dst)
    if Index<3: ImageReviewDict["FFT_magnitude_spectrum_visualise"]=FFT_magnitude_spectrum_visualise
    #PowerSpectralDensity=10*np.log10(abs(fshift).^2)
    

    #get power spectral density 1d array

    PwrSpectralDensity= GetPwrSpcDensity(Image_For_Histogram)
    #if Index<3: ImageReviewDict["PwrSpectralDensity_visualise"]=PwrSpectralDensity

    #get a version of the Fourier magnitude that will work with the opencv phase correlation function to get similarity metric
    #this works with the fourier as it is positioned in the centre of the image - this wouldnt work well with
    #images that have translation and rotation differences
    PhaseCorrelate_FourierMagImg=GetPhaseCorrelationReadyImage(FFT_magnitude_spectrum)
    if Index<3: ImageReviewDict["PhaseCorrelate_FourierMagImg visualise"]=cv2.convertScaleAbs(PhaseCorrelate_FourierMagImg)



    PhaseCorrelate_Std=cv2.resize(Colour_Resized,(25,25))#PrepareMatchImages.USERFunction_ResizePercent(GrayScale_Resized,40)
    #if we are using an image we hvae to convert to float
    #can only do grayscale images
    if len(PhaseCorrelate_Std.shape)==3:
        PhaseCorrelate_Std=cv2.cvtColor(PhaseCorrelate_Std, cv2.COLOR_BGR2GRAY)
    PhaseCorrelate_Std = np.float32(PhaseCorrelate_Std)
    #PhaseCorrelate_Std=GetPhaseCorrelationReadyImage(PhaseCorrelate_Std)
    if Index<3: ImageReviewDict["PhaseCorrelate_Std visualise"]=cv2.resize(cv2.convertScaleAbs(PhaseCorrelate_Std),(300,300))



    if PrepareMatchImages.PreviewImagePrep==True and Index<1:
        #on first loop show image to user
        FM_DrawnKeypoints=_3DVisLabLib.draw_keypoints_v2(StackedColour_AndGradient_img.copy(),keypoints)
        ImageReviewDict["FM_DrawnKeypoints"]=FM_DrawnKeypoints


        for imagereviewimg in ImageReviewDict:
            Img=ImageReviewDict[imagereviewimg]
            print(imagereviewimg)
            _3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(Img,(Img.shape[1]*1,Img.shape[0]*1)),0,True,True)
        


    #load into image object
    #WARNING this is easy to overpopulate and block out the RAM, then PC will do pagefile stuff on the harddrive and reduce performance
    ImageInfo.EigenValues=[EigVal]
    ImageInfo.EigenVectors=[EigVec]
    ImageInfo.PrincpleComponents=[None]#PC
    ImageInfo.Histogram=[hist]
    ImageInfo.OriginalImage=[None]#InputImage#OriginalImage_col
    ImageInfo.ImageGrayscale=[None]#OriginalImage_GrayScale
    ImageInfo.ImageColour=[None]#Colour_Resized
    ImageInfo.ImageAdjusted=[None]#Image_For_FM
    ImageInfo.FM_Keypoints=[keypoints]
    ImageInfo.FM_Descriptors=[descriptor]
    ImageInfo.FourierTransform_mag=[FFT_magnitude_spectrum]
    ImageInfo.HOG_Mag=[None]#HOG_mag
    ImageInfo.HOG_Angle=[None]#HOG_angle
    ImageInfo.OPENCV_hog_descriptor=[OPENCV_hog_descriptor]
    ImageInfo.PhaseCorrelate_FourierMagImg=[PhaseCorrelate_Std]
    ImageInfo.DebugImage=[None]#DebugImage
    ImageInfo.OriginalImageFilePath=ImagePath
    ImageInfo.PwrSpectralDensity=[PwrSpectralDensity]
    ImageInfo.MacroStructure_img=[MacroStructure_img]
    ImageInfo.PCA_Struct_EigenVecs = [PCA_Struct_EigenVecs]
    ImageInfo.PCA_Struct_EigenVals = [PCA_Struct_EigenVals]
    
    return ImageInfo

def GetHOG_featureVector(image):
    #bad code to convert gray to colour - this is inefficient when we process similarity
    if len(image.shape)!=3:
        img=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        

    img = np.float32(image) / 255.0
    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    # Python Calculate gradient magnitude and direction ( in degrees )
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    Return_norm = cv2.normalize(mag, mag,0, 255, cv2.NORM_MINMAX)
    return Return_norm, angle

def Get_PCA_(InputImage):
    #https://towardsdatascience.com/principal-component-analysis-in-depth-understanding-through-image-visualization-892922f77d9f
    #n_bands=Pod1Image_col.shape[2]-1
    # 3 dimensional dummy array with zeros
    #multi channel PCA
    PCA_image=InputImage

    #bad code to convert gray to colour - this is inefficient when we process similarity
    if len(PCA_image.shape)!=3:
        PCA_image=cv2.cvtColor(PCA_image,cv2.COLOR_GRAY2RGB)
        
    #test if square matrix
    if InputImage.shape[0]!=InputImage.shape[1]:
        print("WARNING! Input Image to PCA not square, PCA will give invalid results")


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

    ReturnSize=min(len(EigVal),6)
    return PC,EigVal[0:ReturnSize],EigVec[0:ReturnSize]

def Resize_toPixel_keepRatio(Image,PixelsY,PixelsX):
    #resize image keeping aspect ratio, fill out rest of image with blurred resized version without maintaining aspect ratio
    FillImage=cv2.resize(Image,(PixelsX,PixelsY))
    FillImage=cv2.blur(src=FillImage, ksize=(15, 15))
    #fillimage will be canvas
    #lazy code, resize by Y and if it doesnt fit resize again :3
    PercenttoResize=PixelsY/Image.shape[0]
    Resized=cv2.resize(Image,(int(Image.shape[1]*PercenttoResize),int(Image.shape[0]*PercenttoResize)))
    #redo this and resize to longest vs shortest dims
    if Resized.shape[1]>PixelsX:
        PercenttoResize=PixelsX/Image.shape[1]
        Resized=cv2.resize(Image,(int(Image.shape[1]*PercenttoResize),int(Image.shape[0]*PercenttoResize)))
    #if colour/grayscale have to treat differently
    #place in canvas
    if len(Image.shape)==3:
        FillImage[0:Resized.shape[0],0:Resized.shape[1],:]=Resized[:,:,:]
    else:#grayscale
        FillImage[0:Resized.shape[0],0:Resized.shape[1]]=Resized[:,:]
    #_3DVisLabLib.ImageViewer_Quick_no_resize(FillImage,0,True,True)
    return FillImage
    
def CreateCropInMatrixOfImage(Image,StartCropPc_100pc,EndCropPC_100pc,steps,PolarWrapMode):
    #for an input image, successively crop into the image in stages and create horizontal
    #quilt
    #if PolarWrapMode==True:
    #    raise Exception("CreateCropInMatrixOfImage, this function not yet developed")
    if steps==0:
        return Image,[Image]#return canvas and list of image

    if len(Image.shape)!=3:
        raise Exception("CreateCropInMatrixOfImage, please use colour image as input, or convert to 3 channels")

    if StartCropPc_100pc>400:raise Exception("Currently cannot support CreateCropInMatrixOfImage of greater than 200%")
    if EndCropPC_100pc>100:raise Exception("Currently cannot support CreateCropInMatrixOfImage of greater than 100%")


    StartCropPc=StartCropPc_100pc/100
    EndCropPC=EndCropPC_100pc/100


    
    
    #if we are cropping out(zoom out) - have to create a start image with bigger canvas and fill in the pixels somehow
    #  
    if StartCropPc_100pc>100:
        #create bigger canvas
        #create random colour noise canvas so we dont blow out the histogram - TODO probably better sending back a mask
        RandomNoiseImg=np.random.randint(255, size=(int(StartCropPc*Image.shape[0]), int(StartCropPc*Image.shape[1]),3),dtype="uint8")
        KernelSize=3
        kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing - maybe make smoother as such small images
        RandomNoiseImg=cv2.filter2D(RandomNoiseImg,-1,kernel)#blur it
        #now place the original image in the middle of this random pixel colour field
        OffsetX=int((RandomNoiseImg.shape[0]-Image.shape[0])/2)#how far off from the left
        OffsetY=int((RandomNoiseImg.shape[1]-Image.shape[1])/2)#how far off from the top
        #now drop the input image into the center of the larger image
        RandomNoiseImg[OffsetX:Image.shape[0]+OffsetX,OffsetY:Image.shape[1]+OffsetY,:]=Image[:,:,:]
        #redefine input image to be bigger canvas
        #RandomNoiseImg=cv2.resize(RandomNoiseImg,(Image.shape[1],Image.shape[0]))
        OperationalImage=RandomNoiseImg
        #set the start crop size at 100 % - this is a pretty shonky way to do this - not truly dynamic
        StartCropPc_100pc=100
        StartCropPc=1.0
        #_3DVisLabLib.ImageViewer_Quick_no_resize(OperationalImage,0,True,True)
    else:
        OperationalImage=Image



    #for each crop, resize it back to original input image dimensions
    #create canvas - stretch it horizontally - matrix maybe isnt necessary
    Canvas=cv2.resize(Image,(Image.shape[1]*steps,Image.shape[0]))
    #_3DVisLabLib.ImageViewer_Quick_no_resize(Canvas,0,True,True)
    StepsX=np.linspace(StartCropPc*OperationalImage.shape[1],EndCropPC*Image.shape[1],steps)
    StepsY=np.linspace(StartCropPc*OperationalImage.shape[0],EndCropPC*Image.shape[0],steps)
    StepsX = [int(x) for x in StepsX]
    StepsY = [int(x) for x in StepsY]

    #get offset from side
    SideOffsetX=[]
    SideOffsetY=[]
    for StepIndex,StepACtion in enumerate(StepsX):
        SideOffsetX.append((OperationalImage.shape[1]-StepsX[StepIndex]))
        SideOffsetY.append((OperationalImage.shape[0]-StepsY[StepIndex]))

    #convert to ints
    SideOffsetX = [int(x) for x in SideOffsetX]
    SideOffsetY = [int(x) for x in SideOffsetY]
    


    #for quilted image use original image size
    List_CroppingImg=[]
    for stepaction in range(0,steps):
        Xposition_start=stepaction*Image.shape[1]#position horizontally
        Xposition_end=(stepaction+1)*Image.shape[1]

       
        #crop into input image
        CroppedImage=OperationalImage[SideOffsetY[stepaction]:StepsY[stepaction],SideOffsetX[stepaction]:StepsX[stepaction],:]
        
        #make sure not too cropped
        if CroppedImage.shape[0]==0 or CroppedImage.shape[1]==0:
            print("Warning:CreateCropInMatrixOfImage  overcrop, image is null, please check parameters. Using dummy data")
            #use dummy data
            List_CroppingImg.append(Image)
            continue
        #_3DVisLabLib.ImageViewer_Quick_no_resize(CroppedImage,0,True,True)
        if PolarWrapMode==True:
             CroppedImage=GetPhaseCorrelationReadyImage(CroppedImage)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(CroppedImage,0,True,True)
        #resize back to original input shape
        CroppedImage_resize=cv2.resize(CroppedImage,(Image.shape[1],Image.shape[0]))
        List_CroppingImg.append(CroppedImage_resize)
        if PolarWrapMode==False:
            Canvas[:,Xposition_start:Xposition_end,:]=CroppedImage_resize
        else:
            Canvas[:,Xposition_start:Xposition_end,0]=CroppedImage_resize
            Canvas[:,Xposition_start:Xposition_end,1]=CroppedImage_resize
            Canvas[:,Xposition_start:Xposition_end,2]=CroppedImage_resize

    #_3DVisLabLib.ImageViewer_Quick_no_resize(Canvas,0,True,True)
    return Canvas,List_CroppingImg



    

def GetPhaseCorrelationReadyImage(Image):
    #experiment with using rotational cross correlation type approach to the fourier magnitude
    #similiarity, must be an easier way to reinterpret fourier as 1D - like using fourier cofficients
    #https://stackoverflow.com/questions/57801071/get-rotational-shift-using-phase-correlation-and-log-polar-transform
    if len(Image.shape)==3:
        Image=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
    
    base_img = Image
    (h, w) = base_img.shape
    
    (cX, cY) = (w // 2, h // 2)
    base_polar = cv2.linearPolar(base_img,(cX, cY), min(cX, cY), 0)
    return base_polar

def TestImagePhaseCorr(Image):
    #experiment with using rotational cross correlation type approach to the fourier magnitude
    #similiarity, must be an easier way to reinterpret fourier as 1D - like using fourier cofficients
    #https://stackoverflow.com/questions/57801071/get-rotational-shift-using-phase-correlation-and-log-polar-transform
    base_img = Image
    base_img = np.float32(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)) / 255.0

    (h, w) = base_img.shape
    (cX, cY) = (w // 2, h // 2)

    angle = 38
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    curr_img = cv2.warpAffine(base_img, M, (w, h))

    cv2.imshow("base_img", base_img)
    cv2.imshow("curr_img", curr_img)

    base_polar = cv2.linearPolar(base_img,(cX, cY), min(cX, cY), 0)
    curr_polar = cv2.linearPolar(curr_img,(cX, cY), min(cX, cY), 0) 

    cv2.imshow("base_polar", base_polar)
    cv2.imshow("curr_polar", curr_polar)

    (sx, sy), sf = cv2.phaseCorrelate(base_polar, curr_polar)

    rotation = -sy / h * 360
    print(rotation) 

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def TemplateMatch():
    #res = cv2.matchTemplate(MainImage,TemplateImage,cv2.TM_SQDIFF_NORMED)
    pass

def GetPwrSpcDensity(image):
    #https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/

    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    npix = image.shape[0]

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Length=min(len(fourier_amplitudes),len(knrm))
    Abins, _, _ = stats.binned_statistic(knrm[0:Length], fourier_amplitudes[0:Length],
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return Abins


def FFt(inputdata):
    #https://stackoverflow.com/questions/30527902/numpy-fft-fast-fourier-transformation-of-1-dimensional-array
    img = (inputdata)
    f = np.fft.fft(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = (np.abs(fshift))

    pl.subplot(121)
    pl.plot(img)
    pl.title('Input Image')
    pl.xticks([]), pl.yticks([])

    pl.subplot(122)
    pl.plot(magnitude_spectrum)
    pl.title('Magnitude Spectrum')
    pl.xticks([]), pl.yticks([])

    pl.show()

def PairWise_Matching(MatchImages,PlotAndSave_2datas,PlotAndSave,ImgCol_InfoSheet_Class):

    #create reference object of filename VS ID
    ImgNameV_ID=dict()
    #roll through and pull out images vs ID
    for BaseImageList in MatchImages.ImagesInMem_Pairing:
        if len(MatchImages.ImagesInMem_Pairing[BaseImageList][0])!=1:
            raise Exception("MatchImages.ImagesInMem_Pairing",BaseImageList," error, not correct number of images (1)")
        FileName=MatchImages.ImagesInMem_Pairing[BaseImageList][0][0]
        ImgNameV_ID[BaseImageList]=FileName


    #copy input object which should only have 1 image per ID
    ImagesInMem_Pairing=copy.deepcopy(MatchImages.ImagesInMem_Pairing)

    #create indexed dictionary of images referenced by ID so we can start combining lists of images
    #dictionary key has no relevance - only image IDs which are the ID of the result matrices
    Pairings_Indexes=dict()
    for OuterIndex, img in enumerate(MatchImages.ImagesInMem_to_Process):
        ImgCol_InfoSheet=ImgCol_InfoSheet_Class()
        Pairings_Indexes["NOTIMG"+ str(OuterIndex) + "NOTIMG"]=([OuterIndex],ImgCol_InfoSheet)


    #record similarity scores
    MatchMetric_all=[]
    MatchMetric_Std_PerList=[]
    MatchMetric_mean_PerList=[]
    
    for Looper in range (0,2):
        #lame way of marking refinement loops
        MatchMetric_Std_PerList.append(3)
        MatchMetric_mean_PerList.append(3)
        #roll through all list of images
        for OuterIndex,BaseImgList in enumerate(Pairings_Indexes):
            
            #print(BaseImgList,"/",len(Pairings_Indexes))
            
            #if images have been disabled (consumed by another list)
            if Pairings_Indexes[BaseImgList][1].InUse==False:
                continue
            
            #have to test first list of images against second list of images
            Similarity_List_base=[]
            #get similarity between all images in base list 
            for outdex,SingleBaseImg in enumerate(Pairings_Indexes[BaseImgList][0]):
                for index,SingleTestImg in enumerate(Pairings_Indexes[BaseImgList][0]):
                    if index<outdex:
                        continue
                    #print(MatchImages.HM_data_MetricDistances[SingleBaseImg,SingleTestImg])
                    Similarity_List_base.append(round(MatchImages.HM_data_MetricDistances[SingleBaseImg,SingleTestImg],6))
            std_d=statistics.pstdev(Similarity_List_base)
            mean=statistics.mean(Similarity_List_base)
            MatchMetric_Std_PerList.append(std_d)
            MatchMetric_mean_PerList.append(mean)
            Pairings_Indexes[BaseImgList][1].StatsOfList.append((std_d,mean))

            
            TestImgLists_Similarities=dict()
            #roll through all list of images
            for InnerIndex,TestImgList in enumerate(Pairings_Indexes):
                if InnerIndex<OuterIndex:
                    continue
                #if images have been disabled (consumed by another list)
                if Pairings_Indexes[TestImgList][1].InUse==False:
                    continue
                #dont test yourself
                if BaseImgList==TestImgList:
                    continue

                #have to test first list of images against second list of images
                Similarity_List=[]
                #get similarity between all images in base list and all images in test list
                #as we start with one image per list this should work out
                for SingleBaseImg in Pairings_Indexes[BaseImgList][0]:
                    for SingleTestImg in Pairings_Indexes[TestImgList][0]:
                        #print(MatchImages.HM_data_MetricDistances[SingleBaseImg,SingleTestImg])
                        Similarity_List.append(round(MatchImages.HM_data_MetricDistances[SingleBaseImg,SingleTestImg],6))

                TestImgLists_Similarities[TestImgList]=Similarity_List

            #now get standard deviation and mean
            TestImgLists_Similarities_Stats=dict()
            DefaultFail=999999
            LowestMean=DefaultFail
            Lowest_meanID=None
            for ListSimilarities in TestImgLists_Similarities:
                #pstdevdev used for entire population which might be true in this case
                #otherwise use stddev
                std_d=statistics.pstdev(TestImgLists_Similarities[ListSimilarities])
                mean=statistics.mean(TestImgLists_Similarities[ListSimilarities])
                TestImgLists_Similarities_Stats[ListSimilarities]=(std_d,mean)
                if mean<LowestMean:
                    Lowest_meanID=ListSimilarities
                    LowestMean=mean

            #no matches left
            if LowestMean==DefaultFail:
                continue

            MatchMetric_all.append(LowestMean)
            #now choose the list with the lowest mean - so must in theory be closest match for the batch
            #might want to do some stats here to filter out matches beyond a certain std deviation
            BaseList_info=Pairings_Indexes[BaseImgList]
            TestList_info=Pairings_Indexes[Lowest_meanID]
            #modifiy dictionaries
            Pairings_Indexes[BaseImgList]=(BaseList_info[0]+TestList_info[0],Pairings_Indexes[BaseImgList][1])
            Pairings_Indexes[Lowest_meanID][1].InUse=False
    #except Exception as e:
      #  print(e)

                        

            #lets write out pairing
    for ListIndex, ListOfImages in enumerate(Pairings_Indexes):
        #make folder for each set of images
        if Pairings_Indexes[ListOfImages][1].InUse==True:
            LengthImages=len(Pairings_Indexes[ListOfImages][0])
            SetMatchImages_folder=MatchImages.OutputPairs +"\\" + str(ListIndex) + "len_" + str(LengthImages) +"\\"
            _3DVisLabLib. MakeFolder(SetMatchImages_folder)
            #save plot of std and mean in folder
            std_dev=[]
            Mean=[]
            for Stats in Pairings_Indexes[ListOfImages][1].StatsOfList:
                std_dev.append(Stats[0])
                Mean.append(Stats[1])
            #now have lists of std and mean, save images into the folder
            PlotAndSave("std_dev",SetMatchImages_folder +"\\std_dev.jpg",std_dev,1)
            PlotAndSave("Mean",SetMatchImages_folder +"\\Mean.jpg",Mean,1)

            for imgIndex, Images in enumerate (Pairings_Indexes[ListOfImages][0]):
                #MatchDistance=str(MatchImages.ImagesInMem_Pairing[ListOfImages][1])
                FileName=ImgNameV_ID[Images]
                TempFfilename=SetMatchImages_folder  + "00" + str(ListIndex) + "_" +str(imgIndex)  + ".jpg"
                shutil.copyfile(FileName, TempFfilename)
                #cv2.imwrite(TempFfilename,MatchImages.ImagesInMem_to_Process[FileName].OriginalImage)


    PlotAndSave("MatchMetric_all",MatchImages.OutputPairs +"\\MatchMetric_all.jpg",MatchMetric_all,1)
    PlotAndSave("MatchMetric_Std_PerList",MatchImages.OutputPairs +"\\MatchMetric_Std_PerList.jpg",MatchMetric_Std_PerList,1)
    PlotAndSave("MatchMetric_mean_PerList",MatchImages.OutputPairs +"\\MatchMetric_mean_PerList.jpg",MatchMetric_mean_PerList,1)

    
    #MatchImages.Endtime= time.time()
    #print("time taken (hrs):",round((MatchImages.Endtime- MatchImages.startTime)/60/60,2))
    #print("time taken (mins):",round((MatchImages.Endtime- MatchImages.startTime)/60,2))
    exit()

def PrintResults(MatchImages,PlotAndSave_2datas,PlotAndSave):


    for MatchMetric in MatchImages.Metrics_dict:
        FilePath=MatchImages.OutputPairs +"\\" + str(MatchMetric) + "auto" +".jpg"
        PlotAndSave_2datas(str(MatchMetric) + "_auto",FilePath,MatchImages.Metrics_dict[MatchMetric])
    FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +("HM_data_MetricDistances_auto") +".jpg"
    PlotAndSave_2datas("HM_data_MetricDistances_auto",FilePath,MatchImages.HM_data_MetricDistances)

def MatchImagestoInputImages(MatchImages,PlotAndSave_2datas,PlotAndSave):
    #debug final data
    MatchImages.HM_data_All=MatchImages.HM_data_MetricDistances
     #for every image or subsets of images, roll through heatmap finding nearest best match then
    #cross referencing it
    OrderedImages=dict()
    #blank out the self test
    BlankOut=MatchImages.HM_data_All.max()*2.00000#should be "2" if normalised
    #for item in MatchImages.ImagesInMem_Pairing:
    #    MatchImages.HM_data_All[item,item]=0
    BaseImageList=0#random.choice(list(MatchImages.ImagesInMem_Pairing.keys()))
    Counter=0


    
    #images to match should be ordered at start of metrics, so only need to do these
    for IndexImg,Image in enumerate(MatchImages.List_ImagesToMatchFIlenames):
        counter=0
        MaxMatches=30
        MatchCounter=0
        #create output folder
        SetMatchImages_folder=MatchImages.OutputPairs +"\\" + str(IndexImg) + "_" + str(Image.split(".")[-2]) + "\\"
        _3DVisLabLib. MakeFolder(SetMatchImages_folder)
        #save test image first 
        FilePath=SetMatchImages_folder + "_00" + str(counter) + "_"+ Image.replace(" ","_")
        ImagePath=MatchImages.List_ImagesToMatchFIlenames[Image]
        shutil.copyfile(ImagePath, FilePath)

        #get row of image with similarity of all other images
        Row=MatchImages.HM_data_All[IndexImg,0:len(MatchImages.ImagesInMem_Pairing)]





        # generate sequential match metric graphs for each metric stored in dictionary
        # dictionary of lists
        MatchMetricGraphDict = dict()
        for MatchMetric in MatchImages.Metrics_dict:
            MatchMetricGraphDict[MatchMetric] = []
        #manually add overall score
        MatchMetricGraphDict["MatchMetric_all"]=[]
        MatchMetric_all = []

        # get best match for each metric used
        for MatchMetric in MatchImages.Metrics_dict:
            if MatchMetric=="MatchMetric_all": continue #not a good idea putting this in with the other metrics

            TestRow = MatchImages.Metrics_dict[MatchMetric][IndexImg,0:len(MatchImages.ImagesInMem_Pairing)]

            # make sure no input images are used for analysis - unmem this to check all metrics are working correctly
            # (same image should be found for each metric as will have best match)
            for RowIndex, TestImage in enumerate(TestRow):
                TestImagePath = MatchImages.ImagesInMem_Pairing[RowIndex][0][0]
                # if an image from folder of images to match to main folder - blank out value (so no match)
                if TestImagePath in MatchImages.List_ImagesToMatchFIlenames.values():
                    TestRow[RowIndex] = BlankOut

            # get minimum value
            Testresult = np.where(TestRow == np.amin(TestRow))
            TestElement = random.choice(Testresult[0])  # incase we have two identical results
            # record MatchMetric
            TestMatchMetric_figure = round(TestRow[TestElement], 6)
            FilePath=SetMatchImages_folder + "____BEST_" + str(MatchMetric) + ".jpg"
            ImagePath = MatchImages.ImagesInMem_Pairing[TestElement][0][0]
            shutil.copyfile(ImagePath, FilePath)


        #roll through all results in row, get each min and blank it out for next iteration
        for RowIndex in range (0,len(Row)):
            counter=counter+1
            #get minimum value
            result = np.where(Row == np.amin(Row))
            #print("result",Row)
            Element=random.choice(result[0])#incase we have two identical results
            #record MatchMetric
            MatchMetric_figure=round(MatchImages.HM_data_All[IndexImg,Element],4)


            #blank out similarity element
            MatchImages.HM_data_All[IndexImg,Element]=BlankOut
            #MatchImages.HM_data_All[IndexImg,Element]=BlankOut

            #if perfect match probably a duplicate - skip
            if MatchMetric_figure==0.0: continue

            
            #save out image

            ImagePath=MatchImages.ImagesInMem_Pairing[Element][0][0]
            #save top matches and worst matches
            if MatchCounter<=MaxMatches:
                #beyond first image we dont want other items in the match image folder to be used
                if not ImagePath in MatchImages.List_ImagesToMatchFIlenames.values():
                    FilePath = SetMatchImages_folder + "BestMatch_00" + str(MatchCounter) + " MatchMetric_" + str(MatchMetric_figure) + "  .jpg"
                    shutil.copyfile(ImagePath, FilePath)
                    MatchCounter = MatchCounter + 1
                    # populate dynamic match metrics
                    #dont update if we are skipping images or will confuse debugging
                    for MatchMetric in MatchMetricGraphDict:
                        if MatchMetric == "MatchMetric_all":
                            # special case for aggregate
                            MatchMetricGraphDict["MatchMetric_all"].append(MatchMetric_figure)
                        else:
                            MatchMetricGraphDict[MatchMetric].append(
                                MatchImages.Metrics_dict[MatchMetric][ IndexImg,Element])
            else:
                #dont need to process the rest if we have our matches
                break

        # save out dynamic match metrics
        for MatchMetric in MatchMetricGraphDict:
            PlotAndSave(MatchMetric, SetMatchImages_folder + str(MatchMetric) + "_AUTO.jpg", MatchMetricGraphDict[MatchMetric], 1)

def SequentialMatchingPerImage(MatchImages,PlotAndSave_2datas,PlotAndSave):
    
    #debug final data
    MatchImages.HM_data_All=MatchImages.HM_data_MetricDistances

    #save metrics for exporting to json
    ExportMetrics_dict=dict()
    #for every image or subsets of images, roll through heatmap finding nearest best match then
    #cross referencing it
    OrderedImages=dict()
    #BaseImageList=random.choice(list(MatchImages.ImagesInMem_Pairing.keys()))

    #get minimum 
    #result = np.where(HM_data_All == np.amin(HM_data_All))
    #Element=random.choice(result[0])#incase we have two identical results


    #blank out the self test
    BlankOut=MatchImages.HM_data_All.max()*2.00000#should be "2" if normalised
    for item in MatchImages.ImagesInMem_Pairing:
        MatchImages.HM_data_All[item,item]=BlankOut

    #print(HM_data_All)
    #print("-----")
    BaseImageList=0#random.choice(list(MatchImages.ImagesInMem_Pairing.keys()))
    Counter=0

    #generate sequential match metric graphs for each metric stored in dictionary
    #dictionary of lists
    MatchMetricGraphDict=dict()
    for MatchMetric in MatchImages.Metrics_dict:
        MatchMetricGraphDict[MatchMetric]=[]

    MatchMetric_all=[]

    while len(OrderedImages)+1<len(MatchImages.ImagesInMem_Pairing):#+1 is a fudge or it crashes out with duplicate image bug - cant figure this out 
        
        #print(len(OrderedImages),"/",len(MatchImages.ImagesInMem_Pairing))
        #FilePath=MatchImages.OutputPairs +"\\00" + str(Counter) +  str(OutOfUse) +("HM_data_All") +".jpg"
        #PlotAndSave_2datas("HM_data_All",FilePath,normalize_2d(HM_data_All))
        
        #print("looking at row",BaseImageList,"for match for for")
        #HM_data_All[BaseImageList,BaseImageList]=BlankOut
        Row=MatchImages.HM_data_All[0:len(MatchImages.ImagesInMem_Pairing),BaseImageList]
        #print(Row)
        #get minimum value
        result = np.where(Row == np.amin(Row))
        #print("REsult",Row)
        Element=random.choice(result[0])#incase we have two identical results
        
        #populate dynamic match metrics
        for MatchMetric in MatchMetricGraphDict:
            MatchMetricGraphDict[MatchMetric].append(MatchImages.Metrics_dict[MatchMetric][Element,BaseImageList])
        #populate overall metric
        MatchMetric_all.append(MatchImages.HM_data_All[Element,BaseImageList])
        #add to output images
        
        for imgIndex, Images in enumerate (MatchImages.ImagesInMem_Pairing[Element][0]):
            Counter=Counter+1
            #if len(Images)>1:y
                #raise Exception("too many images")
            SplitImagePath=Images.split("\\")[-1]
            FilePath=MatchImages.OutputPairs +"\\00" +str(Counter)+ "_ImgNo_" + str(BaseImageList) + "_score_" + str(round(MatchImages.HM_data_All[Element,BaseImageList],3))+ "_" + SplitImagePath
            ImagePath=MatchImages.ImagesInMem_to_Process[Images].OriginalImageFilePath

            #add to dictionary to export details for subsequent operations such as find unique sets of images
            ExportMetrics_dict[Counter]={"SequencedImgFilePath":FilePath,"OriginalImgFlePath":ImagePath,"MatchScore":MatchImages.HM_data_All[Element,BaseImageList]}

            shutil.copyfile(ImagePath, FilePath)
            #cv2.imwrite(FilePath,MatchImages.ImagesInMem_to_Process[Images].OriginalImage)
            if Images in OrderedImages:
                raise Exception("output images file already exists!!! logic error " + FilePath)
            else:
                OrderedImages[Images]=BaseImageList
        #now print out histogram with skew?
        #FilePath=MatchImages.OutputPairs +"\\00" +str(Counter)+ "_ImgNo_" + str(BaseImageList) + "_" + str(round(HM_data_All[Element,BaseImageList],3))+ "_HISTO_" + SplitImagePath
        #PlotAndSaveHistogram("self similar histogram",FilePath,HM_data_All_Copy[0:len(MatchImages.ImagesInMem_Pairing),BaseImageList],0,30)


        #blank out element in All places
        MatchImages.HM_data_All[0:len(MatchImages.ImagesInMem_Pairing),BaseImageList]=BlankOut
        MatchImages.HM_data_All[BaseImageList,0:len(MatchImages.ImagesInMem_Pairing)]=BlankOut
        #work in columns to find nearest match, data should be mirrored diagonally to make it easier to visualise#
        
        #move to next element
        BaseImageList=Element

    #create accumulation mean/std deviation -maybe can see where drop off is
    MatchMetric_Filter_mean=[]
    MatchMetric_Filter_std=[]
    FilterSize=2
    Buffer = [0] * FilterSize
    Buffered_Metric=Buffer + MatchMetric_all + Buffer
    for OuterIndexer, Metric in enumerate(Buffered_Metric):
        #break out before hitting end
        if OuterIndexer==len(MatchMetric_all)-FilterSize:break
        #get subset
        MatchMetric_all_subset=Buffered_Metric[OuterIndexer:OuterIndexer+FilterSize]
        MatchMetric_Filter_std.append(statistics.pstdev(MatchMetric_all_subset))
        MatchMetric_Filter_mean.append(statistics.mean(MatchMetric_all_subset))

    MatchMetricProdFilter = [a * b for a, b in zip(MatchMetric_Filter_std, MatchMetric_Filter_mean)]
    PlotAndSave("MatchMetricProdFilter",MatchImages.OutputPairs +"\\MatchMetricProdFilter.jpg",MatchMetricProdFilter,1)

    
    #save out dynamic match metrics
    for MatchMetric in MatchMetricGraphDict:
        PlotAndSave(MatchMetric,MatchImages.OutputPairs +"\\" + str(MatchMetric) +"_AUTO.jpg",MatchMetricGraphDict[MatchMetric],1)
    #summed metric saved out seperately
    PlotAndSave("MatchMetric_all",MatchImages.OutputPairs +"\\MatchMetric_all.jpg",MatchMetric_all,1)

    #save out image filenames and match score to use later
    ExportMetrics_filename=MatchImages.OutputPairs +"\\ExportedMatchData.md"
    _3DVisLabLib.JSON_Save(ExportMetrics_filename,ExportMetrics_dict)
    print("saved metrics to ",ExportMetrics_filename)

    #MatchImages.Endtime= time.time()
    #print("time taken (hrs):",round((MatchImages.Endtime- MatchImages.startTime)/60/60,2))
    #print("time taken (mins):",round((MatchImages.Endtime- MatchImages.startTime)/60,2))
    exit()

def PowerSpectralDensity(image):
    #https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
    npix = image.shape[0]

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    pl.loglog(kvals, Abins)
    pl.xlabel("$k$")
    pl.ylabel("$P(k)$")
    pl.tight_layout()
    pl.savefig("cloud_power_spectrum.png", dpi = 300, bbox_inches = "tight")

def CompareHistograms(histo_Img1,histo_Img2):
    def L2Norm(H1,H2):
        distance =0
        for i in range(len(H1)):
            distance += np.square(H1[i]-H2[i])
        return np.sqrt(distance)

    similarity_metric = L2Norm(histo_Img1,histo_Img2)
    return similarity_metric

def ProcessSimilarity(Input):
    #print("starting int loop",Input[1])

    MatchImages=Input[0]
    CurrentBaseImage=Input[1]


    if MatchImages.MatchInputSet==True:
        if CurrentBaseImage>(len(MatchImages.List_ImagesToMatchFIlenames)+1):
            return None#return empty dic

    BaseImage_Object=None
    TestImage_Object=None
    #get info for base image
    Base_Image_name=MatchImages.ImagesInMem_Pairing[CurrentBaseImage][1].FirstImage

    #if we have a prebuilt in image processsing function - use that to process images on the fly, otherwise use the preloaded analytics
    if MatchImages.ImagesInMem_to_Process[Base_Image_name].ProcessImages_function==None:
        #if test image is broken we cant do anything
        BaseImage_Object=MatchImages.ImagesInMem_to_Process[Base_Image_name].Metrics_functions
        
    else:
        #process images on the fly using function which is passed in from image info object (so can use different functions depending on application)
        #pass in the object info and it will be returned loaded with analytics for the image in lists of images
        BaseImage_Object=MatchImages.ImagesInMem_to_Process[Base_Image_name].ProcessImages_function(MatchImages.ImagesInMem_to_Process[Base_Image_name],True,MatchImages.Metrics_dict).Metrics_functions
        

        
    for CountIterations, TestImageList in enumerate(MatchImages.ImagesInMem_Pairing):
        if TestImageList<CurrentBaseImage:
            #data is diagonally symmetrical
            continue
        #print("doing",TestImageList,"of",CurrentBaseImage)
        PluralImages_BestIndex=[]
        #test images - this is where different strategies may come in
        #get first image, can also use the list for this
        #get info for test images
        if (CountIterations%1000)==0 and (CountIterations>0):
            #pass
            print("Position",TestImageList,"/",len(MatchImages.ImagesInMem_Pairing),"of item",CurrentBaseImage)

        Test_Image_name=MatchImages.ImagesInMem_Pairing[TestImageList][1].FirstImage

        if MatchImages.ImagesInMem_to_Process[Test_Image_name].ProcessImages_function==None:
            TestImage_Object=Test_Image_Histo=MatchImages.ImagesInMem_to_Process[Test_Image_name].Metrics_functions
        else:
            #process images on the fly
            TestImage_Object=MatchImages.ImagesInMem_to_Process[Test_Image_name].ProcessImages_function(MatchImages.ImagesInMem_to_Process[Test_Image_name],False,MatchImages.Metrics_dict).Metrics_functions
            
        BestMatch=MatchImages.DummyMaxValue

        try:
                
            if "HM_data_MacroStructure" in MatchImages.Metrics_dict:
                
                BestIndex=-1
                
                for Indexer,testimage in enumerate(TestImage_Object["HM_data_MacroStructure"]):
                    #very small image (3*3) used to check macro structure
                    diff = cv2.absdiff(testimage, BaseImage_Object["HM_data_MacroStructure"][0])
                    diff_sum=diff.sum()
                    # if len(testimage.shape)==3:
                    #     diff_sum=0
                    #     for Channel in range(2):
                    #         TestChannel=testimage[:,:,Channel]
                    #         BaseChannel=BaseImage_Object["HM_data_MacroStructure"][0][:,:,Channel]

                    #         diff=np.sum(TestChannel*BaseChannel)
                    #         diff_sum=diff_sum+(1-diff)

                    if (diff_sum<BestMatch) or Indexer==0:
                        BestMatch=diff_sum
                        BestIndex=Indexer
                        MatchImages.Metrics_dict["HM_data_MacroStructure"][CurrentBaseImage, TestImageList] = diff_sum
                PluralImages_BestIndex.append(BestIndex)

            if "HM_data_StructuralPCA_dotProd" in MatchImages.Metrics_dict:
            
                BestIndex=-1

                for Indexer,testimage in enumerate(TestImage_Object["HM_data_StructuralPCA_dotProd"]):
                    ListEigenDots = []
                    MaxRange = min(len(testimage), len(BaseImage_Object["HM_data_StructuralPCA_dotProd"][0]))
                    for EVector in range(MaxRange):
                        # if unit vector, same direction =1 , opposite = -1,perpendicular=0
                        RawDotProd = testimage[EVector] @ BaseImage_Object["HM_data_StructuralPCA_dotProd"][0][EVector]
                        if RawDotProd<-1.001 or RawDotProd>1.001:#rounding errors
                            print("ERROR HM_data_StructuralPCA_dotProd, dot product should be between -1 and 1",RawDotProd)
                        #move into positive numbers just in case
                        RawDotProd=RawDotProd+1
                        ListEigenDots.append(abs(2-RawDotProd))
                    
                    sum_ListEigenDots=sum(ListEigenDots)

                    if (sum_ListEigenDots<BestMatch) or Indexer==0:
                        BestMatch=sum_ListEigenDots
                        BestIndex=Indexer
                        EigenVectorDotProd_struct =sum_ListEigenDots   # round((Base_Image_EigenVectors[0] @ Test_Image_EigenVectors[0]),5)
                        MatchImages.Metrics_dict["HM_data_StructuralPCA_dotProd"][CurrentBaseImage, TestImageList] = EigenVectorDotProd_struct
                PluralImages_BestIndex.append(BestIndex)
            
            
            
            
            
            if "HM_data_StructuralPCA_VectorValue" in MatchImages.Metrics_dict:
                
                BestIndex=-1

                for Indexer,testimage in enumerate(TestImage_Object["HM_data_StructuralPCA_VectorValue"]):
                    Diff=0
                    MaxRange=min(len(testimage),len(BaseImage_Object["HM_data_StructuralPCA_VectorValue"][0]))
                    for eigenelem in range(MaxRange):
                        Diff=Diff+(testimage[eigenelem]-BaseImage_Object["HM_data_StructuralPCA_VectorValue"][0][eigenelem])**2

                    if (Diff<BestMatch) or Indexer==0:
                        BestMatch=Diff
                        BestIndex=Indexer
                        MatchImages.Metrics_dict["HM_data_StructuralPCA_VectorValue"][CurrentBaseImage,TestImageList]=Diff
                PluralImages_BestIndex.append(BestIndex)



            if "HM_data_EigenValueDifference" in MatchImages.Metrics_dict:
                
                BestIndex=-1

                #eigenvector metric
                #get dot product of top eigenvector (should be sorted for most significant set to [0])
                #if using static scene (like MM1 or a movie rather than freely translateable objects)
                #the eigenvector dot product will probably just add noise
                for Indexer,testimage in enumerate(TestImage_Object["HM_data_EigenValueDifference"]):
                    ListEigenVals=[]
                    #get range of eigen values/vectors to test - arbitrary amount of 5
                    MinVal_Eigens=min(len(BaseImage_Object["HM_data_EigenValueDifference"][0]),len(testimage))
                    ValRange=min(MinVal_Eigens,5)
                    for EVector in range (0,ValRange):
                        #need to look at this closer to see if we need to do anything to vectors before getting dot prod
                        ListEigenVals.append(abs((BaseImage_Object["HM_data_EigenValueDifference"][0][EVector]-testimage[EVector])))

                    EigenValue_diff=sum(ListEigenVals)#abs((Base_Image_EigenValues[0] )-(Test_Image_EigenValues[0] ))

                    if (EigenValue_diff<BestMatch) or Indexer==0:
                        BestMatch=EigenValue_diff
                        BestIndex=Indexer
                    
                        #get distance
                        #print(EigenValue_diff)
                        MatchImages.Metrics_dict["HM_data_EigenValueDifference"][CurrentBaseImage,TestImageList]=EigenValue_diff
                PluralImages_BestIndex.append(BestIndex)

            if "HM_data_EigenVectorDotProd" in MatchImages.Metrics_dict:
                
                BestIndex=-1

                for Indexer,testimage in enumerate(TestImage_Object["HM_data_EigenVectorDotProd"]):
                    ListEigenDots=[]
                    #get range of eigen values/vectors to test - arbitrary amount of 5
                    MinVal_Eigens=min(len(BaseImage_Object["HM_data_EigenVectorDotProd"][0]),len(testimage))
                    ValRange=min(MinVal_Eigens,5)
                    for EVector in range (0,ValRange):
                        #need to look at this closer to see if we need to do anything to vectors before getting dot prod
                        #if unit vector, same direction =1 , opposite = -1,perpendicular=0
                        RawDotProd=BaseImage_Object["HM_data_EigenVectorDotProd"][0][EVector] @ testimage[EVector]
                        if RawDotProd<-1.001 or RawDotProd>1.001:#rounding errors
                            print("ERROR, dot product should be between -1 and 1",RawDotProd)
                        #move into positive numbers just in case
                        RawDotProd=RawDotProd+1
                        SCore=abs(2-RawDotProd)
                        ListEigenDots.append(round(SCore,8))
                        break#just do first for now till we figure out what is going on 

                    EigenVectorDotProd=sum(ListEigenDots)#round((Base_Image_EigenVectors[0] @ Test_Image_EigenVectors[0]),5)
                    
                    if (EigenVectorDotProd<BestMatch) or Indexer==0:
                        BestMatch=EigenVectorDotProd
                        BestIndex=Indexer

                        MatchImages.Metrics_dict["HM_data_EigenVectorDotProd"][CurrentBaseImage,TestImageList]=EigenVectorDotProd
                #StackTwoimages=MatchImages.StackTwoimages(Base_Image_FM,Test_Image_FM)
                #_3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(StackTwoimages,(StackTwoimages.shape[1]*1,StackTwoimages.shape[0]*1)),0,True,True)
                
                PluralImages_BestIndex.append(BestIndex)
            #histogram metric

            if "HM_data_HistogramStriping" in MatchImages.Metrics_dict:
                BestIndex=-1
                for Indexer,testimage in enumerate(TestImage_Object["HM_data_HistogramStriping"]):
                    
                    HistoSum=0
                    for HistoIndex, HistoGramStrip in enumerate(testimage):
                        HistogramSimilarity=CompareHistograms(BaseImage_Object["HM_data_HistogramStriping"][0][HistoIndex],HistoGramStrip)
                        HistoSum=HistoSum+HistogramSimilarity

                    if (BestMatch>HistoSum) or Indexer==0:
                        BestMatch=HistoSum
                        BestIndex=Indexer
                MatchImages.Metrics_dict["HM_data_HistogramStriping"][CurrentBaseImage,TestImageList]=BestMatch
                PluralImages_BestIndex.append(BestIndex)

            if "HM_data_histo" in MatchImages.Metrics_dict:
                
                BestIndex=-1

                for Indexer,testimage in enumerate(TestImage_Object["HM_data_histo"]):
                    HistogramSimilarity=CompareHistograms(BaseImage_Object["HM_data_histo"][0],testimage)

                    if (BestMatch>HistogramSimilarity) or Indexer==0:
                        BestMatch=HistogramSimilarity
                        BestIndex=Indexer
                MatchImages.Metrics_dict["HM_data_histo"][CurrentBaseImage,TestImageList]=BestMatch
                PluralImages_BestIndex.append(BestIndex)
                
            #CheckImages_InfoSheet.AllHisto_results.append(HistogramSimilarity)

            #feature match metric
            if "HM_data_FM" in MatchImages.Metrics_dict:
                
                BestIndex=-1
                BaseKeypoints=BaseImage_Object["HM_data_FM"][0][0]
                BaseDescriptors=BaseImage_Object["HM_data_FM"][0][1]
                for Indexer,testimage_FMMatchers in enumerate(TestImage_Object["HM_data_FM"]):
                    TestKeypoints=testimage_FMMatchers[0]
                    TestDescriptors=testimage_FMMatchers[1]
                    try:
                        MatchedPoints,OutputImage,PointsA,PointsB,FinalMatchMetric=_3DVisLabLib.Orb_FeatureMatch(BaseKeypoints,BaseDescriptors,TestKeypoints,TestDescriptors,99999,None,None,0.7,MatchImages.DummyMinValue)
                        AverageMatchDistance=FinalMatchMetric#smaller the better
                        #print("Feature match",FinalMatchMetric,len(Base_Image_FMatches),len(Test_Image_FMatches))
                    except:
                        print("ERROR with feature match",len(BaseKeypoints),len(TestKeypoints))
                        #watch out this might not be a valid maximum!!
                        AverageMatchDistance=MatchImages.DummyMinValue

                    if (AverageMatchDistance<BestMatch) or Indexer==0:
                        BestMatch=AverageMatchDistance
                        BestIndex=Indexer
                        MatchImages.Metrics_dict["HM_data_FM"][CurrentBaseImage,TestImageList]=AverageMatchDistance

                PluralImages_BestIndex.append(BestIndex)

            if "HM_data_HOG_Dist" in MatchImages.Metrics_dict:
            
                BestIndex=-1

                for Indexer,testimage_FMMatchers in enumerate(TestImage_Object["HM_data_HOG_Dist"]):
                    HOG_distance=CompareHistograms(BaseImage_Object["HM_data_HOG_Dist"][0], testimage_FMMatchers)

                    if (HOG_distance<BestMatch) or Indexer==0:
                        BestMatch=HOG_distance
                        BestIndex=Indexer


                        MatchImages.Metrics_dict["HM_data_HOG_Dist"][CurrentBaseImage,TestImageList]=HOG_distance
                PluralImages_BestIndex.append(BestIndex)



            if "HM_data_FourierPowerDensity" in MatchImages.Metrics_dict:
                
                BestIndex=-1

                for Indexer,ToTestPwrSpectralDensity in enumerate(TestImage_Object["HM_data_FourierPowerDensity"]):
                    if len(ToTestPwrSpectralDensity.shape)!=1:
                        raise Exception("Similarity test error, HM_data_FourierPowerDensity, expected input should be 1D histogram")
                    #HM_data_FourierPowerDensity=random.random()
                    #HM_data_FourierPowerDensity=np.correlate(Base_PwrSpectralDensity,Test_PwrSpectralDensity,mode='full')[0]
                    #print(np.correlate(Base_PwrSpectralDensity,Test_PwrSpectralDensity,mode='full')[0:10])
                    HM_data_PowerDensity=CompareHistograms(BaseImage_Object["HM_data_FourierPowerDensity"][0],ToTestPwrSpectralDensity)

                    if (HM_data_PowerDensity<BestMatch) or Indexer==0:
                        BestMatch=HM_data_PowerDensity
                        BestIndex=Indexer


                        MatchImages.Metrics_dict["HM_data_FourierPowerDensity"][CurrentBaseImage,TestImageList]=HM_data_PowerDensity
                PluralImages_BestIndex.append(BestIndex)
            #fourier difference metric
            #get differnce between fourier magnitudes of image
            #not the best solution as fourier magnitude will rotate with image 
            #generally this performs well on its own as matches similar notes with similar skew
            if "HM_data_FourierDifference" in MatchImages.Metrics_dict:
                
                BestIndex=-1

                for Indexer,testimage_FMMatchers in enumerate(TestImage_Object["HM_data_FourierDifference"]):
                    FourierDifference=(abs(BaseImage_Object["HM_data_FourierDifference"][0]-testimage_FMMatchers)).sum()


                    if (FourierDifference<BestMatch) or Indexer==0:
                        BestMatch=FourierDifference
                        BestIndex=Indexer

                        MatchImages.Metrics_dict["HM_data_FourierDifference"][CurrentBaseImage,TestImageList]=FourierDifference
                
                PluralImages_BestIndex.append(BestIndex)



            if "HM_data_PhaseCorrelation" in MatchImages.Metrics_dict:
            
                BestIndex=-1
                #phase correlation difference
                #use a polar wrapped version of the fourier transform magnitude
                #this is probably a silly way to do this
                #x and y are translation
                for Indexer,testimage_FMMatchers in enumerate(TestImage_Object["HM_data_PhaseCorrelation"]):

                    (sx, sy), PhaseCorrelationMatch_raw = cv2.phaseCorrelate(BaseImage_Object["HM_data_PhaseCorrelation"][0], testimage_FMMatchers)
                    if PhaseCorrelationMatch_raw>1.9:#account for rounding errors - very big account! this shouldnt be possible according to literature
                        print("ERROR, PhaseCorrelationMatch is greater than 1, this shoudnt be possible",PhaseCorrelationMatch_raw )
                    PhaseCorrelationMatch=1-PhaseCorrelationMatch_raw#signal power so we will reverse it 
                    
                    if (PhaseCorrelationMatch<BestMatch) or Indexer==0:
                        BestMatch=PhaseCorrelationMatch
                        BestIndex=Indexer
                    
                        MatchImages.Metrics_dict["HM_data_PhaseCorrelation"][CurrentBaseImage,TestImageList]=PhaseCorrelationMatch
                    #if np.isnan(PhaseCorrelationMatch):
                    #    PhaseCorrelationMatch=MatchImages.DummyMinValue
                PluralImages_BestIndex.append(BestIndex)


            #if using multiple images and same index for all metrics - probability of a good match
            if "HM_data_QuiltScan" in MatchImages.Metrics_dict:
                PluralImages_BestIndex_std_d=statistics.pstdev(PluralImages_BestIndex)
                PluralImages_BestIndex_mean=statistics.mean(PluralImages_BestIndex)
                #probably want to convert to binary encoding here and get Hamming Distance



            if "HM_data_HistogramCentralis"  in MatchImages.Metrics_dict:
                BestIndex=-1

                for Indexer,testimage in enumerate(TestImage_Object["HM_data_HistogramCentralis"]):
                    HistogramSimilarity=CompareHistograms(BaseImage_Object["HM_data_HistogramCentralis"][0],testimage)

                    if (BestMatch>HistogramSimilarity) or Indexer==0:
                        BestMatch=HistogramSimilarity
                        BestIndex=Indexer
                MatchImages.Metrics_dict["HM_data_HistogramCentralis"][CurrentBaseImage,TestImageList]=BestMatch
                PluralImages_BestIndex.append(BestIndex)

            if "HM_data_TemplateMatching" in MatchImages.Metrics_dict:
                BestIndex=-1
                for Indexer,testimage_templates in enumerate(TestImage_Object["HM_data_TemplateMatching"]):
                    img_template_probability_match = cv2.matchTemplate(BaseImage_Object["HM_data_TemplateMatching"][0], testimage_templates, cv2.TM_CCOEFF_NORMED)[0][0]
                    img_template_diff = 1 - img_template_probability_match

                    if (img_template_diff<BestMatch) or Indexer==0:
                        BestMatch=img_template_diff
                        BestIndex=Indexer
                        MatchImages.Metrics_dict["HM_data_TemplateMatching"][CurrentBaseImage,TestImageList]=img_template_diff
                PluralImages_BestIndex.append(BestIndex)

        except Exception as e:
            print("Error with ",Test_Image_name,"vs",Base_Image_name,"in process similarity")
            print(repr(e))
            #must fill item with dummy value
            for Metric in MatchImages.Metrics_dict:
                MatchImages.Metrics_dict[Metric][CurrentBaseImage,TestImageList]=MatchImages.DummyMinValue




        #StackTwoimages=MatchImages.StackTwoimages(Base_Image_FM,Test_Image_FM)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(StackTwoimages,(StackTwoimages.shape[1]*1,StackTwoimages.shape[0]*1)),0,True,True)
        #populate output metric comparison matrices

        #make data symmetrical for visualisation
        #for MatchMetric in MatchImages.Metrics_dict:
        #    MatchImages.Metrics_dict[MatchMetric][TestImageList,CurrentBaseImage]=MatchImages.Metrics_dict[MatchMetric][CurrentBaseImage,TestImageList]





    #build up return object
    ReturnList=dict()
    ReturnList["BASEIMAGE"]=CurrentBaseImage
    for MatchMetric in MatchImages.Metrics_dict:
        ReturnList[MatchMetric]=MatchImages.Metrics_dict[MatchMetric][CurrentBaseImage,:]
    return ReturnList