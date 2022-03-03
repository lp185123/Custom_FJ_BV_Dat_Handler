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

def PrepareImageMetrics_Faces(PrepareMatchImages,ImagePath,Index,ImageReviewDict,HOG_extrator):
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

    GrayScale_Resized=PrepareMatchImages.USERFunction_Crop_Pixels(GrayScale_Resized,(90,178),(53,130))#Y range then X range
    Colour_Resized=PrepareMatchImages.USERFunction_Crop_Pixels(Colour_Resized,(90,178),(53,130))#Y range then X range
    if Index<3: ImageReviewDict["Colour_Resized"]=Colour_Resized

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
    For_HOG_FeatureMatch=PrepareMatchImages.USERFunction_PrepareForHOG(Colour_Resized.copy())
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
    Image_For_PCA=PrepareMatchImages.USERFunction_ResizePercent(Colour_Resized,90)
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
    FFT_magnitude_spectrum_visualise=cv2.convertScaleAbs(FFT_magnitude_spectrum)
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


    PhaseCorrelate_Std=PrepareMatchImages.USERFunction_ResizePercent(GrayScale_Resized,40)
    #if we are using an image we hvae to convert to float
    PhaseCorrelate_Std = PhaseCorrelate_Std.astype("float32")
    #PhaseCorrelate_Std=GetPhaseCorrelationReadyImage(PhaseCorrelate_Std)
    if Index<3: ImageReviewDict["PhaseCorrelate_Std visualise"]=cv2.convertScaleAbs(PhaseCorrelate_Std)


    DebugImage=PrepareMatchImages.StackTwoimages(Colour_Resized,FFT_magnitude_spectrum_visualise)

    if Index==1:
        #on first loop show image to user
        FM_DrawnKeypoints=_3DVisLabLib.draw_keypoints_v2(StackedColour_AndGradient_img.copy(),keypoints)
        ImageReviewDict["FM_DrawnKeypoints"]=FM_DrawnKeypoints


        for imagereviewimg in ImageReviewDict:
            Img=ImageReviewDict[imagereviewimg]
            print(imagereviewimg)
            _3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(Img,(Img.shape[1]*1,Img.shape[0]*1)),0,True,True)
        

    #get fourier transform
    #dft = cv2.dft(np.float32(OriginalImage_GrayScale),flags = cv2.DFT_COMPLEX_OUTPUT)
    #dft_shift = np.fft.fftshift(dft)
    #magnitude = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
    #product = 20*np.log(magnitude)
    


    #magnitude_spectrum_normed=PrepareMatchImages.ForceNormalise_forHOG(magnitude_spectrum)
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
    #WARNING this is easy to overpopulate and block out the RAM, then PC will do pagefile stuff on the harddrive and reduce performance
    ImageInfo.EigenValues=EigVal
    ImageInfo.EigenVectors=EigVec
    ImageInfo.PrincpleComponents=None#PC
    ImageInfo.Histogram=hist
    ImageInfo.OriginalImage=None#InputImage#OriginalImage_col
    ImageInfo.ImageGrayscale=None#OriginalImage_GrayScale
    ImageInfo.ImageColour=None#Colour_Resized
    ImageInfo.ImageAdjusted=None#Image_For_FM
    ImageInfo.FM_Keypoints=keypoints
    ImageInfo.FM_Descriptors=descriptor
    ImageInfo.FourierTransform_mag=FFT_magnitude_spectrum
    ImageInfo.HOG_Mag=None#HOG_mag
    ImageInfo.HOG_Angle=None#HOG_angle
    ImageInfo.OPENCV_hog_descriptor=OPENCV_hog_descriptor
    ImageInfo.PhaseCorrelate_FourierMagImg=PhaseCorrelate_Std
    ImageInfo.DebugImage=None#DebugImage
    ImageInfo.OriginalImageFilePath=ImagePath
    ImageInfo.PwrSpectralDensity=PwrSpectralDensity

    return ImageInfo

def PrepareImageMetrics_NotesSide(PrepareMatchImages,ImagePath,Index,ImageReviewDict,HOG_extrator):
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
    GrayScale_Resized=PrepareMatchImages.USERFunction_Crop_Pixels(GrayScale_Resized,(0,65),(0,144))#Y range then X range
    Colour_Resized=PrepareMatchImages.USERFunction_Crop_Pixels(Colour_Resized,(0,65),(0,144))#Y range then X range
    if Index<3: ImageReviewDict["Colour_Resized"]=Colour_Resized

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
    For_HOG_FeatureMatch=PrepareMatchImages.USERFunction_PrepareForHOG(Colour_Resized.copy())
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
    FFT_magnitude_spectrum_visualise=cv2.convertScaleAbs(FFT_magnitude_spectrum)
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


    PhaseCorrelate_Std=PrepareMatchImages.USERFunction_ResizePercent(GrayScale_Resized,40)
    #if we are using an image we hvae to convert to float
    PhaseCorrelate_Std = PhaseCorrelate_Std.astype("float32")
    #PhaseCorrelate_Std=GetPhaseCorrelationReadyImage(PhaseCorrelate_Std)
    if Index<3: ImageReviewDict["PhaseCorrelate_Std visualise"]=cv2.convertScaleAbs(PhaseCorrelate_Std)


    DebugImage=PrepareMatchImages.StackTwoimages(Colour_Resized,FFT_magnitude_spectrum_visualise)

    if Index==1:
        #on first loop show image to user
        FM_DrawnKeypoints=_3DVisLabLib.draw_keypoints_v2(StackedColour_AndGradient_img.copy(),keypoints)
        ImageReviewDict["FM_DrawnKeypoints"]=FM_DrawnKeypoints


        for imagereviewimg in ImageReviewDict:
            Img=ImageReviewDict[imagereviewimg]
            print(imagereviewimg)
            _3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(Img,(Img.shape[1]*1,Img.shape[0]*1)),0,True,True)
        

    #get fourier transform
    #dft = cv2.dft(np.float32(OriginalImage_GrayScale),flags = cv2.DFT_COMPLEX_OUTPUT)
    #dft_shift = np.fft.fftshift(dft)
    #magnitude = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
    #product = 20*np.log(magnitude)
    


    #magnitude_spectrum_normed=PrepareMatchImages.ForceNormalise_forHOG(magnitude_spectrum)
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
    #WARNING this is easy to overpopulate and block out the RAM, then PC will do pagefile stuff on the harddrive and reduce performance
    ImageInfo.EigenValues=EigVal
    ImageInfo.EigenVectors=EigVec
    ImageInfo.PrincpleComponents=None#PC
    ImageInfo.Histogram=hist
    ImageInfo.OriginalImage=None#InputImage#OriginalImage_col
    ImageInfo.ImageGrayscale=None#OriginalImage_GrayScale
    ImageInfo.ImageColour=None#Colour_Resized
    ImageInfo.ImageAdjusted=None#Image_For_FM
    ImageInfo.FM_Keypoints=keypoints
    ImageInfo.FM_Descriptors=descriptor
    ImageInfo.FourierTransform_mag=FFT_magnitude_spectrum
    ImageInfo.HOG_Mag=None#HOG_mag
    ImageInfo.HOG_Angle=None#HOG_angle
    ImageInfo.OPENCV_hog_descriptor=OPENCV_hog_descriptor
    ImageInfo.PhaseCorrelate_FourierMagImg=PhaseCorrelate_Std
    ImageInfo.DebugImage=None#DebugImage
    ImageInfo.OriginalImageFilePath=ImagePath
    ImageInfo.PwrSpectralDensity=PwrSpectralDensity

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


def GetPhaseCorrelationReadyImage(Image):
    #experiment with using rotational cross correlation type approach to the fourier magnitude
    #similiarity, must be an easier way to reinterpret fourier as 1D - like using fourier cofficients
    #https://stackoverflow.com/questions/57801071/get-rotational-shift-using-phase-correlation-and-log-polar-transform
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
    PlotAndSave_2datas("HM_data_MetricDistances_auto",FilePath,MatchImages.HM_data_MetricDistances_auto)



def MatchImagestoInputImages(MatchImages,PlotAndSave_2datas,PlotAndSave):
    #debug final data
    MatchImages.HM_data_All=MatchImages.HM_data_MetricDistances
     #for every image or subsets of images, roll through heatmap finding nearest best match then
    #cross referencing it
    OrderedImages=dict()
    #blank out the self test
    BlankOut=MatchImages.HM_data_All.max()*2.00000#should be "2" if normalised
    for item in MatchImages.ImagesInMem_Pairing:
        MatchImages.HM_data_All[item,item]=0
    BaseImageList=0#random.choice(list(MatchImages.ImagesInMem_Pairing.keys()))
    Counter=0

    #generate sequential match metric graphs for each metric stored in dictionary
    #dictionary of lists
    MatchMetricGraphDict=dict()
    for MatchMetric in MatchImages.Metrics_dict:
        MatchMetricGraphDict[MatchMetric]=[]

    MatchMetric_all=[]

    
    #images to match should be ordered at start of metrics, so only need to do these
    for IndexImg,Image in enumerate(MatchImages.List_ImagesToMatchFIlenames):
        counter=0
        #create output folder
        SetMatchImages_folder=MatchImages.OutputPairs +"\\" + str(IndexImg) + "_" + str(Image.split(".")[-2]) + "\\"
        _3DVisLabLib. MakeFolder(SetMatchImages_folder)
        #save test image first 
        FilePath=SetMatchImages_folder + "_00" + str(counter) + "_"+ Image
        ImagePath=MatchImages.List_ImagesToMatchFIlenames[Image]
        shutil.copyfile(ImagePath, FilePath)

        #get row of image with similarity of all other images
        Row=MatchImages.HM_data_All[0:len(MatchImages.ImagesInMem_Pairing),IndexImg]
        #roll through all results in row, get each min and blank it out for next iteration
        for RowIndex in range (0,len(Row)):
            counter=counter+1
            #get minimum value
            result = np.where(Row == np.amin(Row))
            #print("result",Row)
            Element=random.choice(result[0])#incase we have two identical results
            #record MatchMetric
            MatchMetric=round(MatchImages.HM_data_All[Element,IndexImg],3)
            #blank out similarity element
            MatchImages.HM_data_All[Element,IndexImg]=BlankOut
            MatchImages.HM_data_All[IndexImg,Element]=BlankOut
            #save out image
            FilePath=SetMatchImages_folder + "_00" + str(counter) +" MatchMetric_" + str(MatchMetric)+ "  .jpg"
            ImagePath=MatchImages.ImagesInMem_Pairing[Element][0][0]
            shutil.copyfile(ImagePath, FilePath)
            
            

def SequentialMatchingPerImage(MatchImages,PlotAndSave_2datas,PlotAndSave):
    
    #debug final data
    MatchImages.HM_data_All=MatchImages.HM_data_MetricDistances

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
        Counter=Counter+1
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
            #if len(Images)>1:y
                #raise Exception("too many images")
            SplitImagePath=Images.split("\\")[-1]
            FilePath=MatchImages.OutputPairs +"\\00" +str(Counter)+ "_ImgNo_" + str(BaseImageList) + "_score_" + str(round(MatchImages.HM_data_All[Element,BaseImageList],3))+ "_" + SplitImagePath
            ImagePath=MatchImages.ImagesInMem_to_Process[Images].OriginalImageFilePath
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

    #get info for base image
    Base_Image_name=MatchImages.ImagesInMem_Pairing[CurrentBaseImage][1].FirstImage
    Base_Image_Histo=MatchImages.ImagesInMem_to_Process[Base_Image_name].Histogram
    Base_Image_FMatches=MatchImages.ImagesInMem_to_Process[Base_Image_name].FM_Keypoints
    Base_Image_Descrips=MatchImages.ImagesInMem_to_Process[Base_Image_name].FM_Descriptors
    Base_Image_FourierMag=MatchImages.ImagesInMem_to_Process[Base_Image_name].FourierTransform_mag
    Base_Image_FM=MatchImages.ImagesInMem_to_Process[Base_Image_name].ImageAdjusted
    Base_Image_EigenVectors=MatchImages.ImagesInMem_to_Process[Base_Image_name].EigenVectors
    Base_Image_EigenValues=MatchImages.ImagesInMem_to_Process[Base_Image_name].EigenValues
    Base_Image_HOG_Descriptor=MatchImages.ImagesInMem_to_Process[Base_Image_name].OPENCV_hog_descriptor
    Base_Image_Phase_CorImg=MatchImages.ImagesInMem_to_Process[Base_Image_name].PhaseCorrelate_FourierMagImg
    Base_PwrSpectralDensity=MatchImages.ImagesInMem_to_Process[Base_Image_name].PwrSpectralDensity

    for TestImageList in MatchImages.ImagesInMem_Pairing:
        if TestImageList<CurrentBaseImage:
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
        Test_Image_Phase_CorImg=MatchImages.ImagesInMem_to_Process[Test_Image_name].PhaseCorrelate_FourierMagImg
        Test_PwrSpectralDensity=MatchImages.ImagesInMem_to_Process[Test_Image_name].PwrSpectralDensity


        if "HM_data_EigenValueDifference" in MatchImages.Metrics_dict:
            
            #eigenvector metric
            #get dot product of top eigenvector (should be sorted for most significant set to [0])
            #if using static scene (like MM1 or a movie rather than freely translateable objects)
            #the eigenvector dot product will probably just add noise
            ListEigenVals=[]
            #get range of eigen values/vectors to test - arbitrary amount of 5
            MinVal_Eigens=min(len(Base_Image_EigenVectors),len(Test_Image_EigenVectors))
            ValRange=min(MinVal_Eigens,5)
            for EVector in range (0,ValRange):
                #need to look at this closer to see if we need to do anything to vectors before getting dot prod
                ListEigenVals.append(abs((Base_Image_EigenValues[EVector]-Test_Image_EigenValues[EVector])))

            EigenValue_diff=sum(ListEigenVals)#abs((Base_Image_EigenValues[0] )-(Test_Image_EigenValues[0] ))
            #get distance
            #print(EigenValue_diff)
            MatchImages.Metrics_dict["HM_data_EigenValueDifference"][CurrentBaseImage,TestImageList]=EigenValue_diff


        if "HM_data_EigenVectorDotProd" in MatchImages.Metrics_dict:
            ListEigenDots=[]
            #get range of eigen values/vectors to test - arbitrary amount of 5
            MinVal_Eigens=min(len(Base_Image_EigenVectors),len(Test_Image_EigenVectors))
            ValRange=min(MinVal_Eigens,5)
            for EVector in range (0,ValRange):
                #need to look at this closer to see if we need to do anything to vectors before getting dot prod
                #if unit vector, same direction =1 , opposite = -1,perpendicular=0
                RawDotProd=Base_Image_EigenVectors[EVector] @ Test_Image_EigenVectors[EVector]
                if RawDotProd<-1.001 or RawDotProd>1.001:#rounding errors
                    print("ERROR, dot product should be between -1 and 1",RawDotProd)
                #move into positive numbers just in case
                RawDotProd=RawDotProd+1
                SCore=abs(2-RawDotProd)
                ListEigenDots.append(round(SCore,8))
                break#just do first for now till we figure out what is going on 

            EigenVectorDotProd=sum(ListEigenDots)#round((Base_Image_EigenVectors[0] @ Test_Image_EigenVectors[0]),5)
            
            MatchImages.Metrics_dict["HM_data_EigenVectorDotProd"][CurrentBaseImage,TestImageList]=EigenVectorDotProd
        #StackTwoimages=MatchImages.StackTwoimages(Base_Image_FM,Test_Image_FM)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(StackTwoimages,(StackTwoimages.shape[1]*1,StackTwoimages.shape[0]*1)),0,True,True)
        

        #histogram metric
        if "HM_data_histo" in MatchImages.Metrics_dict:
            HistogramSimilarity=CompareHistograms(Base_Image_Histo,Test_Image_Histo)
            MatchImages.Metrics_dict["HM_data_histo"][CurrentBaseImage,TestImageList]=HistogramSimilarity
        
        #CheckImages_InfoSheet.AllHisto_results.append(HistogramSimilarity)

        #feature match metric
        if "HM_data_FM" in MatchImages.Metrics_dict:
            try:
                MatchedPoints,OutputImage,PointsA,PointsB,FinalMatchMetric=_3DVisLabLib.Orb_FeatureMatch(Base_Image_FMatches,Base_Image_Descrips,Test_Image_FMatches,Test_Image_Descrips,99999,Base_Image_FM,Test_Image_FM,0.7,MatchImages.DummyMinValue)
                AverageMatchDistance=FinalMatchMetric#smaller the better
                #print("Feature match",FinalMatchMetric,len(Base_Image_FMatches),len(Test_Image_FMatches))
            except:
                print("ERROR with feature match",len(Base_Image_FMatches),len(Test_Image_FMatches))
                #watch out this might not be a valid maximum!!
                AverageMatchDistance=MatchImages.DummyMinValue
            MatchImages.Metrics_dict["HM_data_FM"][CurrentBaseImage,TestImageList]=AverageMatchDistance



        if "HM_data_HOG_Dist" in MatchImages.Metrics_dict:
            HOG_distance=CompareHistograms(Base_Image_HOG_Descriptor, Test_Image_HOG_Descriptor)
            MatchImages.Metrics_dict["HM_data_HOG_Dist"][CurrentBaseImage,TestImageList]=HOG_distance
        
        if "HM_data_FourierPowerDensity" in MatchImages.Metrics_dict:
            #HM_data_FourierPowerDensity=random.random()
            #HM_data_FourierPowerDensity=np.correlate(Base_PwrSpectralDensity,Test_PwrSpectralDensity,mode='full')[0]
            #print(np.correlate(Base_PwrSpectralDensity,Test_PwrSpectralDensity,mode='full')[0:10])
            HM_data_PowerDensity=CompareHistograms(Base_PwrSpectralDensity,Test_PwrSpectralDensity)
            MatchImages.Metrics_dict["HM_data_FourierPowerDensity"][CurrentBaseImage,TestImageList]=HM_data_PowerDensity
        
        #fourier difference metric
        #get differnce between fourier magnitudes of image
        #not the best solution as fourier magnitude will rotate with image 
        #generally this performs well on its own as matches similar notes with similar skew
        if "HM_data_FourierDifference" in MatchImages.Metrics_dict:
            FourierDifference=(abs(Base_Image_FourierMag-Test_Image_FourierMag)).sum()
            MatchImages.Metrics_dict["HM_data_FourierDifference"][CurrentBaseImage,TestImageList]=FourierDifference
        

        if "HM_data_PhaseCorrelation" in MatchImages.Metrics_dict:
            #phase correlation difference
            #use a polar wrapped version of the fourier transform magnitude
            #this is probably a silly way to do this
            #x and y are translation
            (sx, sy), PhaseCorrelationMatch_raw = cv2.phaseCorrelate(Base_Image_Phase_CorImg, Test_Image_Phase_CorImg)
            if PhaseCorrelationMatch_raw>1.9:#account for rounding errors - very big account! this shouldnt be possible according to literature
                print("ERROR, PhaseCorrelationMatch is greater than 1, this shoudnt be possible",PhaseCorrelationMatch_raw )
            PhaseCorrelationMatch=1-PhaseCorrelationMatch_raw#signal power so we will reverse it 
            MatchImages.Metrics_dict["HM_data_PhaseCorrelation"][CurrentBaseImage,TestImageList]=PhaseCorrelationMatch
            #if np.isnan(PhaseCorrelationMatch):
            #    PhaseCorrelationMatch=MatchImages.DummyMinValue
            


        #StackTwoimages=MatchImages.StackTwoimages(Base_Image_FM,Test_Image_FM)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(StackTwoimages,(StackTwoimages.shape[1]*1,StackTwoimages.shape[0]*1)),0,True,True)
        #populate output metric comparison matrices

        
        
        
        
        
        



        #make data symmetrical for visualisation
        for MatchMetric in MatchImages.Metrics_dict:
            MatchImages.Metrics_dict[MatchMetric][TestImageList,CurrentBaseImage]=MatchImages.Metrics_dict[MatchMetric][CurrentBaseImage,TestImageList]



    #build up return object
    ReturnList=dict()
    ReturnList["BASEIMAGE"]=CurrentBaseImage
    for MatchMetric in MatchImages.Metrics_dict:
        ReturnList[MatchMetric]=MatchImages.Metrics_dict[MatchMetric][CurrentBaseImage,:]


    return ReturnList