# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:50:19 2019

@author: Liell Plane
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import _3DVisLabLib
import random
import time
import copy

ImageCounter=0
#%matplotlib inline
SparseRecon_Report=["Sparse Reconstruct Report"]
SparseRecon_Report.append(("CV2 version: " + str(cv2.__version__)))
def display(text,img,cmap='gray'):
    fig = plt.figure(figsize=(15,10 ))
    ax = fig.add_subplot(111)
    plt.title(text)
    ax.imshow(img,cmap='gray')


    
ParamsObj=_3DVisLabLib.SparseReconstruct_InputArgs()


#set size on inline matlab plots
plt.rcParams['figure.figsize'] = [10, 8]

#sketchy way to conduct the parameter loop
#but we dont want to change MAIN here just yet with more input parameters
#TODO temporary method 

    
def SparseRecon(InputArguments):
    
    
    
  # try:
  
  
    MainLoop(InputArguments)
   #except Exception as e:
   #    print(e)
    #   return e
   
def MainLoop(InputArguments):
    global ParamsObj
    ParamsObj=copy.deepcopy(InputArguments)
    
    
    #create a new window
    cv2.namedWindow(winname=ParamsObj.ProcessName)
    x,y=_3DVisLabLib.GetPositionOfWindow_Number(ParamsObj.ProcessNumber)
    cv2.moveWindow(ParamsObj.ProcessName,x,y)
    
    
    
    #load previously calculated calibration details
    CalibrationData_reload=pickle.load(open(ParamsObj.CalibrationFile, 'rb'))
    
    #load in the camera pair from the saved stereocalibratefile
    Image1=ParamsObj.SubjectSession_CommonPrefix + CalibrationData_reload["str_SequenceCamA"] + '.jpg'
    Image2=ParamsObj.SubjectSession_CommonPrefix + CalibrationData_reload["str_SequenceCamB"] + '.jpg'
    
    SparseRecon_Report.append("looking for " + Image1)
    SparseRecon_Report.append("looking for " + Image2)
    SparseRecon_Report.append("Origin camera is : " + CalibrationData_reload["str_SequenceCamA"])
    SparseRecon_Report.append("Pair camera is : " + CalibrationData_reload["str_SequenceCamB"])
    
    
    _3DVisLabLib.ImageFileLock.acquire()
    Pod1Image = cv2.imread(Image1,0)
    Pod2Image = cv2.imread(Image2,0)
    Pod1Image_col = cv2.imread(Image1)
    Pod2Image_col = cv2.imread(Image2)
    _3DVisLabLib.ImageFileLock.release()
    
    #_3DVisLabLib.DisplayWinImage_Sparse("Starting Image", Pod2Image_col,ParamsObj)
    
    pts1=[]
    pts2=[]
    ImageLog=[]
    ImageTextLog=[]
    if InputArguments.FeatureTypeToUse==_3DVisLabLib.Str_Common.SIFT:
        _3DVisLabLib.DisplayWinImage_Sparse("Starting SIFT", Pod2Image_col,ParamsObj)
        pts1,pts2,ORB_Report,ImageLog,ImageTextLog=_3DVisLabLib.SIFT_Feature_and_Match(Pod1Image,Pod2Image,ParamsObj.MODIFIED_Param_DictionarySIFT,False)
    
    if InputArguments.FeatureTypeToUse==_3DVisLabLib.Str_Common.ORB:
        _3DVisLabLib.DisplayWinImage_Sparse("Starting ORB", Pod2Image_col,ParamsObj)
        pts1,pts2,ORB_Report,ImageLog,ImageTextLog=_3DVisLabLib.ORB_Feature_and_Match(Pod1Image,Pod2Image,ParamsObj.MODIFIED_Param_DictionaryORB,True)
    
    if pts1 is None:
        print("Nothing returned from feature matcher" )
        return()
    
    if len(pts1)<5:
        print("Warning: low points returned by feature matcher" )
        return()
        
    if len(ImageLog)==len(ImageTextLog):
        for indexer in range(0,len(ImageLog)):
            _3DVisLabLib.DisplayWinImage_Sparse(ImageTextLog[indexer],ImageLog[indexer],ParamsObj)
            
        
    SparseRecon_Report.append(ORB_Report)
    
    print("OK at this point" )
    _3DVisLabLib.DisplayWinImage_Sparse("Finished Feature Matching", Pod2Image_col,ParamsObj)
    
    #need to convert points to be compatible with the FindFundamentalMatrix function
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    #dont need Fundamental matrix yet
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC,3,0.1)#try RANSAC next time looks we like have a lot of noise
    RavelIt=False
    # Use only the points which RANSAC determined to be model inliers
    #ravel is a continous flattened array??
    if RavelIt==True:
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]
        SparseRecon_Report.append("Pts1 after MASK. RAVEL: " + str(len(pts1)))
    else:
        SparseRecon_Report.append("RAVEL disabled - might need for uncalibrated")
    
    
    
    
    #filter out bad matches
    CheckF_Errors=_3DVisLabLib.GetList_Of_F_Error_SingleArray(pts1,pts2,CalibrationData_reload["Fm"])
    Success, pts1,pts2,CleanUpLog= _3DVisLabLib.CleanUpXYMatches_With_Ferror(CheckF_Errors,CalibrationData_reload["Fm"],pts1,pts2,0.05)
    SparseRecon_Report.append(CleanUpLog)
    if Success==False:
        return
    
    #for logLine in SparseRecon_Report:
    #   print(logLine)
        

    
    def GetEpiConstraintFromPt(FundamentalMatix, MatchingPntA):
        #manually get the epipolar constraint for a point
        #opencv provides a function for this but I am not 100% sure what its giving
        #back (ie it may be sorting results) so for now lets guarantee we are getting valid results
        #by getting one point at a time
        pointXY=(MatchingPntA)
        #i think the XY matrixs have to be in form "X,Y,1".. dunno why though/
        PointXY_Matrix = np.ones(shape=(3))
        PointXY_Matrix[0]=pointXY[0]
        PointXY_Matrix[1]=pointXY[1]
        # convert to actual matrices maybe????
        #this step can be done using numpy matrix casting, or just dot products of basic arrays
        np_XYMatrix=np.matrix(PointXY_Matrix)
        np_FMatrix=np.matrix(FundamentalMatix)
        np_EpipolarLine=(np_XYMatrix*np_FMatrix)
        np_MakeSensible=np.empty(shape=[1, 3])
        
        return np_EpipolarLine#im not sure what form of line eq this is returning
    




    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    #each line encoded as ax+by+c=0 from [a,b,c]
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,CalibrationData_reload["Fm"])
    lines1_Reshaped = lines1.reshape(-1,3) 
    ImgA_EpiLines_OfImgB_pts,ImgB_Points = _3DVisLabLib.drawEpipolarlines(Pod1Image,Pod2Image,lines1_Reshaped,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,CalibrationData_reload["Fm"])
    lines2_Reshaped = lines2.reshape(-1,3)
    ImgB_EpiLines_OfImgA_pts,ImgA_Points = _3DVisLabLib.drawEpipolarlines(Pod2Image,Pod1Image,lines2_Reshaped,pts2,pts1)
    
    
    _3DVisLabLib.DisplayWinImage_Sparse("Epipolar linesImgA_EpiLines_OfImgB_pts",ImgA_EpiLines_OfImgB_pts,ParamsObj)
    #_3DVisLabLib.DisplayWinImage_Sparse("Epipolar linesImgB_Points",ImgB_Points,ParamsObj)
    _3DVisLabLib.DisplayWinImage_Sparse("Epipolar linesImgB_EpiLines_OfImgA_pts",ImgB_EpiLines_OfImgA_pts,ParamsObj)
    #_3DVisLabLib.DisplayWinImage_Sparse("Epipolar linesImgA_Points",ImgA_Points,ParamsObj)
    
    
    #random sample quality check - if we use F from CalculateFundamentalMatrix
    #instead of our calibrated results we get a  smaller error..
    #this isnt right!!!
    print("Quality Check for feature pair")
    RandomIndex=(round(random.random()*(len(pts1)-1)))
    
    
    ImgA_EpiLines_OfImgB_pts,Empty=_3DVisLabLib.drawEpipolarline_UsingIndex(Pod1Image,Pod2Image,lines1_Reshaped,pts1,pts2,RandomIndex)
    ImgB_EpiLines_OfImgA_pts,Empty=_3DVisLabLib.drawEpipolarline_UsingIndex(Pod2Image,Pod1Image,lines2_Reshaped,pts2,pts1,RandomIndex)
    
    
    #testline=GetEpiConstraintFromPt(CalibrationData_reload["Fm"],lines1[RandomIndex][-1,:])
    #print("testline" + str(testline) + "  end testline")
    
    #find corrresponding epipolar line on image X from feature on image Xprime
    #still dont understand why need to use -1 in the index!
    #have tried using unshaped index etc but still does not make sense
    CroppedA=_3DVisLabLib.DrawSingleLine(Pod1Image_col,lines1[RandomIndex][-1,:])#TODO i dont understand why i have to offset this
    CroppedB=_3DVisLabLib.DrawSingleLine(Pod2Image_col,lines2[RandomIndex][-1,:])
    
    #CroppedA=DrawSingleLine(Pod1Image_col,testline[-1,:])#TODO i d
    
    #this plots the feature pair onto input images to provide us with 
    #a manual method of checking veracity
    
    CroppedA,CroppedB, NP_Error=_3DVisLabLib.Fzero_Quality_Check_FeaturePair(pts1,pts2,RandomIndex,CalibrationData_reload["Fm"],CroppedA,CroppedB)
    CroppedA_epilines,CroppedB_epilines, NP_Error=_3DVisLabLib.Fzero_Quality_Check_FeaturePair(pts1,pts2,RandomIndex,CalibrationData_reload["Fm"],ImgA_EpiLines_OfImgB_pts,ImgB_EpiLines_OfImgA_pts)
        
        
    _3DVisLabLib.DisplayWinImage_Sparse("CUSTOM F_Error/EpiConstraint of feature pair: " + str(RandomIndex) +  " Err " + str(round(NP_Error,6)), CroppedA,ParamsObj)
    
    _3DVisLabLib.DisplayWinImage_Sparse("on opencv epiline plot feature pair: " + str(RandomIndex) +  " Err " + str(round(NP_Error,6)), CroppedA_epilines,ParamsObj)
    
    _3DVisLabLib.DisplayWinImage_Sparse("CUSTOM F_Error/EpiConstraint of feature pair: " + str(RandomIndex) + " Err " + str(round(NP_Error,6)), CroppedB,ParamsObj)
    
    _3DVisLabLib.DisplayWinImage_Sparse("on opencv epiline plot feature pair: " + str(RandomIndex) + " Err " + str(round(NP_Error,6)), CroppedB_epilines,ParamsObj)
    
    #ErrorMatrix,EpipolarConstraint=_3DVisLabLib.TestF_All_Image(Pod2Image_col,pts1[20],CalibrationData_reload["Fm"])
    #display("Test every pixel in image for F-error VS input feature XY", EpipolarConstraint)
    #cv2.imwrite('CORRECTDISTORTION/DrawEpiline.jpg',EpipolarConstraint)
   
    
    
    
    
    #draw all matches on an image so we can eyeball its validity with dumped pointcloud
    #we can get all F errors so can visualise error per match
    CheckF_Errors_Visualise=_3DVisLabLib.GetList_Of_F_Error_SingleArray(pts1,pts2,CalibrationData_reload["Fm"])
    print(CalibrationData_reload["Fm"])
    
    imgTestMatches=Pod1Image_col.copy()
    imgTestMatches2=Pod2Image_col.copy()
    #TODO really cheap way to let us draw colour markers on B&W version of image
    imgTestMatches=cv2.cvtColor(imgTestMatches,cv2.COLOR_BGR2GRAY)
    imgTestMatches2=cv2.cvtColor(imgTestMatches2,cv2.COLOR_BGR2GRAY)
    imgTestMatches=cv2.cvtColor(imgTestMatches,cv2.COLOR_GRAY2BGR)
    imgTestMatches2=cv2.cvtColor(imgTestMatches2,cv2.COLOR_GRAY2BGR)
    
    
    for i in range (len(pts1)):
        FRadius=8
        FThickness=5
        if _3DVisLabLib.GetColour_of_F_Error(CheckF_Errors_Visualise[i])==_3DVisLabLib.Str_Common.Red:
             FRadius=12
             FThickness=4
        point=(pts1[i][0],pts1[i][1])
        pointprime=(pts2[i][0],pts2[i][1])
        cv2.circle(img=imgTestMatches,center=point,radius=FRadius,color=_3DVisLabLib.GetColour_of_F_Error(CheckF_Errors_Visualise[i]),thickness=FThickness)
        cv2.circle(img=imgTestMatches2,center=pointprime,radius=FRadius,color=_3DVisLabLib.GetColour_of_F_Error(CheckF_Errors_Visualise[i]),thickness=FThickness)
    #now crop image
    #get corners of extrema so we can crop image 
    PointsA_MaxX=max(pts1[:,0])
    PointsA_MaxY=max(pts1[:,1])
    PointsA_MinX=min(pts1[:,0])
    PointsA_MinY=min(pts1[:,1])
    PointsB_MaxX=max(pts2[:,0])
    PointsB_MaxY=max(pts2[:,1])
    PointsB_MinX=min(pts2[:,0])
    PointsB_MinY=min(pts2[:,1])
    #create rectangle and crop image
    region_of_interestA = (PointsA_MinX, PointsA_MinY, PointsA_MaxX, PointsA_MaxY)
    imgTestMatches = imgTestMatches[region_of_interestA[1]:region_of_interestA[3], region_of_interestA[0]:region_of_interestA[2]]
    region_of_interestB = (PointsB_MinX, PointsB_MinY, PointsB_MaxX, PointsB_MaxY)
    imgTestMatches2 = imgTestMatches2[region_of_interestB[1]:region_of_interestB[3], region_of_interestB[0]:region_of_interestB[2]]
    
    
     
    _3DVisLabLib.DisplayWinImage_Sparse("Draw all " + str(len(pts1)) + " match pairs A", imgTestMatches,ParamsObj)
    _3DVisLabLib.DisplayWinImage_Sparse("Draw all " + str(len(pts2)) + " match pairs B", imgTestMatches2,ParamsObj)  
    #do this twice as i dont understand how the console draw function
    #seems to work yet - this gets wiped out when drawing the error
    #distribution later on
    #display("Draw all " + str(len(pts2)) + " match pairs B", imgTestMatches2)   
    _3DVisLabLib.DisplayWinImage_Sparse("Draw all " + str(len(pts2)) + " match pairs B", imgTestMatches2,ParamsObj)
    
    #turned on by test parameters loop
    if (ParamsObj.SaveString!=_3DVisLabLib.Str_Common.EMPTY) and len(ParamsObj.SaveString)>1:
        imgsave=cv2.resize(imgTestMatches,(800,1000))
        imgsave=cv2.cvtColor(imgsave,cv2.COLOR_BGR2RGB)
        cv2.imwrite(_3DVisLabLib.Paths_Common.ParamLoop_path + str(len(pts2)) + "_" + ParamsObj.SaveString + ".jpg",  imgsave)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #Sample average colour for each feature
    SampledFeatureColour=list()
    for i in range (len(pts1)):
        #sample area of image
        AverageCol, Cropped=_3DVisLabLib.SampleAreaOfImage(Pod1Image_col,pts1[i][0],pts1[i][1],AreaSpan=5)
        #elements R G B - might not be encoded in correct RGB - so convert earlier using cv2.cvtColor(name,cv2.COLOR_BGR2RGB)
        #swapped elements as incoming are in BGR form so swap to RGB
        SampledFeatureColour.append((AverageCol[2],AverageCol[1],AverageCol[0]))
        #display(Cropped)
        #print(AverageCol)
    






    
    #Pmatrix_manualA
       # SR_Project1
      #SR_Project1    
    #Pmatrix_manualA SR_Project1
    
    
    _3DVisLabLib.ImageFileLock.acquire()
    _3DVisLabLib.GenerateSubjectPntCloudsAndImgs_Calibs(_3DVisLabLib.FlipXYpnt_ArrayDims(pts1),
                                           _3DVisLabLib.FlipXYpnt_ArrayDims(pts2),
                                           CalibrationData_reload["Pmatrix_manualA"],
                                           CalibrationData_reload["Pmatrix_manualB"],
                                             None,
                                           CalibrationData_reload["str_SequenceCamA"],
                                           CalibrationData_reload["str_SequenceCamB"],
                                           "_MVS",
                                           _3DVisLabLib.Paths_Common.MVS_path,
                                          SampledFeatureColour,
                                          SeperatorCharactor=";")
    _3DVisLabLib.ImageFileLock.release()
    
    
    
    
    
    
    CheckF_Errors=_3DVisLabLib.GetList_Of_F_Error_SingleArray(pts1,pts2,CalibrationData_reload["Fm"])
    GoodErrorExampleData=_3DVisLabLib.GetStatsPlotHist_of_F_Errors(CheckF_Errors)
    #the library doesnt seem to like plotting the histogram
    #just do it here instead
    #the functions above return demo error distribution - so we
    #can snap-check validity
    HistoBins=256
    MinimumErr=min(min(CheckF_Errors),min(GoodErrorExampleData))
    MaximumErr=max(max(CheckF_Errors),max(GoodErrorExampleData))
    plt.clf()
    plt.hist(GoodErrorExampleData,HistoBins,[MinimumErr,MaximumErr])
    plt.hist(CheckF_Errors,HistoBins,[MinimumErr,MaximumErr])
    plt.title("Histogram of Ferrors for each feature pair, BLUE=target error distribution, RED = result distribution")
    plt.show()
    return

#main("CALIBS/pod2secondarypod1primaryCalib.hey","Calibs_Subjects/CalibrationPoses_Subject_002_")
                          
    
    

