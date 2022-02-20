import copy
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
    
    for Looper in range (0,4):
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
                ImagePath=MatchImages.ImagesInMem_to_Process[Images].OriginalImageFilePath
                shutil.copyfile(ImagePath, TempFfilename)
                #cv2.imwrite(TempFfilename,MatchImages.ImagesInMem_to_Process[FileName].OriginalImage)


    PlotAndSave("MatchMetric_all",MatchImages.OutputPairs +"\\MatchMetric_all.jpg",MatchMetric_all,1)
    PlotAndSave("MatchMetric_Std_PerList",MatchImages.OutputPairs +"\\MatchMetric_Std_PerList.jpg",MatchMetric_Std_PerList,1)
    PlotAndSave("MatchMetric_mean_PerList",MatchImages.OutputPairs +"\\MatchMetric_mean_PerList.jpg",MatchMetric_mean_PerList,1)

    
    MatchImages.Endtime= time.time()
    print("time taken (hrs):",round((MatchImages.Endtime- MatchImages.startTime)/60/60,2))
    print("time taken (mins):",round((MatchImages.Endtime- MatchImages.startTime)/60,2))
    exit()

def PrintResults(MatchImages,PlotAndSave_2datas,PlotAndSave):
    OutOfUse="WhatIsThis"
    #FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_All") +".jpg"
    #PlotAndSave_2datas("HM_data_All",FilePath,MatchImages.HM_data_All)
    FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_FM") +".jpg"
    PlotAndSave_2datas("HM_data_FM",FilePath,MatchImages.HM_data_FM)
    FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_histo") +".jpg"
    PlotAndSave_2datas("HM_data_histo",FilePath,MatchImages.HM_data_histo)
    FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_FourierDifference") +".jpg"
    PlotAndSave_2datas("HM_data_FourierDifference",FilePath,MatchImages.HM_data_FourierDifference)
    FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_EigenVectorDotProd") +".jpg"
    PlotAndSave_2datas("HM_data_EigenVectorDotProd",FilePath,MatchImages.HM_data_EigenVectorDotProd)
    FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_MetricDistances") +".jpg"
    PlotAndSave_2datas("HM_data_MetricDistances",FilePath,MatchImages.HM_data_MetricDistances)
    FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_HOG_Dist") +".jpg"
    PlotAndSave_2datas("HM_data_HOG_Dist",FilePath,MatchImages.HM_data_HOG_Dist)
    FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("HM_data_PhaseCorrelation") +".jpg"
    PlotAndSave_2datas("HM_data_PhaseCorrelation",FilePath,MatchImages.HM_data_PhaseCorrelation)

def SequentialMatchingPerImage(MatchImages,PlotAndSave_2datas,PlotAndSave):
    OutOfUse="OutOfUse"
    #HM_data_All=MatchImages.HM_data_FM
    HM_data_All_Copy=copy.deepcopy(MatchImages.HM_data_All)

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
    MatchMetric_all=[]
    MatchMetric_Histo=[]
    MatchMetric_Fourier=[]
    MatchMetric_FM=[]
    MatchMetric_EigenVectorDotProd=[]
    MatchMetric_HOG_Distance=[]
    MatchMetric_data_PhaseCorrelation=[]
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
        #print("nearest matching is element",Element)
        #print("nearest value",HM_data_All[Element,BaseImageList])
        MatchMetric_all.append(MatchImages.HM_data_All[Element,BaseImageList])
        MatchMetric_Histo.append(MatchImages.HM_data_histo[Element,BaseImageList])
        MatchMetric_Fourier.append(MatchImages.HM_data_FourierDifference[Element,BaseImageList])
        MatchMetric_FM.append(MatchImages.HM_data_FM[Element,BaseImageList])
        MatchMetric_EigenVectorDotProd.append(MatchImages.HM_data_EigenVectorDotProd[Element,BaseImageList])
        MatchMetric_HOG_Distance.append(MatchImages.HM_data_HOG_Dist[Element,BaseImageList])
        MatchMetric_data_PhaseCorrelation.append(MatchImages.HM_data_PhaseCorrelation[Element,BaseImageList])
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
        #if Counter==1:
        #    HM_data_All[0:len(MatchImages.ImagesInMem_Pairing),Element]=BlankOut
        #    HM_data_All[Element,0:len(MatchImages.ImagesInMem_Pairing)]=BlankOut
        #baseimage should be an integer
        #work in columns to find nearest match, data should be mirrored diagonally to make it easier to visualise#
        
        #move to next element
        BaseImageList=Element

    #create accumulation mean/std deviation -maybe can see where drop off is
    MatchMetric_Filter_mean=[]
    MatchMetric_Filter_std=[]
    FilterSize=5
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

    



    PlotAndSave("MatchMetric_all",MatchImages.OutputPairs +"\\MatchMetric_all.jpg",MatchMetric_all,1)
    PlotAndSave("MatchMetric_Fourier",MatchImages.OutputPairs +"\\MatchMetric_Fourier.jpg",MatchMetric_Fourier,1)
    PlotAndSave("MatchMetric_FM",MatchImages.OutputPairs +"\\MatchMetric_FM.jpg",MatchMetric_FM,1)
    PlotAndSave("MatchMetric_Histo",MatchImages.OutputPairs +"\\MatchMetric_Histo.jpg",MatchMetric_Histo,1)
    PlotAndSave("MatchMetric_FM_EigenVectorDotProd",MatchImages.OutputPairs +"\\MatchMetric_FM_EigenVectorDotProd.jpg",MatchMetric_EigenVectorDotProd,1)
    PlotAndSave("HM_data_HOG_Dist",MatchImages.OutputPairs +"\\HM_data_HOG_Dist.jpg",MatchMetric_HOG_Distance,1)
    PlotAndSave("MatchMetric_data_PhaseCorrelation",MatchImages.OutputPairs +"\\MatchMetric_data_PhaseCorrelation.jpg",MatchMetric_data_PhaseCorrelation,1)

            

    MatchImages.Endtime= time.time()
    print("time taken (hrs):",round((MatchImages.Endtime- MatchImages.startTime)/60/60,2))
    print("time taken (mins):",round((MatchImages.Endtime- MatchImages.startTime)/60,2))
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



        #eigenvector metric
        #get dot product of top eigenvector (should be sorted for most significant set to [0])
        #if using static scene (like MM1 or a movie rather than freely translateable objects)
        #the eigenvector dot product will probably just add noise
        ListEigenDots=[]
        ListEigenVals=[]
        for EVector in range (0,min(len(Base_Image_EigenVectors),len(Test_Image_EigenVectors))):
            #need to look at this closer to see if we need to do anything to vectors before getting dot prod
            ListEigenDots.append(1-round((Base_Image_EigenVectors[EVector] @ Test_Image_EigenVectors[EVector]),8))
            ListEigenVals.append(abs((Base_Image_EigenValues[EVector] )-(Test_Image_EigenValues[EVector] )))

        EigenDotProduct=mean(ListEigenDots)#round((Base_Image_EigenVectors[0] @ Test_Image_EigenVectors[0]),5)
        EigenValue_diff=mean(ListEigenVals)#abs((Base_Image_EigenValues[0] )-(Test_Image_EigenValues[0] ))
        #get distance
        EigenValue_diff=math.sqrt((EigenDotProduct**2)+(EigenValue_diff**2))
        #print(EigenValue_diff)
        #CheckImages_InfoSheet.All_EigenDotProd_result.append(EigenValue_diff)

        #StackTwoimages=MatchImages.StackTwoimages(Base_Image_FM,Test_Image_FM)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(StackTwoimages,(StackTwoimages.shape[1]*1,StackTwoimages.shape[0]*1)),0,True,True)
        

        #histogram metric
        HistogramSimilarity=CompareHistograms(Base_Image_Histo,Test_Image_Histo)
        #CheckImages_InfoSheet.AllHisto_results.append(HistogramSimilarity)

        #feature match metric
        try:
            MatchedPoints,OutputImage,PointsA,PointsB,FinalMatchMetric=_3DVisLabLib.Orb_FeatureMatch(Base_Image_FMatches,Base_Image_Descrips,Test_Image_FMatches,Test_Image_Descrips,99999,Base_Image_FM,Test_Image_FM,0.65,MatchImages.DummyMinValue)
            AverageMatchDistance=FinalMatchMetric#smaller the better
            #print("Feature match",FinalMatchMetric,len(Base_Image_FMatches),len(Test_Image_FMatches))
        except:
            print("ERROR with feature match",len(Base_Image_FMatches),len(Test_Image_FMatches))
            #watch out this might not be a valid maximum!!
            AverageMatchDistance=MatchImages.DummyMinValue
        #CheckImages_InfoSheet.All_FM_results.append(AverageMatchDistance)




        HOG_distance=CompareHistograms(Base_Image_HOG_Descriptor, Test_Image_HOG_Descriptor)


        #fourier difference metric
        #get differnce between fourier magnitudes of image
        #not the best solution as fourier magnitude will rotate with image 
        #generally this performs well on its own as matches similar notes with similar skew
        FourierDifference=(abs(Base_Image_FourierMag-Test_Image_FourierMag)).sum()

        #phase correlation difference
        #use a polar wrapped version of the fourier transform magnitude
        #this is probably a silly way to do this
        #x and y are translation
        #(sx, sy), PhaseCorrelationMatch_raw = cv2.phaseCorrelate(Base_Image_Phase_CorImg, Test_Image_Phase_CorImg)
        #PhaseCorrelationMatch=1-PhaseCorrelationMatch_raw#signal power so we will reverse it 
        #if np.isnan(PhaseCorrelationMatch):
        #    PhaseCorrelationMatch=MatchImages.DummyMinValue
        PhaseCorrelationMatch=1#


        #StackTwoimages=MatchImages.StackTwoimages(Base_Image_FM,Test_Image_FM)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(cv2.resize(StackTwoimages,(StackTwoimages.shape[1]*1,StackTwoimages.shape[0]*1)),0,True,True)
        #populate output metric comparison matrices
        MatchImages.HM_data_histo[CurrentBaseImage,TestImageList]=HistogramSimilarity
        MatchImages.HM_data_FM[CurrentBaseImage,TestImageList]=AverageMatchDistance
        MatchImages.HM_data_FourierDifference[CurrentBaseImage,TestImageList]=FourierDifference
        MatchImages.HM_data_EigenVectorDotProd[CurrentBaseImage,TestImageList]=EigenValue_diff
        MatchImages.HM_data_HOG_Dist[CurrentBaseImage,TestImageList]=HOG_distance
        MatchImages.HM_data_PhaseCorrelation[CurrentBaseImage,TestImageList]=PhaseCorrelationMatch
        #data is symmetrical - fill it in to help with visualisation
        MatchImages.HM_data_histo[TestImageList,CurrentBaseImage]=HistogramSimilarity
        MatchImages.HM_data_FM[TestImageList,CurrentBaseImage]=AverageMatchDistance
        MatchImages.HM_data_FourierDifference[TestImageList,CurrentBaseImage]=FourierDifference
        MatchImages.HM_data_EigenVectorDotProd[TestImageList,CurrentBaseImage]=EigenValue_diff
        MatchImages.HM_data_HOG_Dist[TestImageList,CurrentBaseImage]=HOG_distance
        MatchImages.HM_data_PhaseCorrelation[TestImageList,CurrentBaseImage]=PhaseCorrelationMatch
        
    #build up return object
    ReturnList=dict()
    ReturnList["BASEIMAGE"]=CurrentBaseImage
    ReturnList["HM_data_histo"]=MatchImages.HM_data_histo[CurrentBaseImage,:]
    ReturnList["HM_data_FM"]=MatchImages.HM_data_FM[CurrentBaseImage,:]
    ReturnList["HM_data_FourierDifference"]=MatchImages.HM_data_FourierDifference[CurrentBaseImage,:]
    ReturnList["HM_data_EigenVectorDotProd"]=MatchImages.HM_data_EigenVectorDotProd[CurrentBaseImage,:]
    ReturnList["HM_data_HOG_Dist"]=MatchImages.HM_data_HOG_Dist[CurrentBaseImage,:]
    ReturnList["HM_data_PhaseCorrelation"]=MatchImages.HM_data_PhaseCorrelation[CurrentBaseImage,:]
    return ReturnList