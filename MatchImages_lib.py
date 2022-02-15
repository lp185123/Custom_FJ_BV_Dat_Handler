import copy
import numpy as np
import time
import random
import cv2

def PairWise_Matching(MatchImages,CheckImages_InfoSheet,PlotAndSave_2datas,PlotAndSave,ImgCol_InfoSheet_Class):

    #create reference object of filename VS ID
    ImgNameV_ID=dict()
    #roll through and pull out images vs ID
    for BaseImageList in MatchImages.ImagesInMem_Pairing:
        if len(MatchImages.ImagesInMem_Pairing[BaseImageList][0])!=1:
            raise Exception("MatchImages.ImagesInMem_Pairing",BaseImageList," error, not correct number of images (1)")
        FileName=MatchImages.ImagesInMem_Pairing[BaseImageList][0][0]
        ImgNameV_ID[FileName]=BaseImageList


    #copy input object which should only have 1 image per ID
    ImagesInMem_Pairing=copy.deepcopy(MatchImages.ImagesInMem_Pairing)

    #create indexed dictionary of images referenced by ID so we can start combining lists of images
    #dictionary key has no relevance - only image IDs which are the ID of the result matrices
    Pairings_Indexes=dict()
    for Index, img in enumerate(MatchImages.ImagesInMem_to_Process):
        ImgCol_InfoSheet=ImgCol_InfoSheet_Class()
        Pairings_Indexes["NOTIMG"+ str(Index) + "NOTIMG"]=([Index],ImgCol_InfoSheet)


    MatchImages.HM_data_MetricDistances
    #roll through all list of images
    for BaseImgList in Pairings_Indexes:
        print(BaseImgList,"/",len(Pairings_Indexes))
        
        #if images have been disabled (consumed by another list)
        if Pairings_Indexes[BaseImgList][1].InUse==False:
            continue
        
        
        TestImgLists_Similarities=dict()
        #roll through all list of images
        for TestImgList in Pairings_Indexes:
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
                    print(MatchImages.HM_data_MetricDistances[SingleBaseImg,SingleTestImg])
                    Similarity_List.append(MatchImages.HM_data_MetricDistances[SingleBaseImg,SingleTestImg])

            TestImgLists_Similarities[TestImgList]=Similarity_List

        plop=1
            
                    






    MatchImages.Endtime= time.time()
    print("time taken (hrs):",round((MatchImages.Endtime- MatchImages.startTime)/60/60,2))
    exit()




def PrintResults(MatchImages,CheckImages_InfoSheet,PlotAndSave_2datas,PlotAndSave):
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
    FilePath=MatchImages.OutputPairs +"\\" + str("NoLoop") +  str(OutOfUse) +("MatchImages.HM_data_MetricDistances") +".jpg"
    PlotAndSave_2datas("MatchImages.HM_data_MetricDistances",FilePath,MatchImages.HM_data_MetricDistances)

def SequentialMatchingPerImage(MatchImages,CheckImages_InfoSheet,PlotAndSave_2datas,PlotAndSave):
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
    while len(OrderedImages)+1<len(MatchImages.ImagesInMem_Pairing):#+1 is a fudge or it crashes out with duplicate image bug - cant figure this out 
        Counter=Counter+1
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
        #add to output images
        

        for imgIndex, Images in enumerate (MatchImages.ImagesInMem_Pairing[Element][0]):
            #if len(Images)>1:y
                #raise Exception("too many images")
            SplitImagePath=Images.split("\\")[-1]
            FilePath=MatchImages.OutputPairs +"\\00" +str(Counter)+ "_ImgNo_" + str(BaseImageList) + "_score_" + str(round(MatchImages.HM_data_All[Element,BaseImageList],3))+ "_" + SplitImagePath
            cv2.imwrite(FilePath,MatchImages.ImagesInMem_to_Process[Images].ImageColour)
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

        
        
        
    PlotAndSave("MatchMetric_all",MatchImages.OutputPairs +"\\MatchMetric_all.jpg",MatchMetric_all,1)
    PlotAndSave("MatchMetric_Fourier",MatchImages.OutputPairs +"\\MatchMetric_Fourier.jpg",MatchMetric_Fourier,1)
    PlotAndSave("MatchMetric_FM",MatchImages.OutputPairs +"\\MatchMetric_FM.jpg",MatchMetric_FM,1)
    PlotAndSave("MatchMetric_Histo",MatchImages.OutputPairs +"\\MatchMetric_Histo.jpg",MatchMetric_Histo,1)
    PlotAndSave("MatchMetric_FM_EigenVectorDotProd",MatchImages.OutputPairs +"\\MatchMetric_FM_EigenVectorDotProd.jpg",MatchMetric_EigenVectorDotProd,1)


            

    MatchImages.Endtime= time.time()
    print("time taken (hrs):",round((MatchImages.Endtime- MatchImages.startTime)/60/60,2))
    exit()