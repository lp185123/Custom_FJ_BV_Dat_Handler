#import numpy as np
import copy
from datetime import datetime
import random
import copy
import _3DVisLabLib
import cv2
import Snr_test_fitness
import pickle
import os
import matplotlib
matplotlib.use('Agg')#can get "Tcl_AsyncDelete: async handler deleted by the wrong thread" crashes otherwise
import matplotlib.pyplot as plt
import VisionAPI_Demo
from math import floor
from decimal import Decimal
#import TileImages_for_OCR

class GA_Parameters():
    def __init__(self):
        self.FitnessRecord=dict()
        self.SelfCheck=False

        #****************
        #remember to check "Individual" class INIT for the image parameters and the parameter ranges!!
        #****************

        self.No_of_First_gen=50#first generation are the randomly generated cycle, this usually gives the biggest gains, balance with 
        #time commitment, which depeneds on test image batch size

        self.No_TopCandidates=8#how many are skimmed off the top after cross-breeding and fitness testing
        #during cross breeding this number can become 4 times as large which will impact process time
        #making this too low and using image subsets(see later) can make older surviving genomes get pushed out if a few sets of non-typical data

        self.NewIndividualsPerGen=1#add a random genome each generation- generally doesnt make much difference so is pot luck 
        #worth activating a couple for long running times as theoretically can help if stuck in local minima

        self.TestImageBatchSize=150#how many images do we test per iteration. For best results use entire dataset, this is tested to work up
        #to 400 images then after that you dont get enough generations to refine solution due to time per evaluation. If you can use entire dataset then
        #set NewImageCycle to 99999 as all it will do is reshuffle the images which isnt necessary

        self.ImageColumnSize=20#how many images we send at a time to external OCR, don't go more than 40 as performance breaks down

        self.NewImageCycle=5#how many iterations/generations of same image set before choosing another random subset of main images. If you set this too low frequency for image switching it could overtrain to the subset

        self.ImageTapOut=1#terminate anything that has poor performance out the box - with collimated data this doesnt work effectively

        self.GradientDescentCycle=9999#loops before we start a gradient descent - this is generally not so critical for images
        # but if commiting to long process time anyway its worth putting it on, will enter grad descent then stop if no improvement
        
        self.UseCloudOCR=True#always set to true - offline pytesseract is available but has not been updated and now may now be inoperable - but can be reactivated with some development
        self.LanguageHints=None#not implemented yet but if using google cloud better results with language hint

        self.MirrorImage=True#use if comparing SNRs (to cope with upside down s39 images), turn off if not comparing SNs only generating

        self.ForceFieldingLengths=None#put in list of expected SN lengths - use this if the s39/images do not yet have SNR answers from the template
        #set to None if the filenames generated are in the format  "file1[AB12345].jpg" whereby the SN can be found between [] in the filename
        #can also have multiple lengths =[5,7,8] if a particularly awkward SN task
        
        self.FilePath=r"C:\Working\FindIMage_In_Dat\Output"#point to folder of extracted images
        
        self.OutputFolder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\ParameterConvergeImages"#save out parameter converging image





        #END OF USER OPTIONS
        self.DictFilename_V_Images=dict()#load this with fitness check images (automatic)
        self.DefaultError=5#if fitness checking breaks what do we set error too - currently cannot handle Null value
        self.LastImage=None
        self.NameOfSavedState=None
        self.Generation=0
        self.BestPerformerTemp=None
        self.Fielding=None
        self.ImageSendCount=0

        self.GetCollimatedTestSet_AndFielding(self.ImageColumnSize,self.TestImageBatchSize)

    def CheckRepeatingFitness(self,Range,Error):
        LastFitness= self.GetFitnessHistory()
        if len(LastFitness)>Range:
            if abs(abs(sum(LastFitness[-Range:]))-abs(LastFitness[-1])*Range)<Error:
                return True
        return False

    def GetCollimatedTestSet_AndFielding(self,ColSize,TotalNoImages):
        #pick N random test images from set of images and collimate with SNR answers
        self.DictFilename_V_Images=dict()
        #get all files in input folder
        InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(self.FilePath)
        #filter out non images
        ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)

        if len(ListAllImages)<TotalNoImages:
            print("WARNING: GetCollimatedTestSet_AndFielding length: User requested ",TotalNoImages,"image batch size from dataset of",len(ListAllImages) )
            print("can continue")
            TotalNoImages=len(ListAllImages)
        if (ColSize)>TotalNoImages:
            print("please ensure column size is <= image batch size")
            raise Exception("ERROR: GetCollimatedTestSet_AndFielding length: User requested",ColSize,"col size from",TotalNoImages, " total images" )

        #convert to dictionary so we can remove items easier
        ListAllImages_dict=dict()
        for image in ListAllImages:
            ListAllImages_dict[image]=image
        
        #pick random imagepath from list and then delete key so no duplicates
        ImagesToLoad=[]
        for ImageBatchSize in range (0,TotalNoImages):
            RandomImage=random.choice(list(ListAllImages_dict.keys()))
            ImagesToLoad.append(RandomImage)
            del ListAllImages_dict[RandomImage]

        #attempt to get fielding of input images:
        self.Fielding=Snr_test_fitness.GenerateSN_Fielding(ListAllImages)

        #load images into memory
        for ImageInstance in ImagesToLoad:
            #block out images as we will handle them elsewhere - but keep this format 
            #TODO will need refactoring if deployed
             TestImage=None#cv2.imread(ImageInstance,cv2.IMREAD_GRAYSCALE)
             self.DictFilename_V_Images[ImageInstance]=TestImage

        print("Selected ", len(self.DictFilename_V_Images), "random images from dataset, to be sorted into columns of size",ColSize )
        print("Fielding is: ",self.Fielding)
        
    
    def GetRandomSet_TestImages_AndFielding(self):
        #pick N random test images from set of images
        self.DictFilename_V_Images=dict()
        #get all files in input folder
        InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(self.FilePath)
        #filter out non images
        ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)

        #convert to dictionary so we can remove items easier
        ListAllImages_dict=dict()
        for image in ListAllImages:
            ListAllImages_dict[image]=image
        
        #pick random imagepath from list and then delete key so no duplicates
        ImagesToLoad=[]
        for ImageBatchSize in range (0,self.TestImageBatchSize):
            RandomImage=random.choice(list(ListAllImages_dict.keys()))
            ImagesToLoad.append(RandomImage)
            del ListAllImages_dict[RandomImage]

        #load images into memory
        for ImageInstance in ImagesToLoad:
             TestImage=cv2.imread(ImageInstance,cv2.IMREAD_GRAYSCALE)
             self.DictFilename_V_Images[ImageInstance]=TestImage
            
        #attempt to get fielding of input images:
        self.Fielding=Snr_test_fitness.GenerateSN_Fielding(ListAllImages)

        print("Loaded ", len(self.DictFilename_V_Images), "images")
        print("Fielding is: ",self.Fielding)

    def FitnessRecordAdd(self,Fitness,Name,Parameters,AverageFitness,AverageAge,ApplicationSpecificParams):

        #check not a duplicate key
        if Fitness in self.FitnessRecord.keys():
            #delete key if it exists - this should probably be a list #TODO
            del self.FitnessRecord[Fitness]
            print("Deleted duplicate fitness record ",Fitness)
        #add record to dictionary
        self.FitnessRecord[Fitness]=(Name,Parameters,AverageFitness,AverageAge,ApplicationSpecificParams)

    def GetEntireFitnessRecord(self):
        return self.FitnessRecord

    def GetFitnessHistory(self):
        fitnessrecords=[]
        for Element in self.FitnessRecord:
            fitnessrecords.append(GA_Round(Element,4))
        return fitnessrecords
    
    def GetAverageFitnessHistory(self):
        fitnessrecords=[]
        for Element in self.FitnessRecord:
            fitnessrecords.append(self.FitnessRecord[Element][2])
        return fitnessrecords
    
    def GetAverageAgeHistory(self):
        fitnessrecords=[]
        for Element in self.FitnessRecord:
            fitnessrecords.append(self.FitnessRecord[Element][3])
        return fitnessrecords


    def CheckFitness_single(self,GenParams,InputParameters,SNR_fitnessTest,SelfTestTargetParameters):
        
        
        #tally up error/fitness
        Error=None
        Tapout=False
        ReturnImg=None
        #self check is used to ensure the logic hasnt broken
        if self.SelfCheck==True:
            Error=0#if selfcheck is on we handle this differently
            for I in InputParameters:
                TempError=(InputParameters[I]-SelfTestTargetParameters[I])*(InputParameters[I]-SelfTestTargetParameters[I])
                Error=Error+TempError
            
        else:
            try:
                #application specific fitness check
                fitness, Tapout,ReturnImg=ExternalCheckFitness_SNR(InputParameters,self.DictFilename_V_Images,SNR_fitnessTest,self.ImageTapOut,GenParams)
                Error=1-fitness
                #monitor external image sends (imagecolumnsize is image stack size, sent as a single image)
                GenParams.ImageSendCount=GenParams.ImageSendCount+(len(self.DictFilename_V_Images)/GenParams.ImageColumnSize)
            except Exception as e:
                print("error with checking fitness using parameters")
                print(InputParameters)
                print(e)
                Error=GenParams.DefaultError#try and push bad set of parameters to extinction #TODO remove this instead of pushing it down
                #warning! An exception might kill a good genome!!

       
        
        #add tiny amount of random noise to avoid breaking dictionary
        #this is not ideal - works for now
        #Error=round(Error,4)
        if Error is None:
            print("Error is None")
        #even if rounding the error still get very long floating points??
        Error=GA_Round(Error,4)
        #print("check fitness single error",Error)
        return Error,Tapout,ReturnImg#have to invert fitness to get error

class Individual():

    def __init__(self, name):
        #set name of individual according to universal time
        self.name=str(datetime.utcnow()) + str(random.random()) + str(name)
        
        self.Parameters=dict()
        self.SelfTestTargetParameters=dict()

        #user populate this area - each parameter has Upper, Lower, range,  test value and if integer
        self.UserInputParamsDict=dict()
        #self.UserInputParamsDict["PSM"]=self.ParameterDetails(3,3,[3,5,6,7,8,13],3,True)#tesseract psm is always 3 so can disable 
        
        
        #with s39 images seems that Y needs resized - so dont bother resizing X
        #self.UserInputParamsDict["ResizeX"]=self.ParameterDetails(100,100,[],100,True)
        self.UserInputParamsDict["ResizeY"]=self.ParameterDetails(200,100,[],150,True)#india x50 y156
        
        self.UserInputParamsDict["AdapativeThreshold"]=self.ParameterDetails(1,0,[1,0],1,True)#keep it always on
        self.UserInputParamsDict["MedianBlurDist"]=self.ParameterDetails(11,0,[0,1,3,5,7,9,11],3,True)#zero will turn it off while otherwise must have odd numbers for the filter
        self.UserInputParamsDict["GausSize_Threshold"]=self.ParameterDetails(9,3,[3,5,7,9],5,True)#must be 3 or greater or breaks
        self.UserInputParamsDict["SubtractMean"]=self.ParameterDetails(20,1,[],7,True)
        self.UserInputParamsDict["AlphaBlend"]=self.ParameterDetails(1,0,[],0.5,False)
        #self.UserInputParamsDict["Denoise"]=self.ParameterDetails(1,0,[0,1],1,True)
        self.UserInputParamsDict["Negative"]=self.ParameterDetails(1,0,[0,1],0,True)
        self.UserInputParamsDict["Equalise"]=self.ParameterDetails(1,0,[0,1],1,True)


        #self.UserInputParamsDict["Canny"]=self.ParameterDetails(1,0,[],0,True)
        #self.UserInputParamsDict["CannyThresh1"]=self.ParameterDetails(50,150,[],100,True)
        #self.UserInputParamsDict["CannyThresh2"]=self.ParameterDetails(100,255,[],110,True)
        #self.UserInputParamsDict["CropPixels"]=self.ParameterDetails(0,0,[],2,True)


        ##custom groupings if we want to stop genomes being split during cross-over (for instance if some work as a collection)
        # self.ParamGroupDict=dict()
        # self.ParamGroupDict["Resizer"]=("ResizeX","ResizeY")
        # self.ParamGroupDict["Cannies"]=("Canny","CannyThresh1","CannyThresh2")
        # self.ParamGroupDict["AdapativeThresholder"]=("AdapativeThreshold","GausSize_Threshold","SubtractMean")
        # self.ParamGroupDict["g1"]=(["AlphaBlend"])
        # self.ParamGroupDict["g2"]=(["MedianBlurDist"])
        # self.ParamGroupDict["g3"]=(["Denoise"])
        # self.ParamGroupDict["g4"]=(["Negative"])



        #all parameters will be normalised
        for Param in self.UserInputParamsDict:
            self.Parameters[Param]=random.random()
            self.SelfTestTargetParameters[Param]=self.UserInputParamsDict[Param].SelfTestTarget


        #group parameters - as we are addressing vision tools these typically have more than one parameter per function
        #so should cross individuals with groups rather than by discrete parameters
        #self.ParamGroupDict=dict()
        #self.ParamGroupDict["Group1"]=("1","2","3")
        #self.ParamGroupDict["Group2"]=("4","5","6")
        #self.ParamGroupDict["Group3"]=("7","8","9")
        #self.ParamGroupDict["Group5"]=(["10"])** NOTE!! have to make sure single item is in a list/iterable

        #default grouping - all parameters operate individually and will not be crossbreed in groups
        self.ParamGroupDict=dict()
        for elem in self.Parameters:
          self.ParamGroupDict[elem + "Group"]=(elem,)#have to force this to be a single element tuple to be compatible with multiple or single items per group

        self.Fitness=None
        self.Age=0
        
        #check everything aligned
        checklist=[]
        for elem in self.ParamGroupDict:
            for item in self.ParamGroupDict[elem]:
                checklist.append(item)
        for Item in self.Parameters:
            if Item in checklist:
                pass
            else:
                raise Exception("Individual _init_ : cannot align parameters with parameter groups - check each parameter is represented in a group, parameter item not found in group : ", Item)

        #handle idiosyncratic invalid values such as odd numbers only etc
        self.HouseKeep()

    class ParameterDetails:
        def __init__(self, Upper,Lower,SetDiscreteRange,SelfTestTarget,isInteger):
            #upper = upper value, lower = lower value, setdiscreterange = [2,4,29] or[] if no range needed
            self.Upper=Upper
            self.Lower=Lower
            self.SetDiscreteRange=SetDiscreteRange
            self.SelfTestTarget=SelfTestTarget
            self.isInteger=isInteger

    def mapFromTo(self,x,a,b,c,d):
    # x:input value; 
    # a,b:input range
    # c,d:output range
    # y:return value
        y=(x-a)/(b-a)*(d-c)+c
        return y

    def ApplicationSpecificMapping(self):
        #Scale parameters from 0-1 scale to application specific scales
        OutputParameters=dict()

        #user parameter dictionary has information on how to map from normalised
        #to application specific parameters
        for param in self.UserInputParamsDict:
            
            Upper=self.UserInputParamsDict[param].Upper
            Lower=self.UserInputParamsDict[param].Lower
            SetRange=self.UserInputParamsDict[param].SetDiscreteRange
            isInteger=self.UserInputParamsDict[param].isInteger

            #if nothing populated throw up an error
            if (Upper is None) or (Lower is None) or (isInteger is None):
                raise Exception(param, "key not found when mapping parameters to application specific range")
            #map from normalised value to application specific
            #if anything special needed can add overriding condition here
            temp=self.mapFromTo(self.Parameters[param],0,1,Lower,Upper)
            if isInteger:temp=int(round(temp))
             #if non linear list range 
            if len(SetRange)!=0:temp=self.FixRangedValue(temp,SetRange)

            if param=="EXAMPLE SPECIAL PARAMETER":
                #do customised logic here
                pass
            #assign mapped value from 0-1 to range of working parameter for testing fitness
            OutputParameters[param]=temp

        if len(self.Parameters)!=len(OutputParameters):
            raise Exception(param, "length mismatch self.parameters & output mapped parameters in application specific range")
        return OutputParameters
        
    def GradientDescent_OneStep(self,PolarityAsInteger,InputParameter):
        #this is a poor mans "gradient descent"
        #take a "step" for parameter in question - will be application specific

        #Check logic is OK
        if InputParameter in self.Parameters.keys():
            pass
        else:
            raise Exception("Error - key does not exist for Parameter", InputParameter, "please review code")

        #TODO might make more sense normalising all parameters

        DiscreteStep=0.01 #all parmaters normalised now so a discrete step is 1% of map range
        #special case for discrete parameters
        if len(self.UserInputParamsDict[InputParameter].SetDiscreteRange)>0:#if a discrete range - handle it differently than continuous paramter
            DiscreteStep=1/len(self.UserInputParamsDict[InputParameter].SetDiscreteRange)
        #decide what a discrete step would be per parameter - would this be better elsewhere? #TODO
        
        #appy discrete step with polarity from user
        self.Parameters[InputParameter]=self.Parameters[InputParameter]+(DiscreteStep*PolarityAsInteger)
        return


    def SetFitness(self,InputFitness):
        self.Fitness=InputFitness
    def AgeOneGen(self):
        self.Age=self.Age+1
    def HouseKeep(self):
        #fix any invalid numbers - keep between 0 and 1
        for Param in self.Parameters:
            self.Parameters[Param]=GA_Round(_3DVisLabLib.clamp(self.Parameters[Param],0,1),4)

    def FixRangedValue(self,Parameter,Range):
        #if this gets too cpu heavy then optimise with successive approximation
            #keep segmentation mode between these values
        Parameter=int(round(Parameter,0))
        LowestError=9999999#can we do inf here?
        ClosestValue=None
        for PSMvalue in Range:
            Error=(abs(Parameter-PSMvalue))
            if Error<LowestError:
                LowestError=Error
                ClosestValue=PSMvalue
        ReturnValue=ClosestValue
        if not ClosestValue in Range:
            raise Exception("error trying to correct to closest value in set, " , self.Parameters[Parameter])
        return ReturnValue

    def RoundParameters(self):
        for elem in self.Parameters:
            self.Parameters[elem]=GA_Round(self.Parameters[elem],5)

    def GetGenes(self):
        #return parameters/gene of individual
        return self.ParamGroupDict,self.Parameters

    def CrossBreed(self,ListOfParentParameters,GenesToKeep):
        #ListOfParentParameters = 2 individuals for now

        #pass in parents for new indvidual
        #here we can mix the genes 50/50 - with individual parameters or with group parameters

        #if GenesToKeep is used - then we dont overwrite the randomly generated intialised genes
        ListParamsToKeep=[]#if genestokeep is zero then this will be ignored
        for I in range (GenesToKeep):
            ListParamsToKeep.append(random.choice(list(self.Parameters)))



        if len(ListOfParentParameters)==2:
            #roll through groups and randomly assign child group from parent groups
            for Elem in self.ParamGroupDict:#doesnt really matter what parameter list we use as should be all the same
                #"Elem" will match dictionary Keys from Init
                
                #if not Elem in ListParamsToKeep:#if we want to skip groups of parameters - for instance
                    #if this individual is initialised then cross-bred with parents - we can leave in a random genome
                if bool(random.getrandbits(1))==True:
                    #take group from parent 1 
                    for param in self.ParamGroupDict[Elem]:
                        self.Parameters[param]=ListOfParentParameters[0][param]
                else:
                    #take group from parent 2
                    for param in self.ParamGroupDict[Elem]:
                        self.Parameters[param]=ListOfParentParameters[1][param]
            
            #add random genes - always between 0 and 1 as is now normalised
            for I in ListParamsToKeep:
                self.Parameters[I]=random.random()

            self.RoundParameters()
            self.HouseKeep()

            return

        raise Exception("Cross breed error")

    def mutate(self,Strength,Extreme_Freq):
        #Strength=magnitude of mutation
        #0 = 100% +/- 0
        #1=  +/- 1%
        #10 = +/- 10%
        #etc

        #Extreme_Freq= set this to a number between 0 and 1, if its below the threshold then an extreme mutation can happen
        if Extreme_Freq > random.random():
            RandomKey=(random.choice(list(self.Parameters)))
            #generate random float between -1 and 1
            RandomBase_1=(random.random()*2)-1
            #multiply by static value for extreme mutation
            RandomBase_1=RandomBase_1*100
            #add variation to 100%
            FinalVariation=(RandomBase_1 + 100)/100
            #assign variation
            self.Parameters[RandomKey]=self.Parameters[RandomKey]*FinalVariation
            #print("xtreme variation ",RandomKey )
        #usual minor mutations
        for gene in self.Parameters:
            if bool(random.getrandbits(1))==True:
                #generate random float between -1 and 1

                RandomSign=random.choice([-1,1])

                RandomValue=random.random()/100#base unit of 0.01

                StrengthRandomValue=(RandomValue*Strength)*RandomSign

                #assign variation
                self.Parameters[gene]=self.Parameters[gene]+ StrengthRandomValue
        
        #make sure no invalid parameters hanging around
        self.RoundParameters()
        self.HouseKeep()

def GA_Round(number,decimalpoints):
    #round() doesnt seem to work 100%
    #return floor(number * 10 ** decimalpoints/ 10 ** decimalpoints)
    #number=format(number, '.4f')
    #return number
    # temp =Decimal(str(number))
    # penny = Decimal('0.01')
    # temp.quantize(penny)
    # return temp
    # return floor(number * 10 ** decimalpoints) / 10 ** decimalpoints
    if number is None:
        return None
    temp = number* (10 **decimalpoints)
    tempint=int(temp)
    tempfloat=tempint/(10 **decimalpoints)
    return tempfloat

def BuildSNR_Parameters(InputParameters,SNR_fitnessTest,GenParams):
    #use input parameters to drive SNR parameters
    #build SNR input parameters object

    # def LoadParameter(SNRParameter,InputParametersObject, InputParameterName):
    #     try:
    #         SNRParameter=InputParametersObject[InputParameterName]
    #     except:
    #         print("**Warning Failed to load",InputParameterName,"parameter - may be valid error if parameter disabled or obj file did not include parameter")
    #     return SNRParameter

    #instance of object
    SNRparams=Snr_test_fitness.SNR_Parameters()

    # #TODO poor code to make sure is backwards compatible - cant pass by reference as is immutable... needs better solution
    # SNRparams.AdapativeThreshold=LoadParameter(SNRparams.AdapativeThreshold,InputParameters,"AdapativeThreshold")
    # SNRparams.Canny=LoadParameter(SNRparams.Canny,InputParameters,"Canny")
    # SNRparams.MedianBlurDist=LoadParameter(SNRparams.MedianBlurDist,InputParameters,"MedianBlurDist")
    # SNRparams.ResizeX=LoadParameter(SNRparams.ResizeX,InputParameters,"ResizeX")
    # SNRparams.ResizeY=LoadParameter(SNRparams.ResizeY,InputParameters,"ResizeY")
    # SNRparams.PSM=LoadParameter(SNRparams.PSM,InputParameters,"PSM")
    # SNRparams.GausSize_Threshold=LoadParameter(SNRparams.GausSize_Threshold,InputParameters,"GausSize_Threshold")
    # SNRparams.SubtractMean=LoadParameter(SNRparams.SubtractMean,InputParameters,"SubtractMean")
    # SNRparams.AlphaBlend=LoadParameter(SNRparams.AlphaBlend,InputParameters,"AlphaBlend")
    # SNRparams.CropPixels=LoadParameter(SNRparams.CropPixels,InputParameters,"CropPixels")
    # SNRparams.CannyThresh1=LoadParameter(SNRparams.CannyThresh1,InputParameters,"CannyThresh1")
    # SNRparams.CannyThresh2=LoadParameter(SNRparams.CannyThresh2,InputParameters,"CannyThresh2")
    

    for paramname in InputParameters:
        if paramname=="AdapativeThreshold":
            SNRparams.AdapativeThreshold=InputParameters[paramname]
            continue
        if paramname=="Canny":
            SNRparams.Canny=InputParameters[paramname]
            continue
        if paramname=="MedianBlurDist":
            SNRparams.MedianBlurDist=InputParameters[paramname]
            continue
        if paramname=="ResizeX":
            SNRparams.ResizeX=InputParameters[paramname]
            continue
        if paramname=="ResizeY":
            SNRparams.ResizeY=InputParameters[paramname]
            continue
        if paramname=="PSM":
            SNRparams.PSM=InputParameters[paramname]
            continue
        if paramname=="GausSize_Threshold":
            SNRparams.GausSize_Threshold=InputParameters[paramname]
            continue
        if paramname=="SubtractMean":
            SNRparams.SubtractMean=InputParameters[paramname]
            continue
        if paramname=="AlphaBlend":
            SNRparams.AlphaBlend=InputParameters[paramname]
            continue
        if paramname=="AdapativeThreshold":
            SNRparams.AdapativeThreshold=InputParameters[paramname]
            continue
        if paramname=="CannyThresh1":
            SNRparams.CannyThresh1=InputParameters[paramname]
            continue
        if paramname=="CannyThresh2":
            SNRparams.CannyThresh2=InputParameters[paramname]
            continue
        if paramname=="CropPixels":
            SNRparams.CropPixels=InputParameters[paramname]
            continue
        if paramname=="Denoise":
            SNRparams.Denoise=InputParameters[paramname]
            continue
        if paramname=="Negative":
            SNRparams.Negative=InputParameters[paramname]
            continue
        if paramname=="Equalise":
            SNRparams.Equalise=InputParameters[paramname]
            continue

        print(paramname, " parameter not found when building SNR parameters")

    if GenParams is not None:
        if GenParams.Generation==0:
            SNRparams=Snr_test_fitness.SNR_Parameters()
    if GenParams is not None: SNRparams.Mirror= GenParams.MirrorImage


    # SNRparams.AdapativeThreshold=InputParameters["AdapativeThreshold"]

    # SNRparams.Canny=InputParameters["Canny"]

    # SNRparams.MedianBlurDist=InputParameters["MedianBlurDist"]

    # SNRparams.ResizeX=InputParameters["ResizeX"]

    # SNRparams.ResizeY=InputParameters["ResizeY"]

    # SNRparams.PSM=str(InputParameters["PSM"])

    # SNRparams.GausSize_Threshold=InputParameters["GausSize_Threshold"]

    # SNRparams.SubtractMean=InputParameters["SubtractMean"]

    # SNRparams.AlphaBlend=InputParameters["AlphaBlend"]

    # SNRparams.CropPixels=InputParameters["CropPixels"]

    return SNRparams

def ExternalCheckFitness_SNR(InputParameters,List_of_Fitness_images,SNR_fitnessTest,ImageTapOut,GenParams):
    #use input parameters to drive SNR parameters and quantifiy success in range 0: 1
    SNRparams=BuildSNR_Parameters(InputParameters,SNR_fitnessTest,GenParams)

    #roll through images and get fitness total
    if len (List_of_Fitness_images)==0:
        raise Exception("no fitness input images!!")
    
    #handle calculating fitness/error SNR_fitnessTest returns a range between 0 and 1 for match
    TotalScore=0
    PotentialScore=0
    TapOut=False
    ReturnImg=None
    BestImg=None
    BestImageFitness=0

    #test code to convert from handling 1 image at a time to handling a list to be collimated and processed
    ReturnImg,ReturnFitness=SNR_fitnessTest.RunSNR_With_Parameters(List_of_Fitness_images,SNRparams,None,SkipOcr=False,GenParams=GenParams,ColSize=GenParams.ImageColumnSize,ListCloudOCR=[CloudOCRObject])
    return ReturnFitness,TapOut,ReturnImg#sending back last image instead so we can potentially see improvement



    for Index, ImagePathInstance in enumerate(List_of_Fitness_images):
        PotentialScore=PotentialScore+1
        #pass the image filepath & name (may have snr read result found between [] brackets), genome parameters for SNR, and image preloaded by ga_params object
        ReturnImg,ReturnFitness=SNR_fitnessTest.RunSNR_With_Parameters([ImagePathInstance],SNRparams,[List_of_Fitness_images[ImagePathInstance]],SkipOcr=False,GenParams=GenParams,ColSize=GenParams.ImageColumnSize)
        TotalScore=TotalScore+ReturnFitness
        _3DVisLabLib.ImageViewer_Quick_no_resize(ReturnImg,0,False,False)

        #no point of checking all images if no potential
        if TotalScore<0.01 and Index>ImageTapOut:
            TapOut=True
            break

        #save image with best score
        if BestImageFitness<ReturnFitness:
            BestImg=ReturnImg
            BestImageFitness=ReturnFitness
    
    NormalisedScore=TotalScore/PotentialScore

    return NormalisedScore,TapOut,ReturnImg#sending back last image instead so we can potentially see improvement
  
def CheckFitness_Multi(InputGenDict,GenParams,SNR_fitnessTest):


    def Reshuffle_Recursive_Add(InputIndv_i, DictGeneration_i,DictOfFitness_i,Age_Penalty,stackno):
            stackno=stackno+1
            #can only have 999 recursions - if we start getting deeper than 10 probably something gone wrong
            #start adding random amounts to the fitness to avoid stack overflow 
            #for instance if we have 1000 members all with same fitness this could potentially happen
            if stackno>500:
                InputIndv_i.Fitness=InputIndv_i.Fitness+(random()/1000)
                print("WARNING!! To avoid recursive function overflow, adding random number to fitness")
            #print("Len gen=",len(DictGeneration_i),"Len fitness=",len(DictOfFitness_i))
            #print("stackn",stackno)
            #print("r enter: ",InputIndv_i.name,InputIndv_i.Fitness)
            #if fitness not in dictionary, can add and exit
            
            if not(InputIndv_i.Fitness in DictOfFitness_i):
                #print("r: adding to dict",InputIndv_i.name,InputIndv_i.Fitness,)
                DictOfFitness_i[InputIndv_i.Fitness]=InputIndv_i.name
                #break out of while
            else:
                #if an individual with the same fitness is in - give boost to the older indv
                #boost = give penalty to younger one and push it down the sorted list
                if InputIndv_i.Age<= DictGeneration_i[DictOfFitness_i[InputIndv_i.Fitness]].Age:
                    #print("r: conflict, is younger, penalty adding",InputIndv_i.name,InputIndv_i.Fitness,)
                    #add age penalty to younger indv
                    InputIndv_i.Fitness=InputIndv_i.Fitness+ Age_Penalty
                    #recursive call here
                    InputIndv_i, DictGeneration_i,DictOfFitness_i,Age_Penalty,stackno=Reshuffle_Recursive_Add(InputIndv_i, DictGeneration_i,DictOfFitness_i,Age_Penalty,stackno)
                else:
                    #print("r: conflict, have to insert as is older ",InputIndv_i.name,InputIndv_i.Fitness,)
                    #trickier problem as now have to add penalty to individual alread in the list
                    #which is same fitness but younger
                    #get individual
                    Fitness=copy.deepcopy(InputIndv_i.Fitness)
                    Idv2Shuffle=DictGeneration_i[DictOfFitness_i[InputIndv_i.Fitness]]
                    #print("r: indv to reshuffle",Idv2Shuffle.name,Idv2Shuffle.Fitness)
                    #delete from dicttionary of fitness
                    #print("len b4 del ",len(DictOfFitness_i),Fitness)
                    del DictOfFitness_i[Fitness]
                    #print("len after del ",len(DictOfFitness_i))
                    #fitness penalty
                    Idv2Shuffle.Fitness=Idv2Shuffle.Fitness+Age_Penalty
                    #print("r: added fitness to reshuffler",Idv2Shuffle.name,Idv2Shuffle.Fitness)
                    #add in current individual
                    #print("r: added older to fitness dict",InputIndv_i.name,InputIndv_i.Fitness,"fitness",Fitness)
                    if Fitness in DictOfFitness_i:
                        print("Error 5: Impossible!!! Should have been deleted")
                    DictOfFitness_i[Fitness]=InputIndv_i.name
                    #print("len after adding older into fitness ",len(DictOfFitness_i))
                    #try to reinsert
                    #print("Inserting younger into loop")
                    InputIndv_i, DictGeneration_i,DictOfFitness_i,Age_Penalty,stackno=Reshuffle_Recursive_Add(Idv2Shuffle, DictGeneration_i,DictOfFitness_i,Age_Penalty,stackno)
            #update stack checker incase we do many recursions due to logic error
            
            return InputIndv_i, DictGeneration_i,DictOfFitness_i,Age_Penalty,stackno


    print("Check Fitness of ", len(InputGenDict), "genomes")
    DictOfFitness=dict()
    TotalTapouts=0
    TotalOldTimers=0
    TapOut=False
    #check fitness against static test (fitness=error)
    print("InputGenDict len", len(InputGenDict), "before sorting")
    for I in InputGenDict:
        #we can skip candidates with known fitness by using name as look-up. WARNING! make sure we arent mutating candidates once established!
        if InputGenDict[I].Fitness is None:

            #map parameters from normalised range to application specific range
            ApplicationParams=InputGenDict[I].ApplicationSpecificMapping()
            InputGenDict[I].Fitness, TapOut,InputGenDict[I].LastImage=(GenParams.CheckFitness_single(GenParams,ApplicationParams,SNR_fitnessTest,InputGenDict[I].SelfTestTargetParameters))
            
        else:
            TotalOldTimers=TotalOldTimers+1
        #TODO fix this - fitnesses are overwriting each other! bad code
        # Current Fitness len
        #CurrentLength=len(DictOfFitness) 
        #for elem in InputGenDict:
        #    if InputGenDict[elem].Fitness is None:
        #        print("b4 Fitness is none!")
        #    print(InputGenDict[elem].Fitness)

        #print("START recursion")
        #InputGenDict[I], DictOfFitness,DictOfFitness,Age_Penalty,stackno=Reshuffle_Recursive_Add(InputGenDict[I], InputGenDict,DictOfFitness,0.01,0)
        #print("END recursion")

        #     if n

        #if InputGenDict[I].Fitness is None:
        #    print("Error 2!! Should never be true!!")
        #if CurrentLength+1!=len(DictOfFitness):
        #    print("Error!!! losing members!!")
            

        if TapOut==True:
            TotalTapouts=TotalTapouts+1
    listMember_names=list(InputGenDict.keys())
    for member in listMember_names:
        Dummy, InputGenDict,DictOfFitness,Age_Penalty,stackno=Reshuffle_Recursive_Add(InputGenDict[member], InputGenDict,DictOfFitness,0.01,0)
        


    print("DictOfFitness len", len(DictOfFitness), "after sorting")
    TotalTapouts=0#TODO shouldnt have to do this

    #sort by error
    SortedFitness=(sorted(DictOfFitness.keys()))


    #TODO here we can start recording matrix stuff for the PCA
    #ensure user hasn't provided incorrect settings
    if GenParams.No_TopCandidates > len(SortedFitness):
        print("*****ERROR*****")
        print("No_TopCandidates",  GenParams.No_TopCandidates)
        print( "SortedFitness", len(SortedFitness))
        raise Exception("Error 1, please check configuration of algorithm parameters,\n topcandidates should be be bigger than pool of individuals.\n Also possible error if all individuals have same fitness and overwriting keys in fitness dictionary - complete convergence")

    #take slice of most fit candidates
    TopFitness=SortedFitness[0:GenParams.No_TopCandidates]


    if TopFitness[0]!=min(TopFitness):
        print("Sorting error for TopFitness!!")
    if SortedFitness[0]!=min(SortedFitness):
        print("Sorting error for SortedFitness!!")


    #print best fitness
    print("Gen lowest Error (best fitness) = ", SortedFitness[0])
    print("name", InputGenDict[DictOfFitness[SortedFitness[0]]].name)
    print("Parameters", InputGenDict[DictOfFitness[SortedFitness[0]]].ApplicationSpecificMapping())
    #print("All fitness:", SortedFitness[:])
    print("Estimated cost",round(GenParams.ImageSendCount*0.001,3),"$   (@March2022 prices 1$ per 1000 Vision OCR calls)")
    #save best performer - this is pretty lazy way to do this
    GenParams.BestPerformerTemp=copy.deepcopy(InputGenDict[DictOfFitness[SortedFitness[0]]])
    # if SortedFitness[0]<2:
    #   sss
    #   exit

    #get moving average of all left in fit list
    TotalFitness=0
    TotalAge=0
    for indv in TopFitness[0:3]:
        TotalFitness=TotalFitness+indv
        TotalAge=TotalAge+InputGenDict[DictOfFitness[indv]].Age
    TotalFitness=TotalFitness/(3)
    TotalAge=TotalAge/(3)
    
    #save out best image #TODO make this formal rather than hard coded- pass in genparams object
    if InputGenDict[DictOfFitness[SortedFitness[0]]].LastImage is not None:
        filepath=GenParams.OutputFolder + "\\ImgPreview_Gen0" + str(GenParams.Generation)  + ".jpg"
        cv2.imwrite(filepath,InputGenDict[DictOfFitness[SortedFitness[0]]].LastImage)

    #save out some plots
    if GenParams.Generation>0:
        PlotAndSave("MostFit_History",GenParams.OutputFolder + "\\MostFit_History_gen" + str(GenParams.Generation) + ".jpg",GenParams.GetFitnessHistory()[:],min(2,max(GenParams.GetFitnessHistory()[:])))
        PlotAndSave("ColonyFitness",GenParams.OutputFolder + "\\ColonyFitness_gen" + str(GenParams.Generation) +".jpg",SortedFitness[:],min(2,max(SortedFitness)))

    #get age of fittest members
    Indv_Age=[]
    Indv_Fitness=[]
    for elemn in SortedFitness:
        Indv_Age.append(InputGenDict[DictOfFitness[elemn]].Age)
        Indv_Fitness.append(InputGenDict[DictOfFitness[elemn]].Fitness)
    try:
        PlotAndSave("TopAge",GenParams.OutputFolder + "\\TopAge_gen" + str(GenParams.Generation) +".jpg",Indv_Age,max(Indv_Age))
        PlotAndSave("TopFitness",GenParams.OutputFolder + "\\TopFitness" + str(GenParams.Generation) +".jpg",Indv_Fitness,max(Indv_Fitness))
        #PlotAndSave_2datas("TopAgeVFitness",GenParams.OutputFolder + "\\TopAgeVFitness" + str(GenParams.Generation) +".jpg",Indv_Age,max(Indv_Age),Indv_Fitness,max(Indv_Fitness))
        PlotAndSave("AverageAge",GenParams.OutputFolder + "\\AverageAge" + str(GenParams.Generation) +".jpg",GenParams.GetAverageAgeHistory(),max(GenParams.GetAverageAgeHistory()))
    except:
        pass
    
    #need random number or will overwrite records #TODO must be another dictionary which allows duplicate keys
    randomNo=random.random()/10000
    GenParams.FitnessRecordAdd(SortedFitness[0]+randomNo,InputGenDict[DictOfFitness[SortedFitness[0]]].name,InputGenDict[DictOfFitness[SortedFitness[0]]].Parameters,TotalFitness,TotalAge,InputGenDict[DictOfFitness[SortedFitness[0]]].ApplicationSpecificMapping())
    #Create dictionary of most fit individuals - copy objects 
    DictFitCandidates=dict()
    for IndvID in TopFitness:
        TempCandidate=InputGenDict[DictOfFitness[IndvID]]
        DictFitCandidates[TempCandidate.name]=copy.deepcopy(TempCandidate)

    #print all fitness records
    print("Most Fit history")
    print(GenParams.GetFitnessHistory())
    #print("Average Fitness history")
    #print(GenParams.GetAverageFitnessHistory())
    #print("Average Age history")
    #print(GenParams.GetAverageAgeHistory())

    return DictFitCandidates

def GradientDescent(InputDict_Candidates,NameOfGen,StepUnits):
    #basic version of gradient descent - find what dimension to move to improve fitness
    ##return original colony plus the gradient descents - original colony should have a Fitness score
    ##so will not need to be processed - but this may change 

    #dictionaries arrive here sorted, first element is the most fit descending
    DictProgenitors=dict()
    #convert to list so we can more conveniently iterate through them
    FitList = list(InputDict_Candidates.items())

    #make sure is sorted by fitness
    DictOfFitness=dict()
    for elem in InputDict_Candidates:
        if InputDict_Candidates[elem].Fitness==None:
            continue
        DictOfFitness[InputDict_Candidates[elem].Fitness]=InputDict_Candidates[elem].name
    SortedFitness=(sorted(DictOfFitness.keys()))


    for index, key in enumerate(SortedFitness):
        Name=DictOfFitness[key]
        Indv=InputDict_Candidates[Name]
        #just do top 2 for now - warning! Magic number!
        if index>0:
            break
        #get genome
        Dummy, Genes1=Indv.GetGenes()
        #roll through genomes for a rough version of gradient descent
        for param in Genes1:
            #get new individual
            TempIdv=Individual(NameOfGen)
            #copy all genomes with no error
            TempIdv.CrossBreed([Genes1,Genes1],0)
            #gradient descent
            TempIdv.GradientDescent_OneStep(StepUnits,param)
            #make sure nothing is going off the scale
            TempIdv.HouseKeep()
            #assign to progenitors
            DictProgenitors[TempIdv.name]=copy.deepcopy(TempIdv)

        for param in Genes1:
            #get new individual
            TempIdv=Individual(NameOfGen)
            #copy all genomes with no error
            TempIdv.CrossBreed([Genes1,Genes1],0)
            #gradient descent
            TempIdv.GradientDescent_OneStep(-1*StepUnits,param)
            #make sure nothing is going off the scale
            TempIdv.HouseKeep()
            #assign to progenitors
            DictProgenitors[TempIdv.name]=copy.deepcopy(TempIdv)

    #copy all previous items into new dictionary
    for index, key in enumerate(SortedFitness):
        Name=DictOfFitness[key]
        Indv=InputDict_Candidates[Name]
        DictProgenitors[Name]=copy.deepcopy(Indv)

    return DictProgenitors

def CrossBreed(InputDict_Candidates,NameOfGen):
    #now have top candidates - start breeding them in pairs - in triplet not yet developed
    DictProgenitors=dict()
    #convert to list so we can more conveniently iterate through them
    FitList = list(InputDict_Candidates.items())

    #for elements of list [0] will be Name of individual, [1] will be individual object
    for index, key in enumerate(FitList):
        if index==len(FitList)-1:
            break
        #RandomIndividual=random.randint(0,len(FitList)-1)
        RandomIndividual=index+1
        #breed in processing pairs
        #create new individual
        TempIdv=Individual(NameOfGen)
        TempIdv2=Individual(NameOfGen)
        TempIdv3=Individual(NameOfGen)
        TempIdv4=Individual(NameOfGen)
        #get parents genes
        Dummy, Genes1=FitList[index][1].GetGenes()
        Dummy, Genes2=FitList[RandomIndividual][1].GetGenes()
        #crossbreed
        TempIdv.CrossBreed([Genes1,Genes2],0)
        TempIdv2.CrossBreed([Genes1,Genes2],0)
        TempIdv3.CrossBreed([Genes1,Genes1],1)#parent1
        TempIdv4.CrossBreed([Genes2,Genes2],2)#parent2
        #mutate
        #TempIdv.mutate(random.randint(0,100),0.00001)#1/100000 chance of extreme mutation
        TempIdv2.mutate(random.randint(0,10),0.00001)#1/100000 chance of extreme mutation
        TempIdv3.mutate(random.randint(0,10),0.00001)#1/100000 chance of extreme mutation
        TempIdv4.mutate(random.randint(0,10),0.00001)#1/100000 chance of extreme mutation
        #housekeep
        TempIdv.HouseKeep()
        TempIdv2.HouseKeep()
        TempIdv3.HouseKeep()
        TempIdv4.HouseKeep()
        #assign to new generation
        DictProgenitors[TempIdv.name]=copy.deepcopy(TempIdv)
        DictProgenitors[TempIdv2.name]=copy.deepcopy(TempIdv2)
        DictProgenitors[TempIdv3.name]=copy.deepcopy(TempIdv3)
        DictProgenitors[TempIdv4.name]=copy.deepcopy(TempIdv4)

        #some parents survive to next generation
        if index<2:#always keep in best performing previous gen
            DictProgenitors[FitList[index][1].name]=copy.deepcopy(FitList[index][1])
            DictProgenitors[FitList[index][1].name].AgeOneGen()
        else:
            if bool(random.getrandbits(1))==True:
                DictProgenitors[FitList[index][1].name]=copy.deepcopy(FitList[index][1])
                DictProgenitors[FitList[index][1].name].AgeOneGen()

    #add more 
    return DictProgenitors

def RemoveDuplicateIndividuals(InputDict_Candidates,NameOfGen):
    DictParameters=dict()#use dictionary overwrite method
    DictProgenitors=dict()
    ListDuplicateIndvs=[]
    for Idv in InputDict_Candidates:
        if str(InputDict_Candidates[Idv].Parameters) in DictParameters.keys():
            #add to naughty list
            ListDuplicateIndvs.append(Idv)
        else:
            DictParameters[str(InputDict_Candidates[Idv].Parameters)]=InputDict_Candidates[Idv]

    Duplicatesremoved=len(InputDict_Candidates)-len(DictParameters)
    if Duplicatesremoved >0:
        print("Duplicates will be removed",Duplicatesremoved)
    for Idv in DictParameters:
        DictProgenitors[DictParameters[Idv].name]=DictParameters[Idv]
    #add randoms/mutated to fill back space
    for Idv in ListDuplicateIndvs:
        TempIdv=Individual(NameOfGen + "_dp")
        Dummy,Genes=InputDict_Candidates[Idv].GetGenes()
        TempIdv.CrossBreed([Genes,Genes],2)#mutate 2 genes - better strategy than new genome
        DictProgenitors[TempIdv.name]=copy.deepcopy(TempIdv)
    #for looper in range (Duplicatesremoved):
     #   TempIdv=Individual(NameOfGen + "_dp")
     #   DictProgenitors[TempIdv.name]=copy.deepcopy(TempIdv)#TODO make this a crossbreed instead

    return DictProgenitors, Duplicatesremoved

def RemoveCloseParameterIndividuals(InputDict_Candidates,NameOfGen,Buffer):
    #remove individuals that has similiar traits within a set range, in a sequential manner
    #replace removed individuals with mutated versions of existing
    #if genomes are close in value they take up processing time
    Copy_InputDict_Candidates=copy.deepcopy(InputDict_Candidates)

    RemovalList=[]
    #wrap in a function so we can dont modifiy dictionary during loop
    #WARNING bad practise relying on scoped variables!
    def RemoveDuplicate():
        for I in Copy_InputDict_Candidates:
            BaseParameters=Copy_InputDict_Candidates[I].Parameters
            for TestI in Copy_InputDict_Candidates:
                TestPass=0
                if TestI==I:#skip over same indiv
                    continue
                if TestI in RemovalList:#dont add second time around
                    continue
                #test each parameter has sufficient difference
                TestParameters=Copy_InputDict_Candidates[TestI].Parameters
                for Param in BaseParameters:
                    if abs(BaseParameters[Param]-TestParameters[Param])<Buffer:
                        TestPass=TestPass+1
                #add up total
                if TestPass==len(TestParameters):
                    #all parameters are almost the same - add to naughty list
                    RemovalList.append(Copy_InputDict_Candidates[TestI])
                    #print("&&&&&added ",Copy_InputDict_Candidates[TestI].name)
                    del Copy_InputDict_Candidates[Copy_InputDict_Candidates[TestI].name]
                    #print(I,BaseParameters)
                   #print(TestI,TestParameters)
                    return False
        return True
               
    #loop parameter similarity check so not modifying dictionaries within loops, making avoiding double adds simpler
    while RemoveDuplicate()==False:
        pass

    if len(RemovalList)>0:
        print("Removed",len(RemovalList), "with similar parameters")
        
    for elem in RemovalList:
        TempIdv=Individual(NameOfGen + "_sim")
        Dummy,Genes=elem.GetGenes()
        TempIdv.CrossBreed([Genes,Genes],2)#mutate 2 genes - better strategy than completely new genome
        Copy_InputDict_Candidates[TempIdv.name]=copy.deepcopy(TempIdv)


    if len(InputDict_Candidates)!=len(Copy_InputDict_Candidates):
        raise Exception("Error removing close fitness genomes - output dictionary does not match")

    return InputDict_Candidates

def PlotAndSave(Title,Filepath,Data,maximumvalue):
    
    #this causes crashes
    #save out plot of 1D data
    try:
        plt.plot(Data)
        plt.ylabel(Title)
        plt.ylim([0, max(Data)])
        plt.savefig(Filepath)
        plt.cla()
        plt.clf()
        plt.close()
    except Exception as e:
        print("Error with matpyplot",e)

def PlotAndSave_2datas(Title,Filepath,Data1,maximumvalue1,Data2,maximumvalue2):
    
    #this causes crashes
    #save out plot of 1D data
    try:
        plt.plot(Data1,Data2,'bo')#bo will draw dots instead of connected line
        plt.ylabel(Title)
        plt.ylim([0, max(Data1)])
        plt.ylim([0, max(Data2)])
        plt.savefig(Filepath)
        plt.cla()
        plt.clf()
        plt.close()
    except Exception as e:
        print("Error with matpyplot",e)

def RemoveIndividualsCloseFitness(InputDict_Candidates,FitnessBuffer,NameOfGen):
    #return InputDict_Candidates
    #remove individuals that are close to duplicate fitness, and most probably will have almost same parameters
    #sort by error
    #TODO this might be legitimate convergence with similar 
    Copy_InputDict_Candidates=copy.deepcopy(InputDict_Candidates)

    DictOfFitness=dict()
    for I in Copy_InputDict_Candidates:
        if not Copy_InputDict_Candidates[I].Fitness is None:#might not have been assessed at this stage and will be None
            DictOfFitness[Copy_InputDict_Candidates[I].Fitness]=Copy_InputDict_Candidates[I].name
    SortedFitness=(sorted(DictOfFitness.keys()))

    ListForRemoval=[]
    if len(SortedFitness)>0:
        BaseFitness=SortedFitness[0]
        for Fitness in SortedFitness:
            if BaseFitness==Fitness:#same genomes - skip - should never have same fitnesses under any circumstances as dictionary
                #will not allow duplicate keys
                continue
            if abs(Fitness-BaseFitness)<FitnessBuffer:
                ListForRemoval.append(Fitness)#add genome to list for removal
            if abs(Fitness-BaseFitness)>FitnessBuffer:
                BaseFitness=Fitness#cleared out buffer of fitness, onto next buffer pitch

        #clean up duplicate fitnesses, instead of deleting lets mutate them

        if len(ListForRemoval)>0:
            for Idv in InputDict_Candidates:
                if InputDict_Candidates[Idv].Fitness in ListForRemoval:
                    TempIdv=Individual(NameOfGen + "_fcu")
                    Dummy,Genes=InputDict_Candidates[Idv].GetGenes()
                    TempIdv.CrossBreed([Genes,Genes],2)#mutate 2 genes - better strategy than new genome
                    del Copy_InputDict_Candidates[Idv]
                    Copy_InputDict_Candidates[TempIdv.name]=copy.deepcopy(TempIdv)

                    if len(InputDict_Candidates)!=len(Copy_InputDict_Candidates):
                        raise Exception("Error removing close fitness genomes - output dictionary does not match")

        return Copy_InputDict_Candidates

            
            

            

        


    print("SortedFitness",SortedFitness)
    return(InputDict_Candidates)

if __name__ == "__main__":
    
    global CloudOCRObject#create Cloud OCR object as global as it interferes when saving the state
    CloudOCRObject=None
    #Global timestamp
    Global_timestamp=str(datetime.utcnow())
    Global_timestamp=Global_timestamp.replace(":","")
    Global_timestamp=Global_timestamp.replace("-","")
    Global_timestamp=Global_timestamp.replace(".","")
    #initialise OCR object
    SNR_fitnessTest=Snr_test_fitness.TestSNR_Fitness()
    #initialise working details and fitness testing details
    GenParams=GA_Parameters()
    #create cloud OCR if we are using it
    if GenParams.UseCloudOCR==True:
        #instancing class will initialise Cloud service and attempt to authenticate agent
        CloudOCRObject=VisionAPI_Demo.CloudOCR()
   
    #load saved state into memory if it exists
    filepath=GenParams.FilePath +"\\" + "SavedState" + ".obj"
    if os.path.isfile(filepath)==True:
        print("Saved state.obj detected:",filepath)
        print("Previous state detected - load state? WARNING - if saved state generated by older version of script may cause error")
        #get input from user after previous state detected
        if _3DVisLabLib.yesno("load previous state?")==True:
            file_pi2 = open(filepath, 'rb')
            SaveList=[]
            SaveList = pickle.load(file_pi2)
            file_pi2.close()
            #depickle genomes and working object
            GenParams=None
            DictFitCandidates=None
            GenParams=copy.deepcopy(SaveList[0])
            DictFitCandidates=copy.deepcopy(SaveList[1])
            #custom modifications to saved state - warning - can cause crash if incompatible new parameters such as
            #more top candidates than generation can provide
            #print("WARNING! Modifying existing run parameters")
            #GenParams.TestImageBatchSize=10
            #GenParams.NewImageCycle=5
            #GenParams.No_TopCandidates=20
            #GenParams.GetRandomSet_TestImages_AndFielding()
            
        else:
            #prompt user to check filepaths are OK for deletion
            print("Please check output folder can be deleted:\n",GenParams.OutputFolder)
            Response=_3DVisLabLib.yesno("Delete contents (y) - continue writing (n) ?")
            if Response==True:
                _3DVisLabLib.DeleteFiles_RecreateFolder(GenParams.OutputFolder)
    else:
        #prompt user to check filepaths are OK for deletion
        print("Please check output folder can be deleted:\n",GenParams.OutputFolder)
        Response=_3DVisLabLib.yesno("Delete contents (y) - continue writing (n) ?")
        if Response==True:
            _3DVisLabLib.DeleteFiles_RecreateFolder(GenParams.OutputFolder)
        
    #start main training loop
    for i in range (GenParams.Generation, 999999):
        GenerationName=" Gen " + str(i)
        print("***********" + GenerationName + "***********")

        #if first generation - get a baseline score for performance without image processing
        if GenParams.Generation==0:
            #create test generation with no processing
            print("Generating baseline fitness with no processing")
            #create population of 1 with no processing (generation=0 detected elsewhere)
            DictOfFirstGen=dict()
            NewIndividual=Individual("baseline gen")
            Temp_No_of_First_gen=GenParams.No_of_First_gen
            Temp_No_TopCandidates=GenParams.No_TopCandidates
            GenParams.No_of_First_gen=1
            GenParams.No_TopCandidates=1
            DictOfFirstGen[NewIndividual.name]=copy.deepcopy(NewIndividual)
            DictFitCandidates=CheckFitness_Multi(DictOfFirstGen,GenParams,SNR_fitnessTest)
            GenParams.No_of_First_gen=Temp_No_of_First_gen
            GenParams.No_TopCandidates=Temp_No_TopCandidates
            #keep persistant generation aligned
            GenParams.Generation=GenParams.Generation+1
            continue
            
        if GenParams.Generation==1:
            #create seed population
            print("Generating first seed generation")
            DictOfFirstGen=dict()
            for  I in range(GenParams.No_of_First_gen):
                NewIndividual=Individual(" Alpha gen")
                DictOfFirstGen[NewIndividual.name]=copy.deepcopy(NewIndividual)#ensure Python is creating instances
            DictFitCandidates=CheckFitness_Multi(DictOfFirstGen,GenParams,SNR_fitnessTest)
            #keep persistant generation aligned
            GenParams.Generation=GenParams.Generation+1
            continue

        if GenParams.Generation>1:
            DictFitCandidates=CheckFitness_Multi(DictFitCandidates,GenParams,SNR_fitnessTest)
        print("generation size=", len(DictFitCandidates))
        DictFitCandidates=CrossBreed(DictFitCandidates,GenerationName)
        print("NextGen size", len(DictFitCandidates))
        #add some random individuals
        for I in range(GenParams.NewIndividualsPerGen):
            NewIndividual=Individual(GenerationName + "r")
            DictFitCandidates[NewIndividual.name]=copy.deepcopy(NewIndividual)#ensure Python is creating instances




        # print("generation size=", len(DictFitCandidates))
        # NextGen=CrossBreed(DictFitCandidates,GenerationName)
        # print("NextGen size", len(NextGen))
        # DictFitCandidates=CheckFitness_Multi(NextGen,GenParams,SNR_fitnessTest)
        # #add some random individuals
        # for I in range(GenParams.NewIndividualsPerGen):
        #     NewIndividual=Individual(GenerationName + "r")
        #     DictFitCandidates[NewIndividual.name]=copy.deepcopy(NewIndividual)#ensure Python is creating instances

        #every n loops and if we arent having much movement with fitness, and fitness is >0 (hasnt converged)
        #WARNING: if fitness is oscillating this will not meet condition, perhaps check gradient as well
        if (i%GenParams.GradientDescentCycle==0 and (GenParams.CheckRepeatingFitness(3,0.01))==True) and (GenParams.GetFitnessHistory()[-1]>0.01):
            #here check that fitness isnt near 0 as we should get a new batch of evaluation data if so 
            StepSize=5
            for looper in range (0,15):
                #gradient descent loop
                print("$$$$$$$$$Gradient descent", looper,"/15")
                DictFitCandidates_gd=GradientDescent(DictFitCandidates,GenerationName +"_gD",StepSize)
                DictFitCandidates_gd=RemoveCloseParameterIndividuals(DictFitCandidates_gd,GenerationName,0.02)
                DictFitCandidates_gd=CheckFitness_Multi(DictFitCandidates_gd,GenParams,SNR_fitnessTest)

                #if gradient descent has done nothing - don't keep the gradient descent prognitors
                if (GenParams.CheckRepeatingFitness(2,0.001)==True) and looper>1:
                    print("$$$$$$$$$Gradient descent, no improvement")
                    pass
                else:
                    #get best performer held in temporary storage and add to original group
                    DictFitCandidates[GenParams.BestPerformerTemp.name]=copy.deepcopy(GenParams.BestPerformerTemp)
                    print("$$$$$$$$$Gradient descent, improvement - adding to progenitors")

                #if no response - increase step size
                if (GenParams.CheckRepeatingFitness(4,0.01)==True) and looper>1:
                    print("$$$$$$$$$Gradient descent", "increasing step")
                    StepSize=StepSize+3
                #if no change in fitness - break out
                if (GenParams.CheckRepeatingFitness(3,0.01)==True) and looper>5:
                    print("$$$$$$$$$Gradient descent", "stuck in minima, breaking")
                    break
    
        #every nth cycle mix up fitness testing set, or if fitness has converged to close to zero
        if (i%GenParams.NewImageCycle==0 and i>0):# or (GenParams.GetFitnessHistory()[-1]<0.01):
            print("New set ofimages")
            #get new testing set of images
            GenParams.GetCollimatedTestSet_AndFielding(GenParams.ImageColumnSize,GenParams.TestImageBatchSize)
            #will have to retest all old timers fitnesses, setting fitness to None will allow the fitness
            #value to be reasessed
            for individual in DictFitCandidates:
                DictFitCandidates[individual].SetFitness(None)

        #check no duplicates
        DictFitCandidates=RemoveCloseParameterIndividuals(DictFitCandidates,GenerationName,0.02)

        #save state every generation loop
        if i%1==0:
            #cant pickle the cloud object
            SaveList=[GenParams,DictFitCandidates]
            #make sure we can save and load state
            filepath_working=GenParams.OutputFolder +"\\" + "SavedState" + ".obj" 
            filepath_local=GenParams.FilePath +"\\" + "SavedState" + ".obj"
            GenParams.NameOfSavedState=filepath_local
            #save local copy
            file_pi = open(filepath_local, 'wb') 
            pickle.dump((SaveList), file_pi)
            file_pi.close()
            #save working copy
            file_working = open(filepath_working, 'wb') 
            pickle.dump((SaveList), file_working)
            file_working.close()

        #keep persistant generation aligned
        GenParams.Generation=GenParams.Generation+1

