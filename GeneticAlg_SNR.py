import numpy as np
import copy
from datetime import datetime
import random
import copy
import _3DVisLabLib
import cv2
import Snr_test_fitness
import pickle



class GA_Parameters():
    def __init__(self):
        self.FitnessRecord=dict()
        self.SelfCheck=False
        self.No_of_First_gen=5
        self.No_TopCandidates=5
        self.NewIndividualsPerGen=2
        self.TestImageBatchSize=5
        self.NewImageCycle=4

        #self checking parameters
        self.TargetParameters=dict()
        self.TargetParameters["ResizeX"]=110
        self.TargetParameters["ResizeY"]=100
        self.TargetParameters["Canny"]=0
        self.TargetParameters["AdapativeThreshold"]=210
        self.TargetParameters["MedianBlurDist"]=5
        self.TargetParameters["PSM"]=3
        self.TargetParameters["GausSize_Threshold"]=19
        self.TargetParameters["SubtractMean"]=8
        self.TargetParameters["AlphaBlend"]=0.5
        self.TargetParameters["CropPixels"]=10
        


        self.DictFilename_V_Images=dict()#load this with fitness check images
        #load fitness checking images into memory - #TODO make dynamic
        #self.FilePath=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\FitnessTest"
        self.FilePath=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\India"
        #save out parameter converging image
        self.OutputFolder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\ParameterConvergeImages"
        self.GetRandomSet_TestImages()

        self.LastImage=None
        self.NameOfSavedState=None
        # #get all files in input folder
        # InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(self.FilePath)
        # #filter out non images
        # ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)
        # print("Fitness images at ", self.FilePath)
        # print("Fitness Images to be loaded into memory: ", len(ListAllImages))
        # #roll through images and load into dictionary, with filepath as key (we need filepath to extract potential snr read)
        # for ImageInstance in ListAllImages:
        #     TestImage=cv2.imread(ImageInstance,cv2.IMREAD_GRAYSCALE)
        #     self.DictFilename_V_Images[ImageInstance]=TestImage
        # print("Fitness images loaded")

    def CheckRepeatingFitness(self,Range,Error):
        LastFitness= self.GetFitnessHistory()
        if len(LastFitness)>Range:
            if abs(abs(sum(LastFitness[-Range:]))-abs(LastFitness[-1])*Range)<Error:
                return True
        return False

    def GetRandomSet_TestImages(self):
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
            

        print("Loaded ", len(self.DictFilename_V_Images), "images")

    def FitnessRecordAdd(self,Fitness,Name,Parameters,AverageFitness,AverageAge):

        #check not a duplicate key
        if Fitness in self.FitnessRecord.keys():
            #delete key if it exists - this should probably be a list #TODO
            del self.FitnessRecord[Fitness]
            print("Deleted duplicate fitness record ",Fitness)
        #add record to dictionary
        self.FitnessRecord[Fitness]=(Name,Parameters,AverageFitness,AverageAge)

    def GetEntireFitnessRecord(self):
        return self.FitnessRecord

    def GetFitnessHistory(self):
        fitnessrecords=[]
        for Element in self.FitnessRecord:
            fitnessrecords.append(round(Element,3))
        return fitnessrecords
    
    def GetAverageFitnessHistory(self):
        fitnessrecords=[]
        for Element in self.FitnessRecord:
            fitnessrecords.append(round(self.FitnessRecord[Element][2],3))
        return fitnessrecords
    
    def GetAverageAgeHistory(self):
        fitnessrecords=[]
        for Element in self.FitnessRecord:
            fitnessrecords.append(round(self.FitnessRecord[Element][3],3))
        return fitnessrecords


    def CheckFitness(self,InputParameters,SNR_fitnessTest):

        #tally up error/fitness
        Error=0
        Tapout=False
        ReturnImg=None
        #self check is used to ensure the logic hasnt broken
        if self.SelfCheck==True:
            for I in InputParameters:
                TempError=(InputParameters[I]-self.TargetParameters[I])*(InputParameters[I]-self.TargetParameters[I])
                Error=Error+TempError

        else:
            try:
                fitness, Tapout,ReturnImg=ExternalCheckFitness_SNR(InputParameters,self.DictFilename_V_Images,SNR_fitnessTest)
                Error=1-fitness
                #print("fitness",fitness)
            except Exception as e:
                print("error with checking fitness using parameters")
                print(InputParameters)
                print(e)
                Error=99999999#try and push bad set of parameters to extinction #TODO remove this instead of pushing it down

       
        
        #add tiny amount of random noise to avoid breaking dictionary
        Error=Error+(random.random()/10000)

        
        return Error,Tapout,ReturnImg#have to invert fitness to get error

class Individual():

    def GradientDescent_OneStep(self,PolarityAsInteger,InputParameter):
        #this is a poor mans "gradient descent"
        #take a "step" for parameter in question - will be application specific

        #Check logic is OK
        if InputParameter in self.Parameters.keys():
            pass
        else:
            raise Exception("Error - key does not exist for Parameter", InputParameter, "please review code")

        #TODO might make more sense normalising all parameters

        #decide what a discrete step would be per parameter - would this be better elsewhere? #TODO
        DiscreteStep=0
        if InputParameter=="ResizeX": DiscreteStep=10
        if InputParameter=="ResizeY": DiscreteStep=10
        if InputParameter=="Canny": DiscreteStep=0.3
        if InputParameter=="AdapativeThreshold": DiscreteStep=10
        if InputParameter=="MedianBlurDist": DiscreteStep=2
        if InputParameter=="GausSize_Threshold": DiscreteStep=2
        if InputParameter=="SubtractMean": DiscreteStep=3
        if InputParameter=="AlphaBlend": DiscreteStep=0.2
        if InputParameter=="CropPixels": DiscreteStep=3
        
        #appy discrete step with polarity from user
        self.Parameters[InputParameter]=self.Parameters[InputParameter]+(DiscreteStep*PolarityAsInteger)
        return

    def __init__(self, name):
        #set name of individual according to universal time
        self.name=str(datetime.utcnow()) + str(random.random()) + str(name)
        
        self.PSMset=[3,5,6,7,8,13]#segmentation for SNR modes
        self.GausThresholds=[3,5,7,9,11,13,15,17,19,21]#threshold gaussian values
        self.Parameters=dict()
        #PSM is always 3 - but may not be for more complex formats
        self.Parameters["PSM"]= 3# random.choice(self.PSMset)
        self.Parameters["ResizeX"]=random.randint(80, 250)
        self.Parameters["ResizeY"]=random.randint(80, 250)
        self.Parameters["Canny"]=random.randint(0,1)
        self.Parameters["AdapativeThreshold"]=random.randint(160, 255)
        self.Parameters["MedianBlurDist"]=random.randint(0, 11)
        self.Parameters["GausSize_Threshold"]=random.choice(self.GausThresholds)
        self.Parameters["SubtractMean"]=random.randint(0,15)
        self.Parameters["AlphaBlend"]=random.random()#blend processed and raw image between 0 and 1
        self.Parameters["CropPixels"]=random.randint(0,0)#crop from borders for X and Y 


        #group parameters - as we are addressing vision tools these typically have more than one parameter per function
        #so should cross individuals with groups rather than by discrete parameters
        #self.ParamGroupDict=dict()
        #self.ParamGroupDict["Group1"]=("1","2","3")
        #self.ParamGroupDict["Group2"]=("4","5","6")
        #self.ParamGroupDict["Group3"]=("7","8","9")

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
        self.ApplicationSpecificHouseKeep()

    def SetFitness(self,InputFitness):
        self.Fitness=InputFitness
    def AgeOneGen(self):
        self.Age=self.Age+1
    def ApplicationSpecificHouseKeep(self):
        #fix any invalid numbers
        self.Parameters["MedianBlurDist"]=int(_3DVisLabLib.clamp(self.Parameters["MedianBlurDist"],0,11))
        if self.Parameters["MedianBlurDist"]%2==0:#has to be odd number
            self.Parameters["MedianBlurDist"]=self.Parameters["MedianBlurDist"]+1
        
        
        self.Parameters["PSM"]=self.FixRangedValue(self.Parameters["PSM"],self.PSMset)
        self.Parameters["GausSize_Threshold"]=self.FixRangedValue(self.Parameters["GausSize_Threshold"],self.GausThresholds)

        self.Parameters["AlphaBlend"]=_3DVisLabLib.clamp(self.Parameters["AlphaBlend"],0,1)

        self.Parameters["ResizeX"]=int(_3DVisLabLib.clamp(self.Parameters["ResizeX"],80,250))
        self.Parameters["ResizeY"]=int(_3DVisLabLib.clamp(self.Parameters["ResizeY"],80,250))

        self.Parameters["SubtractMean"]=int(_3DVisLabLib.clamp(self.Parameters["SubtractMean"],0,15))
        self.Parameters["AdapativeThreshold"]=int(_3DVisLabLib.clamp(self.Parameters["AdapativeThreshold"],160,255))
        self.Parameters["CropPixels"]=int(_3DVisLabLib.clamp(self.Parameters["CropPixels"],0,0))


    def FixRangedValue(self,Parameter,Range):
            #keep segmentation mode between these values
        Parameter=int(round(Parameter,0))
        LowestError=9999999#can we do inf here?
        ClosestValue=3
        for PSMvalue in Range:
            Error=(abs(Parameter-PSMvalue))
            if Error<LowestError:
                LowestError=Error
                ClosestValue=PSMvalue
        ReturnValue=ClosestValue
        if not ClosestValue in Range:
            raise Exception("error trying to correct to closest value in set, " , self.Parameters["PSM"])
        return ReturnValue


        
    def RoundParameters(self):
        for elem in self.Parameters:
            self.Parameters[elem]=round(self.Parameters[elem],5)

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
            ListParamsToKeep.append(random.choice(list(self.ParamGroupDict)))

        if len(ListOfParentParameters)==2:
            #roll through groups and randomly assign child group from parent groups
            for Elem in self.ParamGroupDict:#doesnt really matter what parameter list we use as should be all the same
                #"Elem" will match dictionary Keys from Init
                
                if not Elem in ListParamsToKeep:#if we want to skip groups of parameters - for instance
                    #if this individual is initialised then cross-bred with parents - we can leave in a random genome
                    if bool(random.getrandbits(1))==True:
                        #take group from parent 1 
                        for param in self.ParamGroupDict[Elem]:
                            self.Parameters[param]=ListOfParentParameters[0][param]
                    else:
                        #take group from parent 2
                        for param in self.ParamGroupDict[Elem]:
                            self.Parameters[param]=ListOfParentParameters[1][param]
            
            self.RoundParameters()
            self.ApplicationSpecificHouseKeep()

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
                RandomBase_1=(random.random()*2)-1
                #multiply by strength
                RandomBase_1=RandomBase_1*Strength
                #add variation to 100%
                FinalVariation=(RandomBase_1 + 100)/100
                #assign variation
                self.Parameters[gene]=self.Parameters[gene]* FinalVariation
        
        #make sure no invalid parameters hanging around
        self.RoundParameters()
        self.ApplicationSpecificHouseKeep()

def BuildSNR_Parameters(InputParameters,SNR_fitnessTest):
    #use input parameters to drive SNR parameters
    #build SNR input parameters object

    def LoadParameter(SNRParameter,InputParametersObject, InputParameterName):
        try:
            SNRParameter=InputParametersObject[InputParameterName]
        except:
            print("**Warning Failed to load",InputParameterName,"parameter - may be valid error if parameter disabled or obj file did not include parameter")
        return SNRParameter

    #instance of object
    SNRparams=Snr_test_fitness.SNR_Parameters()

    #TODO poor code to make sure is backwards compatible - cant pass by reference as is immutable... needs better solution
    SNRparams.AdapativeThreshold=LoadParameter(SNRparams.AdapativeThreshold,InputParameters,"AdapativeThreshold")
    SNRparams.Canny=LoadParameter(SNRparams.Canny,InputParameters,"Canny")
    SNRparams.MedianBlurDist=LoadParameter(SNRparams.MedianBlurDist,InputParameters,"MedianBlurDist")
    SNRparams.ResizeX=LoadParameter(SNRparams.ResizeX,InputParameters,"ResizeX")
    SNRparams.ResizeY=LoadParameter(SNRparams.ResizeY,InputParameters,"ResizeY")
    SNRparams.PSM=LoadParameter(SNRparams.PSM,InputParameters,"PSM")
    SNRparams.GausSize_Threshold=LoadParameter(SNRparams.GausSize_Threshold,InputParameters,"GausSize_Threshold")
    SNRparams.SubtractMean=LoadParameter(SNRparams.SubtractMean,InputParameters,"SubtractMean")
    SNRparams.AlphaBlend=LoadParameter(SNRparams.AlphaBlend,InputParameters,"AlphaBlend")
    SNRparams.CropPixels=LoadParameter(SNRparams.CropPixels,InputParameters,"CropPixels")

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

def ExternalCheckFitness_SNR(InputParameters,List_of_Fitness_images,SNR_fitnessTest):
    #use input parameters to drive SNR parameters and quantifiy success in range 0: 1

    # #build SNR input parameters object
    # SNRparams=Snr_test_fitness.SNR_Parameters()
    # SNRparams.AdapativeThreshold=InputParameters["AdapativeThreshold"]
    # SNRparams.Canny=InputParameters["Canny"]
    # SNRparams.MedianBlurDist=InputParameters["MedianBlurDist"]
    # SNRparams.ResizeX=InputParameters["ResizeX"]
    # SNRparams.ResizeY=InputParameters["ResizeY"]
    # SNRparams.PSM=str(InputParameters["PSM"])
    # SNRparams.GausSize_Threshold=InputParameters["GausSize_Threshold"]
    # SNRparams.SubtractMean=InputParameters["SubtractMean"]
    # SNRparams.NoProcessing=InputParameters["NoProcessing"]

    SNRparams=BuildSNR_Parameters(InputParameters,SNR_fitnessTest)

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
    for Index, ImagePathInstance in enumerate(List_of_Fitness_images):
        PotentialScore=PotentialScore+1
        #pass the image filepath & name (may have snr read result found between [] brackets), genome parameters for SNR, and image preloaded by ga_params object
        ReturnImg,ReturnFitness=SNR_fitnessTest.RunSNR_With_Parameters(ImagePathInstance,SNRparams,List_of_Fitness_images[ImagePathInstance])
        TotalScore=TotalScore+ReturnFitness
        _3DVisLabLib.ImageViewer_Quick_no_resize(ReturnImg,0,False,False)

        #no point of checking all images if no potential
        if TotalScore<0.01 and Index>5:
            TapOut=True
            break

        #save image with best score
        if BestImageFitness<ReturnFitness:
            BestImg=ReturnImg
            BestImageFitness=ReturnFitness
    
    NormalisedScore=TotalScore/PotentialScore

    return NormalisedScore,TapOut,BestImg
  
def CheckFitness(InputGenDict,GenParams,SNR_fitnessTest):
    print("Check Fitness of ", len(InputGenDict), "genomes")
    DictOfFitness=dict()
    #CheckFitness_MultiProcess(InputGenDict,SNR_fitnessTest,GenParams)
    TotalTapouts=0
    TotalOldTimers=0
    TapOut=False
    #check fitness against static test (fitness=error)
    for I in InputGenDict:
        #we can skip candidates with known fitness by using name as look-up. WARNING! make sure we arent mutating candidates once established!
        if InputGenDict[I].Fitness is None:
            InputGenDict[I].Fitness, TapOut,InputGenDict[I].LastImage=(GenParams.CheckFitness(InputGenDict[I].Parameters,SNR_fitnessTest))
        else:
            TotalOldTimers=TotalOldTimers+1
        DictOfFitness[InputGenDict[I].Fitness]=InputGenDict[I].name
        if TapOut==True:
            TotalTapouts=TotalTapouts+1
            
    print("Tapouts for fitness check = ",TotalTapouts )
    print("Old timers for fitness check = ",TotalOldTimers)
    TotalTapouts=0#TODO shouldnt have to do this



    #sort by error
    SortedFitness=(sorted(DictOfFitness.keys()))

    #TODO here we can start recording matrix stuff for the PCA
    #ensure user hasn't provided incorrect settings
    if GenParams.No_TopCandidates > len(SortedFitness):
        print("No_TopCandidates",  GenParams.No_TopCandidates)
        print( "SortedFitness", len(SortedFitness))
        raise Exception("Error 1, please check configuration of algorithm parameters, topcandidates should be be bigger than pool of individuals. Also possible error if all individuals have same fitness and overwriting keys in fitness dictionary")

    #take slice of most fit candidates
    TopFitness=SortedFitness[0:GenParams.No_TopCandidates]

    #print best fitness
    print("Gen lowest Error (best fitness) = ", SortedFitness[0])
    print("name", InputGenDict[DictOfFitness[SortedFitness[0]]].name)
    print("Parameters", InputGenDict[DictOfFitness[SortedFitness[0]]].Parameters)

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
        filepath=GenParams.OutputFolder + "\\" + (str(InputGenDict[DictOfFitness[SortedFitness[0]]].Fitness).replace(".","")) + ".jpg"
        cv2.imwrite(filepath,InputGenDict[DictOfFitness[SortedFitness[0]]].LastImage)



    #need random number or will overwrite records #TODO must be another dictionary which allows duplicate keys
    randomNo=random.random()/10000
    GenParams.FitnessRecordAdd(SortedFitness[0]+randomNo,InputGenDict[DictOfFitness[SortedFitness[0]]].name,InputGenDict[DictOfFitness[SortedFitness[0]]].Parameters,TotalFitness,TotalAge)
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
        if index>1:
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
            TempIdv.ApplicationSpecificHouseKeep()
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
            TempIdv.ApplicationSpecificHouseKeep()
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
        RandomIndividual=random.randint(0,len(FitList)-1)
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
        TempIdv.mutate(random.randint(0,100),0.01)#1/100 chance of extreme mutation
        TempIdv2.mutate(random.randint(0,100),0.01)#1/100 chance of extreme mutation
        TempIdv3.mutate(random.randint(0,100),0.01)#1/100 chance of extreme mutation
        TempIdv4.mutate(random.randint(0,100),0.01)#1/100 chance of extreme mutation
        #housekeep
        TempIdv.ApplicationSpecificHouseKeep()
        TempIdv2.ApplicationSpecificHouseKeep()
        TempIdv3.ApplicationSpecificHouseKeep()
        TempIdv4.ApplicationSpecificHouseKeep()
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
    for Idv in InputDict_Candidates:
        if str(InputDict_Candidates[Idv].Parameters) in DictParameters.keys():
            pass
        else:
            DictParameters[str(InputDict_Candidates[Idv].Parameters)]=InputDict_Candidates[Idv]
    Duplicatesremoved=len(InputDict_Candidates)-len(DictParameters)
    if Duplicatesremoved >0:
        print("Duplicates removed",Duplicatesremoved)

    for Idv in DictParameters:
        DictProgenitors[DictParameters[Idv].name]=DictParameters[Idv]
    #add randoms to fill back space
    for looper in range (Duplicatesremoved):
        TempIdv=Individual(NameOfGen + "_dp")
        DictProgenitors[TempIdv.name]=copy.deepcopy(TempIdv)#TODO make this a crossbreed instead

    return DictProgenitors, Duplicatesremoved

if __name__ == "__main__":
    #Global timestamp
    Global_timestamp=str(datetime.utcnow())
    Global_timestamp=Global_timestamp.replace(":","")
    Global_timestamp=Global_timestamp.replace("-","")
    Global_timestamp=Global_timestamp.replace(".","")
    #initialise OCR object
    SNR_fitnessTest=Snr_test_fitness.TestSNR_Fitness()

    #initialise working details and fitness testing details
    GenParams=GA_Parameters()
    #create first generation
    DictOfFirstGen=dict()
    
    for  I in range(GenParams.No_of_First_gen):
        NewIndividual=Individual(" Alpha gen")
        if I ==0:
            #force known result from last batch
            pass
            #NewIndividual.Parameters={'PSM': 3, 'Resize': 268.57872, 'Canny': 0.0, 'AdapativeThreshold': 156.11913, 'MedianBlurDist': 3, 'GausSize_Threshold': 19, 'SubtractMean': 5.68184}
            
        DictOfFirstGen[NewIndividual.name]=copy.deepcopy(NewIndividual)#ensure Python is creating instances
    

    # #load into memory
    # filepath=GenParams.OutputFolder +"\\" + "SavedState" + ".obj"
    # file_pi2 = open(filepath, 'rb')
    # SaveList=[]
    # SaveList = pickle.load(file_pi2)
    # file_pi2.close()

    # GenParams=None
    # DictFitCandidates=None
    # GenParams=copy.deepcopy(SaveList[0])
    # DictFitCandidates=copy.deepcopy(SaveList[1])
    # GenParams.TestImageBatchSize=100
    # GenParams.NewImageCycle=99999999
    # GenParams.No_TopCandidates=20


    #assess first generation - here we can potentially load in a saved state
    DictFitCandidates=CheckFitness(DictOfFirstGen,GenParams,SNR_fitnessTest)

    #start main training loop
    for i in range (0,999999):
        GenerationName=" Gen " + str(i+1)
        print("***********" + GenerationName + "***********")
        print("generation size=", len(DictFitCandidates))
        NextGen=CrossBreed(DictFitCandidates,GenerationName)
        print("NextGen size", len(NextGen))
        DictFitCandidates=CheckFitness(NextGen,GenParams,SNR_fitnessTest)
        #add some random individuals
        for I in range(GenParams.NewIndividualsPerGen):
            NewIndividual=Individual(GenerationName + "r")
            DictFitCandidates[NewIndividual.name]=copy.deepcopy(NewIndividual)#ensure Python is creating instances


        #if modulus 20 then do huge image batch
        if i%9999999==0 and i>0:
            GenParams.TestImageBatchSize=20
            GenParams.GetRandomSet_TestImages()
            for individual in DictFitCandidates:
                DictFitCandidates[individual].SetFitness(None)
            GenParams.TestImageBatchSize=4
            continue
            
        

        if i%3==0 and (GenParams.CheckRepeatingFitness(3,0.01))==True:
            StepSize=1
            for looper in range (0,15):
                #gradient descent loop
                print("$$$$$$$$$Gradient descent", looper,"/15")
                DictFitCandidates=GradientDescent(DictFitCandidates,GenerationName +"_gD",StepSize)
                DictFitCandidates,DuplicateCount=RemoveDuplicateIndividuals(DictFitCandidates,GenerationName)
                DictFitCandidates=CheckFitness(DictFitCandidates,GenParams,SNR_fitnessTest)
                #if no response - increase step size
                if (GenParams.CheckRepeatingFitness(2,0.01)==True) and looper>1:
                    print("$$$$$$$$$Gradient descent", "increasing step")
                    StepSize=StepSize+1
                #if no change in fitness - break out
                if (GenParams.CheckRepeatingFitness(3,0.01)==True) and looper>5:
                    print("$$$$$$$$$Gradient descent", "stuck in minima, breaking")
                    break

    
        #every nth cycle mix up fitness testing set
        if i%GenParams.NewImageCycle==0 and i>0:
            print("New set ofimages")
            #get new testing set of images
            GenParams.GetRandomSet_TestImages()
            #will have to retest all old timers fitnesses, setting fitness to None will allow the fitness
            #value to be reasessed
            for individual in DictFitCandidates:
                DictFitCandidates[individual].SetFitness(None)

            #start increasing image batch size every new batch of image(s)
            #if GenParams.TestImageBatchSize< 6 : GenParams.TestImageBatchSize =GenParams.TestImageBatchSize+1

        #check no duplcaites
        DictFitCandidates,DuplicateCount=RemoveDuplicateIndividuals(DictFitCandidates,GenerationName)

        #save state
        if i%3==0:
            SaveList=[GenParams,DictFitCandidates]
            #make sure we can save and load state
            filepath=GenParams.OutputFolder +"\\" + "SavedState" + ".obj"
            GenParams.NameOfSavedState=filepath
            file_pi = open(filepath, 'wb') 
            pickle.dump((SaveList), file_pi)
            file_pi.close()



            #load into memory
            # file_pi2 = open(filepath, 'rb')
            # SaveList=[]
            # SaveList = pickle.load(file_pi2)
            # file_pi2.close()

            # GenParams=None
            # DictFitCandidates=None
            # GenParams=copy.deepcopy(SaveList[0])
            # DictFitCandidates=copy.deepcopy(SaveList[1])




            

