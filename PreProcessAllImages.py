###apply "best in class" image preprocessing parameters found by ML stage
###use after GeneticAlg_SNR file
import copy
import copy
import _3DVisLabLib
import cv2
import Snr_test_fitness
import pickle
import GeneticAlg_SNR
#needed for serialised data module (Pickle)
from GeneticAlg_SNR import GA_Parameters
from GeneticAlg_SNR import Individual

def ProcessImages(InputPath=None,OutputPath=None,Processing=False):#dont use if __name__ == "__main__" yet 
    InputFolder=InputPath#r"C:\Working\FindIMage_In_Dat\OutputTestSNR\India"
    #get all files in input folder
    InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputFolder)
    #Get pickle file - warning will just take first one
    ListAllObj_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".obj"])
    #if more than one obj file - warn user
    if (len(ListAllObj_files)!=1) and (Processing==True):
        raise Exception("1 OBJ file expected in , ", InputFolder, " - cannot proceed to process images. Please rectify",len(ListAllObj_files) ," obj files found" )
    
    if (len(ListAllObj_files)==1) and (Processing==True):
        print("Loading saved state, ", ListAllObj_files[0])
        #Unpickle files
        #load saved state into memory
        #if this breaks make sure using "from" explicity importing classes into namespace EG "from GeneticAlg_SNR import Individual"
        GenParams=None
        DictFitCandidates=None
        #probably better using "with" here so dont need to explicitly close file handler
        file_pi2 = open(ListAllObj_files[0], 'rb')
        SaveList=[]
        SaveList = pickle.load(file_pi2)
        file_pi2.close()
        GenParams=copy.deepcopy(SaveList[0])
        DictFitCandidates=copy.deepcopy(SaveList[1])

        #now saved state is rebuilt - load list of images 
        #get all files in input folder
        print("looking in saved state", GenParams.FilePath , "for images - *WARNING* will not conform to nested folder structure")
        InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(GenParams.FilePath)
        #Get list of images
        ListAllImg_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)
        print(len(ListAllImg_files), " images found")

        

        #get fitness records
        FitnessRecords=GenParams.GetEntireFitnessRecord()
        #get last item by insertion - this only will work with python 3.7 onwards or will have to use ordered dictionary
        LastFitnessRecord=(list(FitnessRecords)[-1])
        LastRecord_Parameters=(FitnessRecords[LastFitnessRecord][1])#location of parameters in dictionary value tuple
        print("Parameters for last record:",LastRecord_Parameters)

    #initialise OCR object
    SNR_fitnessTest=Snr_test_fitness.TestSNR_Fitness()

    #convert parameters into SNR/image modify parameter object
    #default parameters if we dont want any processing
    if Processing==False: 
        LastRecord_Parameters=dict()
        #get all files in input folder
        print("looking in", InputPath , "for images - *WARNING* will not conform to nested folder structure")
        InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputPath)
        #Get list of images
        ListAllImg_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)
        print(len(ListAllImg_files), " images found")


    SNRparams=GeneticAlg_SNR.BuildSNR_Parameters(LastRecord_Parameters,SNR_fitnessTest)

    

    for Index,ImageFilePath in enumerate(ListAllImg_files):
        if Index%50==0:
            print("Image",Index,"of",len(ListAllImg_files))
        TestImage=cv2.imread(ImageFilePath,cv2.IMREAD_GRAYSCALE)
        ReturnImg,ReturnFitness=SNR_fitnessTest.RunSNR_With_Parameters(ImageFilePath,SNRparams,TestImage,SkipOcr=True)
        #get delimited string
        Get_SNR_string=ImageFilePath.split("[")#delimit
        Get_SNR_string=Get_SNR_string[-1]#get last element of delimited string
        Get_SNR_string=Get_SNR_string.split("]")#delimit
        Get_SNR_string=Get_SNR_string[0]
        if Get_SNR_string is not None:
            if len(Get_SNR_string)>5:#TODO magic number
                #keep consistent format of SNR read string
                Get_SNR_string="[" + Get_SNR_string +"]"
            else:
                Get_SNR_string="NO_SNR"

        #write image into output folder
        SavePath=OutputPath +"\\" +Get_SNR_string + "I" + str(Index) + ".jpg"
        print("saving image to",SavePath)
        cv2.imwrite(SavePath ,ReturnImg)
        _3DVisLabLib.ImageViewer_Quick_no_resize(ReturnImg,0,False,False)

#main(InputPath=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\India",
#OutputPath=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\ParameterConvergeImages",
#Processing=False)
