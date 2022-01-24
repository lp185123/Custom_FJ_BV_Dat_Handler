###apply "best in class" image preprocessing parameters found by ML stage
###if using ML optimisation stage must place the saved state OBJ in the root of the media source folder
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

def GetML_SavedState(InputPath=None,Processing=False):
    if Processing==False: return None
    #if a saved state is available, find it
    #get all files in input folder
    InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputPath)
    #Get pickle file - warning will just take first one
    ListAllObj_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".obj"])
    #create dummy object if we dont use preprocessing
    GenParams=None
    #if more than one obj file - warn user
    if (len(ListAllObj_files)!=1) and (Processing==True):
        raise Exception("1 OBJ file expected in , ", InputPath, " - cannot proceed to process images. Please rectify",len(ListAllObj_files) ," obj files found" )
    if (len(ListAllObj_files)==1) and (Processing==True):
        print("Loading saved state, ", ListAllObj_files[0])
        #Unpickle files
        #load saved state into memory
        #if this breaks make sure using "from" explicity importing classes into namespace EG "from GeneticAlg_SNR import Individual"
        GenParams=None
        #probably better using "with" here so dont need to explicitly close file handler
        file_pi2 = open(ListAllObj_files[0], 'rb')
        SaveList=[]
        SaveList = pickle.load(file_pi2)
        file_pi2.close()
        #GenParams is the governing object for ML stage such as fitness history and other details
        GenParams=copy.deepcopy(SaveList[0])
        return GenParams

    raise Exception("GetML_SavedState: bad logic chain")
    return None



def ProcessImages(InputPath=None,OutputPath=None,Processing=False,MirrorImage=True,GenParams=None):#dont use if __name__ == "__main__" yet 
    InputFolder=InputPath#r"C:\Working\FindIMage_In_Dat\OutputTestSNR\India"



    # #get all files in input folder
    # InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputFolder)
    # #Get pickle file - warning will just take first one
    # ListAllObj_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".obj"])
    # #create dummy object if we dont use preprocessing
    # GenParams=None
    # #if more than one obj file - warn user
    # if (len(ListAllObj_files)!=1) and (Processing==True):
    #     raise Exception("1 OBJ file expected in , ", InputFolder, " - cannot proceed to process images. Please rectify",len(ListAllObj_files) ," obj files found" )
    if (GenParams is None) and (Processing==True):
        raise Exception("ProcessImages, user option use saved state=True but GenParams object is NONE, cannot proceed")

    if (GenParams is not None) and (Processing==True):
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
        LastRecord_Parameters=(FitnessRecords[LastFitnessRecord][4])#location of parameters in dictionary value tuple
        print("Parameters for last record:",LastRecord_Parameters)
    
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

    #initialise image processing object
    SNR_fitnessTest=Snr_test_fitness.TestSNR_Fitness()
    #build parameters (or not if not processing - but still may need to mirror)
    SNRparams=GeneticAlg_SNR.BuildSNR_Parameters(LastRecord_Parameters,SNR_fitnessTest,None)
    #create folder of single images according to user options (processed/raw/mirrored)
    SNR_fitnessTest.GenerateSingleImages_and_linkFile(ListAllImg_files,OutputPath,SNRparams,GenParams,Processing,MirrorImage)

    return