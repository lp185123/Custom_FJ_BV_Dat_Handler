#script to format s39 images into external OCR service formats
#currently needs to have savedstate object in same folder as raw output
#images from extraction software - but can be disabled
from os.path import exists as DoesPathExist
import shutil
import PreProcessAllImages
import TileImages_for_OCR
import _3DVisLabLib
#needed for serialised data module (Pickle)- module must be incorrectly defined
from GeneticAlg_SNR import GA_Parameters
from GeneticAlg_SNR import Individual

#at time of writing point this to folder of S39 images with savedstate obj file (preprocessing)
#can disable preprocessing but need this file until fixed
Input_S39_ExtractionImages=r"C:\Working\FindIMage_In_Dat\Output"
#output folder for single processed files with snr answer as filename
OutputFolderSingleImages=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\ProcessSingles"
#output folder tiled images
OutputFolderTiledImages=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\CollimatedOutput"
#column size of tiled images - if a ML saved state is available this will be overridden
ColumnSize=30
#Preprocessing on/off - if ML optimisation has been used
PreProcessing=True
#can force mirroring or it can be found in genparameters
MirrorImage=False # turn this on if preprocessing is false




#prompt user to check filepaths are OK for deletion
print("Please check output folders can be deleted:\n",OutputFolderSingleImages,"\n",OutputFolderTiledImages)
Response=_3DVisLabLib.yesno("Continue?")
if Response==False:
    raise Exception("User declined to delete folders - process terminated")
#delete output folder
_3DVisLabLib.DeleteFiles_RecreateFolder(OutputFolderSingleImages)
#delete output folder
_3DVisLabLib.DeleteFiles_RecreateFolder(OutputFolderTiledImages)
#check file which links raw extracted images to the dat file & record and copy it into new folders
Img2DatRecordLinker_file="\\" + "TraceImg_to_DatRecord.json"
Img2DatRecordLinker_path=Input_S39_ExtractionImages + Img2DatRecordLinker_file
if DoesPathExist(Img2DatRecordLinker_path):
    shutil.copy(Img2DatRecordLinker_path, OutputFolderSingleImages +Img2DatRecordLinker_file)
#check if saved state exists - if so deserialise it
SavedState_GenParams=PreProcessAllImages.GetML_SavedState(Input_S39_ExtractionImages,PreProcessing)
#create single images
PreProcessAllImages.ProcessImages(InputPath=Input_S39_ExtractionImages,
OutputPath=OutputFolderSingleImages,
Processing=PreProcessing,MirrorImage=MirrorImage,GenParams=SavedState_GenParams)
#create tiled images
TileImages_for_OCR.TileImage("DISPATCH",
OutputFolderSingleImages,OutputFolderTiledImages,ColumnSize,20,SavedState_GenParams)



