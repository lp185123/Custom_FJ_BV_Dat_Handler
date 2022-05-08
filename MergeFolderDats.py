from ctypes.wintypes import HHOOK

from numpy import DataSource
import _3DVisLabLib
#import ImageExtractorModule_testNoMM8 as Merger_newest
import Merger_newest
import time
import os
InputPath=r"E:\NCR\Currencies\Bangladesh_SR2800\Bangladesh\SR DC\MM8\100\04_100(2006-2010)\Brand New_100pcs Notes\MM8\D"


#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputPath)
#Get pickle file - warning will just take first one
ListAllDat_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".dat"])

RenamedFiles = [Filename.replace(".dat",".zat") for Filename in ListAllDat_files]

#create dictionary
FileToRename_dict=dict()
for OldFile,NewFile in zip(ListAllDat_files,RenamedFiles):
    FileToRename_dict[OldFile]=NewFile



#merge dats together in folder
merger = Merger_newest.ImageMerger()

#give to merger tool
merger.files = ListAllDat_files
merger.outputDirectory=InputPath + "\\"
merger.start()
#time.sleep(5)#a

#rename dats

xxx

for OldFile in FileToRename_dict:
    NewFile=FileToRename_dict[OldFile]
    os.rename(OldFile,NewFile)
