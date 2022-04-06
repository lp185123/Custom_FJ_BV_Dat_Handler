import _3DVisLabLib
import Merger_newest
import time
import os
InputPath=r"E:\NCR\Currencies\Bangladesh_SR2800\Bangladesh\SR DC\MM1\TEST_SETS\D2\4A\ATM Fit"


#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputPath)
#Get pickle file - warning will just take first one
ListAllDat_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".dat"])



#merge dats together in folder
merger = Merger_newest.ImageMerger()

#give to merger tool
merger.files = ListAllDat_files
merger.outputDirectory=InputPath + "\\"
merger.start()
time.sleep(5)#a
