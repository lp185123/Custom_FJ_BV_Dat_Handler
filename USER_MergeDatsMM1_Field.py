import _3DVisLabLib
import Merger_newest
import time
import os
InputPath=r"E:\NCR\SR_Generations\Bangladesh\20211130_ReleaseVer2.2.1\TemplateGenData"


#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputPath)
#Get pickle file - warning will just take first one
ListAllDat_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".dat"])



#get all folder paths for the mm1 genuine field data used to improve SR gen tool performance
ListPaths=dict()
for DatItem in ListAllDat_files:
    if "Genuine\Field\Direction" in DatItem:
        FileName=DatItem.split("\\")[-1]
        FilePath=DatItem.replace(FileName,"")
        ListPaths[FilePath]=FilePath





#merge dats together in folder
merger = Merger_newest.ImageMerger()

for FolderPath in ListPaths:
    print("Merging dats found in ",FolderPath)
    #get all files in input folder
    Int_InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(FolderPath)
    #Get pickle file - warning will just take first one
    Int_ListAllDat_files=_3DVisLabLib.GetList_Of_ImagesInList(Int_InputFiles,ImageTypes=[".dat"])
    
    #give to merger tool
    merger.files = Int_ListAllDat_files
    merger.outputDirectory=FolderPath

    merger.start()
    time.sleep(5)#at the moment the merger tool uses time for filename with resoluton of 1 sec,
    #potentially can overwrite small collection of dats. Use this as a workaround until theres an update

    #remove/rename all files that have been merged
    for datfile in Int_ListAllDat_files:
        Datfile_dummyExt=datfile.lower().replace(".dat",".zat")
        os.rename(datfile,Datfile_dummyExt)