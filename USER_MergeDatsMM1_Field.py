import _3DVisLabLib
import Merger_newest
import time
import os
import shutil
InputPath=r"E:\NCR\SR_Generations\Bangladesh\20211130_ReleaseVer2.2.1\TemplateGenData"


#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputPath)
#Get pickle file - warning will just take first one
ListAllDat_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".dat"])




lookupDenomi=dict()
lookupDenomi["D1"]="Denomi.1"
lookupDenomi["D2"]="Denomi.2"
lookupDenomi["D3"]="Denomi.3"
lookupDenomi["D4"]="Denomi.4"
lookupDenomi["D5"]="Denomi.5"
lookupDenomi["D6"]="Denomi.6"
lookupDenomi["D7"]="Denomi.7"
lookupDenomi["D8"]="Denomi.8"



# #get all folder paths for the mm1 genuine field data used to improve SR gen tool performance
# ListPaths=dict()
# for DatItem in ListAllDat_files:

#     if "MergedDamage" == DatItem.split("\\")[-3]:
#         Denom=lookupDenomi[DatItem.split("\\")[-4]]
#         Orientation="Direction" + DatItem.split("\\")[-2]

#         #output folder
#         #E:\NCR\SR_Generations\Bangladesh\20211130_ReleaseVer2.2.1\TemplateGenData\Denomi.1\Damage\Field\DirectionA
#         OutputFolder="E:\\NCR\\SR_Generations\\Bangladesh\\20211130_ReleaseVer2.2.1\\TemplateGenData\\" 
#         OutputFolder=OutputFolder+Denom+"\\Damage\\Field\\"+ Orientation + "\\" + DatItem.split("\\")[-1]

#         print(DatItem)
#         print(OutputFolder)

#         shutil.copy(DatItem,OutputFolder)

#get all folder paths for the mm1 genuine field data used to improve SR gen tool performance
ListPaths=dict()
for DatItem in ListAllDat_files:

    if "Damage" == DatItem.split("\\")[-4]:
        if "Field" == DatItem.split("\\")[-3]:
            #fix the booboo
            if not "20220406" in DatItem.split("\\")[-1]:
                print("Will delete file",DatItem)

                os.remove(DatItem)

            #FolderToMerge=DatItem.replace(DatItem.split("\\")[-1],"")
            #ListPaths[FolderToMerge]="plop"
        #Denom=lookupDenomi[DatItem.split("\\")[-4]]
        #Orientation="Direction" + DatItem.split("\\")[-2]



ssss
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
        #os.rename(datfile,Datfile_dummyExt)