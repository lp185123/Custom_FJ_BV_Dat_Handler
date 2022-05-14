import _3DVisLabLib
import os
from os.path import exists

#create file list for the sr simulator tool, pointing to dats we wish to test
GetFileList = input("Please enter folder of dats to collect nested files and create filelist for simulator:")

#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(GetFileList)
#Get pickle file - warning will just take first one
ListAllDat_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".dat"])

#strip out incompatible dats 
CleanedDats=[]
for Dat in ListAllDat_files:
    if not "GBVX_GraphData" in Dat:
        CleanedDats.append(Dat)
ListAllDat_files=CleanedDats

FileListFilePathAndName=GetFileList + "\\FileList.txt"
if exists(FileListFilePathAndName):
    os.remove(FileListFilePathAndName)
f = open(GetFileList + "\\FileList.txt", "x")#???

f = open(GetFileList + "\\FileList.txt", "w")
for Datfile in ListAllDat_files:
    f.write(Datfile+"\n")
f.close()


#native file list for sru simulator
FileListFilePathAndName=r"E:\NCR\Gen_Tools\SRU_Simulator\SRU_MAIN_Rev.639\SRU_MAIN\Debug\FileList.txt"
if exists(FileListFilePathAndName):
    os.remove(FileListFilePathAndName)
f = open(FileListFilePathAndName, "x")#???

f = open(FileListFilePathAndName, "w")
for Datfile in ListAllDat_files:
    f.write(Datfile+"\n")
f.close()

a,b,*c,d=(1,2,3,4,5,6)