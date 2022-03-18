import _3DVisLabLib
import os
from os.path import exists

#create file list for the sr simulator tool, pointing to dats we wish to test
GetFileList = input("Please enter folder of dats to collect nested files and create filelist for simulator:")

#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(GetFileList)
#Get pickle file - warning will just take first one
ListAllDat_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".dat"])

FileListFilePathAndName=GetFileList + "\\FileList.txt"
if exists(FileListFilePathAndName):
    os.remove(FileListFilePathAndName)
f = open(GetFileList + "\\FileList.txt", "x")

f = open(GetFileList + "\\FileList.txt", "w")
for Datfile in ListAllDat_files:
    f.write(Datfile+"\n")
f.close()