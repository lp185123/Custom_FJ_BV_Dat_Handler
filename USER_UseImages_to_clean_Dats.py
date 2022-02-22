import DatScraper_tool_broken_onlyExtract_NEW
import _3DVisLabLib
import json
import shutil
import os

InputPath=r"C:\Working\FindIMage_In_Dat\Output"
InputJson=InputPath+"\\TraceImg_to_DatRecord.json" 
OutputFolder=r"C:\Working\FindIMage_In_Dat\Output"
RecordOffset=-1#fudge factor
with open(InputJson) as json_file:
    ImgVDatFile_andRecord = json.load(json_file)

#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputPath)
#Get pickle file - warning will just take first one
ListAllJpg_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".jpg"])

RecordToRemove_dict=dict()
RecordToKeep_dict=dict()
#filenames have to match EXACTLY or this will cause issues
for imgfilename in ImgVDatFile_andRecord:
    if imgfilename in ListAllJpg_files:
        DatFilename=ImgVDatFile_andRecord[imgfilename][0]
        DatRecord=ImgVDatFile_andRecord[imgfilename][1]
        if DatFilename in RecordToKeep_dict:
            RecordToKeep_dict[DatFilename].append(int(DatRecord)+RecordOffset)
        else:
            RecordToKeep_dict[DatFilename]=[int(DatRecord)+RecordOffset]
    else:
        DatFilename=ImgVDatFile_andRecord[imgfilename][0]
        DatRecord=ImgVDatFile_andRecord[imgfilename][1]
        if DatFilename in RecordToRemove_dict:
            RecordToRemove_dict[DatFilename].append(int(DatRecord)+RecordOffset)
        else:
            RecordToRemove_dict[DatFilename]=[int(DatRecord)+RecordOffset]

print("Dat files operated upon:",len(RecordToKeep_dict))
print("WARNING! If filenames are not identical this process will fail\nyou will be able to review actions before commiting")
#should have a dictionary with the missing images aligned with dat files and list
#we can now spin through the dictionary and pass the list to the dat scraper library

#WIP 
if (_3DVisLabLib.yesno("WIP - is this single record MM8 data and only record images you want remain in image proxy?")):
#if only one record per dat file have to use this!! IE MM8 data
    for datfiletoclean in RecordToKeep_dict:
        OriginalFile=datfiletoclean
        FileName=OriginalFile.split("\\")[-1]
        dst=OutputFolder +"\\" + FileName
        print("\nWill copy:    ",OriginalFile,"\nto:     ",dst)
    exit
    if (_3DVisLabLib.yesno("do you wish to proceed?")):
        for datfiletoclean in RecordToKeep_dict:
            OriginalFile=datfiletoclean
            FileName=OriginalFile.split("\\")[-1]
            dst=OutputFolder +"\\" + FileName
            print("Copying",OriginalFile,"to",dst)
            shutil.copyfile(datfiletoclean, dst)
    else:
        raise Exception("User declined to continue")


if (_3DVisLabLib.yesno("WIP - is this multiple record data and records to be excluded have been deleted from image proxy?")):
#otherwise use this if lots of records per dat
    for datfiletoclean in RecordToRemove_dict:
        print("will remove",RecordToRemove_dict[datfiletoclean] , "from",datfiletoclean)
        
    exit

    if (_3DVisLabLib.yesno("do you wish to proceed? WARNING! WILL OVERWRITE YOUR DATS!!!")):
        for datfiletoclean in RecordToRemove_dict:
            DelimitedFilePath=datfiletoclean.split("\\")
            BaseFilePath=datfiletoclean.replace(DelimitedFilePath[-1],"")
            CleanFileName=datfiletoclean
            DirtyFileName=BaseFilePath +"dummyDirtyData.dat"
            print("will remove",RecordToRemove_dict[datfiletoclean] , "from",datfiletoclean)
            imageExtractor = DatScraper_tool_broken_onlyExtract_NEW.ImageExtractor(datfiletoclean)
            imageExtractor.clean(RecordToRemove_dict[datfiletoclean],cleanPath=CleanFileName,dirtyPath=DirtyFileName)
            os.remove(DirtyFileName)
            
    else:
        raise Exception("User declined to continue")
