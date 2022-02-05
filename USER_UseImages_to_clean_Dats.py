import DatScraper_broken_onlyExtract
import _3DVisLabLib
import json
import shutil

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
for imgfilename in ImgVDatFile_andRecord:
    if imgfilename in InputFiles:
        DatFilename=ImgVDatFile_andRecord[imgfilename][0]
        DatRecord=ImgVDatFile_andRecord[imgfilename][1]
        if DatFilename in RecordToRemove_dict:
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
#should have a dictionary with the missing images aligned with dat files and list
#we can now spin through the dictionary and pass the list to the dat scraper library

#WIP 
if (_3DVisLabLib.yesno("WIP - is this single record MM8 data and only record images you want remain in image proxy?")):
#if only one record per dat file have to use this!! IE MM8 data
    for datfiletoclean in RecordToKeep_dict:
        OriginalFile=datfiletoclean
        FileName=OriginalFile.split("\\")[-1]
        dst=OutputFolder +"\\" + FileName
        print("Copying",OriginalFile,"to",dst)
        shutil.copyfile(datfiletoclean, dst)

if (_3DVisLabLib.yesno("WIP - is this multiple record data and records to be excluded have been deleted from image proxy?")):
#otherwise use this if lots of records per dat
    for datfiletoclean in RecordToRemove_dict:
        print("will remove",RecordToRemove_dict[datfiletoclean] , "from",datfiletoclean)
        imageExtractor = DatScraper_broken_onlyExtract.ImageExtractor(datfiletoclean)
        imageExtractor.clean(RecordToRemove_dict[datfiletoclean])
