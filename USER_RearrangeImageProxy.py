import DatScraper_tool_broken_onlyExtract_NEWER
import _3DVisLabLib
import json
import shutil
import os

InputPath=r"C:\Working\FindIMage_In_Dat\Output"
InputJson=InputPath+"\\TraceImg_to_DatRecord.json" 
OutputFolder=r"C:\Working\FindIMage_In_Dat\Output"

#what folder has user placed in images (dat records) to merge
TestMergingFolder=r"C:\Working\FindIMage_In_Dat\Output\TestMerge"

#load json file which links image proxies back to their dat files and record
RecordOffset=-1#fudge factor
with open(InputJson) as json_file:
    ImgVDatFile_andRecord = json.load(json_file)

#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(TestMergingFolder)
#Get pickle file - warning will just take first one
ListAllJpg_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".jpg"])

#should check here that user hasnt duplicated images with copy and paste (or has nested folders)
CheckDuplicates=dict()
for ImgProxy in ListAllJpg_files:
    imgfilename=ImgProxy.split("\\")[-1]
    if not imgfilename in CheckDuplicates:
        CheckDuplicates[imgfilename]=ImgProxy
    else:
        print("WARNING duplicate image found (will be ignored)\n", ImgProxy,"\n",CheckDuplicates[imgfilename],"please do not used nested folders in folder to be merged")

#convert back to list
ListAllJpg_files=list(CheckDuplicates.keys())

RecordToRemove_dict=dict()
RecordToKeep_dict=dict()
for imgfilename in ListAllJpg_files:
    imgfilename=imgfilename.split("\\")[-1]
    FoundImage=False
    for datkeys in ImgVDatFile_andRecord:
        if imgfilename in datkeys:
            FoundImage=True
            DatFilename=ImgVDatFile_andRecord[datkeys][0]
            DatRecord=ImgVDatFile_andRecord[datkeys][1]
            if DatFilename in RecordToKeep_dict:
                #check here that it hasnt already been added - this will probably break the code
                Ok2Add=True
                for RecordNo in RecordToKeep_dict[DatFilename]:
                    if int(DatRecord)+RecordOffset==RecordNo:
                        print("ERROR: Record already exists",int(DatRecord)+RecordOffset,DatFilename)
                        Ok2Add=False
                        break
                if Ok2Add==True: RecordToKeep_dict[DatFilename].append(int(DatRecord)+RecordOffset)
            else:
                RecordToKeep_dict[DatFilename]=[int(DatRecord)+RecordOffset]
    if FoundImage==False:
        print("ERROR with file",imgfilename,"no match found in JSON - potentially desynchronised state")



list_dats=[]
for Indexer , datfiletoclean in enumerate(RecordToKeep_dict):
    OutputDat=TestMergingFolder + "\\" + str(Indexer) + ".dat"
    OutputUnusedDat=TestMergingFolder + "\\dummyDirtyData.dat"
    imageExtractor = DatScraper_tool_broken_onlyExtract_NEWER.ImageExtractor(datfiletoclean)
    imageExtractor.clean(RecordToKeep_dict[datfiletoclean],cleanPath=OutputUnusedDat,dirtyPath=OutputDat)
    os.remove(OutputUnusedDat)
    list_dats.append(OutputDat)

merger = DatScraper_tool_broken_onlyExtract_NEWER.ImageMerger()
merger.files = list_dats
merger.outputDirectory=TestMergingFolder + "\\"
merger.start()

for DatToDelete in list_dats:
    os.remove(DatToDelete)