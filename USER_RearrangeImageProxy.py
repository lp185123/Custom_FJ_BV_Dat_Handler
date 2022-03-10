import DatScraper_tool_broken_onlyExtract_NEWER
import _3DVisLabLib
import json
import shutil
import os
from math import floor

InputPath=r"C:\Working\FindIMage_In_Dat\Output"
InputJson=InputPath+"\\TraceImg_to_DatRecord.json" 
OutputFolder=r"C:\Working\FindIMage_In_Dat\Output"
MergedRecordSize=300

#what folder has user placed in images (dat records) to merge
TestMergingFolder=r"C:\Working\FindIMage_In_Dat\Output\ReorganiseOrMerge"

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

#check if user has doubled up images - this might very well break if the user has copied the images twice
#example
#File13_Image_47_2a_aa_190116_120731.jpg
#File13_Image_48_2a_aa_190116_120731 - Copy.jpg
#File13_Image_48_2a_aa_190116_120731 - Copy - Copy.jpg
#File13_Image_47_2a_aa_190116_120731 (2).jpg<----------this one harder to detect

List_WindowsCopy_Suffix=[" - Copy", " ("]
#list of duplicate list filenames we can add to safelist
Duplicates_list=[]
for ImgProxy in ListAllJpg_files:
    imgfilename=ImgProxy.split("\\")[-1]
    #if List_WindowsCopy_Suffix in imgfilename:
    if any(ele in imgfilename for ele in List_WindowsCopy_Suffix):
        for testsubstring in List_WindowsCopy_Suffix:
            IndexCopySubString=(ImgProxy.find(testsubstring))
            #-1 is failure to find index
            if IndexCopySubString!=-1:
                break
        if IndexCopySubString==-1:
            raise Exception("Could not find copy substring from list",List_WindowsCopy_Suffix,"in",ImgProxy)
        #find original file - have to work off assumption that windows created the "copy" substring
        OriginalFileName=ImgProxy[0:IndexCopySubString] + ".jpg"
        OriginalFileName in ImgVDatFile_andRecord
        #link back to image vs dat dictionary - but have to match with filename only - not filepath as 
        #user will have moved the file
        imgfilename=OriginalFileName.split("\\")[-1]
        FoundImage=False
        for datkeys in ImgVDatFile_andRecord:
            if imgfilename in datkeys:
                FoundImage=True
                DatFilename=ImgVDatFile_andRecord[datkeys][0]
                DatRecord=ImgVDatFile_andRecord[datkeys][1]
                #now we have found the identical note - we can add this copy filename to the dictionary linking dats to images
                ImgVDatFile_andRecord[ImgProxy]=(DatFilename,DatRecord)
                #add both files to safe list (original file and copied file) - this will needed later on when we tie up dats & records to check for logic errors
                Duplicates_list.append(ImgProxy.split("\\")[-1])
                Duplicates_list.append(datkeys.split("\\")[-1])
                break
        if FoundImage==False:
            print("ERROR with file",imgfilename,"no match found in JSON - potentially desynchronised state")

#convert back to list
ListAllJpg_files=list(CheckDuplicates.keys())
RecordToRemove_dict=dict()
RecordToKeep_dict=dict()
for imgfilename in ListAllJpg_files:
    #imgfilename=imgfilename.split("\\")[-1]
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
                        #if filename isnt on safelist (list of duplicates discovered earlier) we have a logic error 
                        if not imgfilename in Duplicates_list:
                            print("ERROR: Record already exists",int(DatRecord)+RecordOffset,DatFilename)
                            Ok2Add=False
                            break
                if Ok2Add==True: RecordToKeep_dict[DatFilename].append(int(DatRecord)+RecordOffset)
            else:
                RecordToKeep_dict[DatFilename]=[int(DatRecord)+RecordOffset]
    if FoundImage==False:
        print("ERROR with file",imgfilename,"no match found in JSON - potentially desynchronised state")



#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(TestMergingFolder)
#Get pickle file - warning will just take first one
ListAllDat_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".dat"])

#clean up old dats left in merge folder
for DatToDelete in ListAllDat_files:
    os.remove(DatToDelete)


list_dats_Cleaup=[]
for Indexer , datfiletoclean in enumerate(RecordToKeep_dict):
    OutputDat=TestMergingFolder + "\\" + str(Indexer) + ".dat"
    OutputUnusedDat=TestMergingFolder + "\\dummyDirtyData.dat"
    imageExtractor = DatScraper_tool_broken_onlyExtract_NEWER.ImageExtractor(datfiletoclean)
    print(datfiletoclean,RecordToKeep_dict[datfiletoclean])
    imageExtractor.clean(RecordToKeep_dict[datfiletoclean],cleanPath=OutputUnusedDat,dirtyPath=OutputDat)
    os.remove(OutputUnusedDat)
    list_dats_Cleaup.append(OutputDat)

#merge dats together in folder
merger = DatScraper_tool_broken_onlyExtract_NEWER.ImageMerger()
merger.files = list_dats_Cleaup
merger.outputDirectory=TestMergingFolder + "\\"
RecordCount=len(InputFiles)
Chunks=floor(RecordCount/MergedRecordSize)
RemainderChunk=RecordCount%MergedRecordSize
merger.arrayOfNoteCounts=([MergedRecordSize]*Chunks)+[RemainderChunk]#build string for merging library
if sum(merger.arrayOfNoteCounts)!=RecordCount:
    print("ERROR with merger.arrayOfNoteCounts logic, defaulting to 100 size chunks for merged file")
    merger.arrayOfNoteCounts=([300]*100)#give the user something to work with - unlikely to have such a huge dat file
merger.start()
#clean up working dats
for DatToDelete in list_dats_Cleaup:
    os.remove(DatToDelete)

#get final dat and rename it to folder
#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(TestMergingFolder)
#Get pickle file - warning will just take first one
ListAllDat_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".dat"])
#should only be one in the folder

