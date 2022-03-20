
import _3DVisLabLib
import statistics
import shutil
import os

MaxLength_of_SN=9
DictOfFilterCats=dict()

BaseSNR_Folder = input("Please enter images folder: Default is C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\CloudOCR")
if len(BaseSNR_Folder)==0:
    BaseSNR_Folder = r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\CloudOCR"

OutputFolder=BaseSNR_Folder +"\\FilterOutput"
print("Creating output folder",OutputFolder)
_3DVisLabLib.DeleteFiles_RecreateFolder(OutputFolder)

FolderFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(BaseSNR_Folder)

Files_s39=[]
S39_and_image=dict()
CharacterSet=dict()


for CheckExt in FolderFiles:
    try:
        if ("s39") in CheckExt.split(".")[1].lower():
            Files_s39.append(CheckExt)
            S39_and_image[CheckExt]=None
    except:
        pass

if len(Files_s39)==0:
    print("no s39 files found in ",BaseSNR_Folder,"possibly because of no tracer files found in process")
    raise Exception("not handled yet")


UnicodeList=[]
SN_Length=[]
for S39File in Files_s39:
    print(S39File)
    S39File_sansFileExt=S39File.split("\\")[-1]
    S39File_sansFileExt=S39File_sansFileExt.split(".")[0]
    for Charac in S39File_sansFileExt:
        UnicodeOfChar=ord(Charac)
        UnicodeList.append(UnicodeOfChar)
    S39_and_image[S39File]=S39File_sansFileExt
    #save s39 and complimentary image - test for presence of image later
    #S39_and_image[S39File]=S39File.split(".")[-2]+".jpg"
std_d_Unicode=statistics.pstdev(UnicodeList)
mean_Unicode=statistics.mean(UnicodeList)


#set standard deviations to filter by
LowLevel_Unicode=mean_Unicode-(std_d_Unicode*2)
HighLevel_Unicode=mean_Unicode+(std_d_Unicode*2)

#_RemoveCHars_outwithUnicodeArea=[]
#run again - remove bad characters (not in unicode span)
for S39File in Files_s39:
    S39File_sansFileExt=S39File.split(".")[-2]
    S39File_sansFileExt=S39File_sansFileExt.split("\\")[-1]
    NewSNR_String=""
    for Charac in S39File_sansFileExt:
        UnicodeOfChar=ord(Charac)
        #if in same-ish area of unicode - dont filter out
        if UnicodeOfChar>LowLevel_Unicode and UnicodeOfChar<HighLevel_Unicode:
            NewSNR_String=NewSNR_String+Charac
        else:
            #dont add it
            if not "NotInUniCodeRange" in DictOfFilterCats:
                DictOfFilterCats["NotInUniCodeRange"]=0
            DictOfFilterCats["NotInUniCodeRange"]=DictOfFilterCats["NotInUniCodeRange"]+1
    #add length to corrected string
    SN_Length.append(len(NewSNR_String))
    print("cleaning SN using unicode block proximity:",NewSNR_String)
    S39_and_image[S39File]=NewSNR_String

std_d_Length=statistics.pstdev(SN_Length)
mean_Length=statistics.mean(SN_Length)



FilteredByLength=[]
#clean out by length
for S39File in S39_and_image:
    #S39File_sansFileExt=S39File.split(".")[-2]
    if len(S39_and_image[S39File])==MaxLength_of_SN:#magic number
        pass
    else:
        print("Not correct length:",S39_and_image[S39File])
        S39_and_image[S39File]=None
        if not "NotExactLength" in DictOfFilterCats:
            DictOfFilterCats["NotExactLength"]=0
        if S39_and_image[S39File] is None:
            DictOfFilterCats["NotExactLength"]=DictOfFilterCats["NotExactLength"]+1
        #FilteredByLength.append(S39File)


#save out and get metrics for filtered sn reads
Filtered=0
for S39File in S39_and_image:
    if S39_and_image[S39File] is None:
        Filtered=Filtered+1
    else:

        FileNameOnly=S39File.split("\\")[-1]
        FIleNameNoExt=FileNameOnly.split(".")[0]
        FilePathAndName_noExt=S39File.split(".")[0]
        #save out S39 file, renamed to repaired OCR read
        shutil.copy(S39File,OutputFolder + "\\" +S39_and_image[S39File] + ".S39" )
        #image file might not exist
        try:
            shutil.copy(FilePathAndName_noExt + ".jpg",OutputFolder + "\\" +S39_and_image[S39File] + ".jpg" )
        except:
            pass

print("Fail categories")
for FailCAt in DictOfFilterCats:
    print(FailCAt,DictOfFilterCats[FailCAt])
print("Filter Metric")
print(int(Filtered/len(S39_and_image)*100),"% filtered from SNR reads")

