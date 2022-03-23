
import _3DVisLabLib
import statistics
import shutil
import os
import pickle
import random

MaxLength_of_SN=9
MinConfidence=0.8#from 0.0 to 1.0
LanguageFilter='[language_code: "bn"\n]'
RareChars_Unfiltered=5#we dont want to filter out rare characters - so save them in a seperate folder 

DictOfFilterCats=dict()

BaseSNR_Folder = input("Please enter images folder: Default is C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\CloudOCR")
if len(BaseSNR_Folder)==0:
    BaseSNR_Folder = r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\CloudOCR"

OutputFolder=BaseSNR_Folder +"\\FilterOutput"
BestMatches=BaseSNR_Folder +"\\FilterOutput\\Topmatches\\"
BestCharMatchesOnly=BaseSNR_Folder +"\\FilterOutput\\BestCharMatchesOnly\\"
RareCharactersNoFilter=BaseSNR_Folder +"\\FilterOutput\\RareCharactersNoFilter\\"
print("Creating output folder",OutputFolder)
_3DVisLabLib.DeleteFiles_RecreateFolder(OutputFolder)
_3DVisLabLib.DeleteFiles_RecreateFolder(BestMatches)
_3DVisLabLib.DeleteFiles_RecreateFolder(BestCharMatchesOnly)
_3DVisLabLib.DeleteFiles_RecreateFolder(RareCharactersNoFilter)
FolderFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(BaseSNR_Folder)

#randomise files
Randdict=dict()
for item in FolderFiles:
    Randdict[item]=item
#rebuild files
FolderFiles=[]
while len(Randdict)>0:
    Randomchoice=random.choice(list(Randdict))
    FolderFiles.append(Randomchoice)
    del Randdict[Randomchoice]



Files_s39=[]
S39_and_image=dict()
CharacterSet=dict()

CharacAnalysis_database=None
try:
    with open(BaseSNR_Folder+"\\CharAnalysis.ca", 'rb') as pickle_file:
        CharacAnalysis_database = pickle.load(pickle_file)
except:
    print("Could not load Character analysis database",BaseSNR_Folder+"\\CharAnalysis.ca")
    print("this error can be ignored")


#load in character set - might be pre-filtered already so make sure you dont use unicode block area sigma filtering if it has - could cut off characters!
CharacterDictionary=None
try:
    CharacterDictionary = _3DVisLabLib.JSON_Open(BaseSNR_Folder+"\\FoundCharDic.fc")
except:
    print("Could not load Character analysis database",BaseSNR_Folder+"\\FoundCharDic.fc")
    print("this error can be ignored")



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


#roll through the dictionary which has the jpg/s39 filename sans extension as a key, then details of the OCR answer (unfiltered)

#firstly create folder of perfect matches
#BestMatches
for trackname in CharacterDictionary:
    TotalGoodChars=0
    if len(CharacterDictionary[trackname]['CHARACTER'])==MaxLength_of_SN:
    
        for Indexer,Charac_confidence in enumerate(CharacterDictionary[trackname]['CONFIDENCE']):
            if LanguageFilter in CharacterDictionary[trackname]['LANGUAGE'][Indexer]:
                if Charac_confidence>=MinConfidence:
                    TotalGoodChars=TotalGoodChars+1
            if TotalGoodChars==MaxLength_of_SN:
                #if files exist, copy to output folder
                if os.path.exists(BaseSNR_Folder + "\\" + trackname + ".jpg"):
                    shutil.copy(BaseSNR_Folder + "\\" + trackname + ".jpg",BestMatches+ "\\" + trackname + ".jpg")
                #if files exist, copy to output folder
                if os.path.exists(BaseSNR_Folder + "\\" + trackname + ".s39"):
                    shutil.copy(BaseSNR_Folder + "\\" + trackname + ".s39",BestMatches+ "\\" + trackname + ".s39")



#BestCharMatchesOnly
for Index, trackname in enumerate(CharacterDictionary):
    TotalGoodChars=0
    if len(CharacterDictionary[trackname]['CHARACTER'])==MaxLength_of_SN:
        for Indexer,Charac_confidence in enumerate(CharacterDictionary[trackname]['CONFIDENCE']):
            if LanguageFilter in CharacterDictionary[trackname]['LANGUAGE'][Indexer]:
                if Charac_confidence>=MinConfidence:
                    TotalGoodChars=TotalGoodChars+1
            
            if TotalGoodChars==2:
                #if files exist, copy to output folder
                if os.path.exists(BaseSNR_Folder + "\\" + trackname + ".jpg"):
                    shutil.copy(BaseSNR_Folder + "\\" + trackname + ".jpg",BestCharMatchesOnly+ "\\" + trackname + ".jpg")
                #if files exist, copy to output folder
                if os.path.exists(BaseSNR_Folder + "\\" + trackname + ".s39"):
                    shutil.copy(BaseSNR_Folder + "\\" + trackname + ".s39",BestCharMatchesOnly+ "\\" + trackname + ".s39")
            if Index==2:
                continue




#get frequency of characters
DictCharCount=dict()
for Index, trackname in enumerate(CharacterDictionary):
    TotalGoodChars=0
    #roll through each character
    for Indexer,Charac_tocheck in enumerate(CharacterDictionary[trackname]['CHARACTER']):
        #ensure correct language
        if LanguageFilter in CharacterDictionary[trackname]['LANGUAGE'][Indexer]:
            if Charac_tocheck not in DictCharCount:
                DictCharCount[Charac_tocheck]=[]
            DictCharCount[Charac_tocheck].append(trackname)

#Restrict characters frequency so we can roll through (randomised) list of input files
#and create limited examples of each
RareChars_dict=()
for Index, CharSymbol in enumerate(DictCharCount):
    try:
        if len(DictCharCount[CharSymbol])<=RareChars_Unfiltered:
            print("rare unfiltered character",CharSymbol)
            if os.path.exists(BaseSNR_Folder + "\\" + trackname + ".jpg"):
                shutil.copy(BaseSNR_Folder + "\\" + trackname + ".jpg",RareCharactersNoFilter+ "\\" + trackname + ".jpg")
            #if files exist, copy to output folder
            if os.path.exists(BaseSNR_Folder + "\\" + trackname + ".s39"):
                shutil.copy(BaseSNR_Folder + "\\" + trackname + ".s39",RareCharactersNoFilter+ "\\" + trackname + ".s39")
    except:
        print("bad symbol, skipping - cannot print in case causes further error")





#first filter - find unicode 
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

