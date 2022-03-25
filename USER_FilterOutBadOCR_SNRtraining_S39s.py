
import _3DVisLabLib
import statistics
import shutil
import os
import pickle
import random

#expect a folder with txt files containing ocr answers, and files which can trace the text files
#back to images/s39 images and other jsons with data for each character read (confidence, language etc)
MaxLength_of_SN=9
MinConfidence=0.6#from 0.0 to 1.0
LanguageFilter='[language_code: "bn"\n]'
RareChars_Unfiltered=5#we dont want to filter out rare characters - so save them in a seperate folder 
InstancesOfChar=10 # maximum instances of character to create a manageable subset for training
FirstCharsOnly=2

DictOfFilterCats=dict()

BaseSNR_Folder = input("Please enter images folder: Default is C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\CloudOCR")
if len(BaseSNR_Folder)==0:
    BaseSNR_Folder = r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\CloudOCR"

OutputFolder=BaseSNR_Folder +"\\FilterOutput"
BestMatches=BaseSNR_Folder +"\\FilterOutput\\Topmatches\\"
BestCharMatchesOnly=BaseSNR_Folder +"\\FilterOutput\\BestCharMatchesOnly\\"
RareCharactersNoFilter=BaseSNR_Folder +"\\FilterOutput\\RareCharactersNoFilter\\"
TrainingSetFolder=BaseSNR_Folder +"\\FilterOutput\\TrainingSets\\" 
print("Creating output folder",OutputFolder)

TryAgain=True
while TryAgain==True:
    try:
        _3DVisLabLib.DeleteFiles_RecreateFolder(OutputFolder)
        _3DVisLabLib.DeleteFiles_RecreateFolder(BestMatches)
        _3DVisLabLib.DeleteFiles_RecreateFolder(BestCharMatchesOnly)
        _3DVisLabLib.DeleteFiles_RecreateFolder(RareCharactersNoFilter)
        _3DVisLabLib.DeleteFiles_RecreateFolder(TrainingSetFolder)
        TryAgain=False
    except Exception as e:
        print(e)
        TryAgain=_3DVisLabLib.yesno("Error trying to delete output folders - try again?")


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


#randomise char dictionary as well
CharacterDictionary_rand=dict()
while len(CharacterDictionary)>0:
    RandomChoice_item=random.choice(list(CharacterDictionary))
    CharacterDictionary_rand[RandomChoice_item]=CharacterDictionary[RandomChoice_item]
    del CharacterDictionary[RandomChoice_item]
CharacterDictionary=CharacterDictionary_rand

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

#firstly create folder of perfect matches - with mindset to use straight into the SNR tool
#every char will have to be greater than the confidence - so a situation may arise whereby a low-confidence number 
#(of which we have many examples) may result in filtering out a low-frequency alpha character - so use the second
#segment of logic to keep all rare characters no matter what confidnece level is associated
BestMatchAllChars_CharDictionary=dict()
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
                #add all characters to the dictionary
                for CharacToAdd in CharacterDictionary[trackname]['CHARACTER']:
                    if not CharacToAdd in BestMatchAllChars_CharDictionary:
                        BestMatchAllChars_CharDictionary[CharacToAdd]=[]
                    BestMatchAllChars_CharDictionary[CharacToAdd].append(trackname)



#BestCharMatchesOnly - so we dont filter out characters for more frequent numerals
BestCharactersOnly_CharDictionary=dict()
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
                #add all characters to the dictionary
                for CharacToAdd in CharacterDictionary[trackname]['CHARACTER']:
                    if not CharacToAdd in BestCharactersOnly_CharDictionary:
                        BestCharactersOnly_CharDictionary[CharacToAdd]=[]
                    BestCharactersOnly_CharDictionary[CharacToAdd].append(trackname)

            
            if Index==2:
                continue

    




#get frequency of all characters
DictCharCount_Unfiltered=dict()
for Index, trackname in enumerate(CharacterDictionary):
    TotalGoodChars=0
    #roll through each character
    for Indexer,Charac_tocheck in enumerate(CharacterDictionary[trackname]['CHARACTER']):
        #ensure correct language
        if LanguageFilter in CharacterDictionary[trackname]['LANGUAGE'][Indexer]:
            if Charac_tocheck not in DictCharCount_Unfiltered:
                DictCharCount_Unfiltered[Charac_tocheck]=[]
            DictCharCount_Unfiltered[Charac_tocheck].append(trackname)

#Restrict characters frequency so we can roll through (randomised) list of input files
#and create limited examples of each
RareChars_dict=dict()
for Index, CharSymbol in enumerate(DictCharCount_Unfiltered):
    try:
        if len(DictCharCount_Unfiltered[CharSymbol])<=RareChars_Unfiltered:
            print("rare unfiltered character",CharSymbol)
            CharacterFolder=RareCharactersNoFilter  + str(len(DictCharCount_Unfiltered[CharSymbol])) + "___"+ str(CharSymbol)
            _3DVisLabLib.DeleteFiles_RecreateFolder(CharacterFolder)
            for RandomCHoiceFilename in DictCharCount_Unfiltered[CharSymbol]:
                if os.path.exists(BaseSNR_Folder + "\\" + RandomCHoiceFilename + ".jpg"):
                    shutil.copy(BaseSNR_Folder + "\\" + RandomCHoiceFilename + ".jpg",CharacterFolder+ "\\" + RandomCHoiceFilename + ".jpg")
                #if files exist, copy to output folder
                if os.path.exists(BaseSNR_Folder + "\\" + RandomCHoiceFilename + ".s39"):
                    shutil.copy(BaseSNR_Folder + "\\" + RandomCHoiceFilename + ".s39",CharacterFolder+ "\\" + RandomCHoiceFilename + ".s39")
                #add all characters to the dictionary
                if not CharSymbol in RareChars_dict:
                    RareChars_dict[CharSymbol]=[]
                RareChars_dict[CharSymbol].append(RandomCHoiceFilename)


    except:
        print("bad symbol, skipping - cannot print in case causes further error")



#we want to create a manageable training set
#run through the character set found for every read (without filtering)
#then create folders for each character using a combination of the other dictionaryies
#be mindful of accidently missing characters - for example, a symbol mayb be very frequent but low-confidence
#so we might add rare characters, add filtered characters - then miss the low confidence ones as they will fall through
#the cracks
CharForTraining_Dict=dict()
for CharForTraining in DictCharCount_Unfiltered:
    if CharForTraining=="ন":
        plop=1
    if CharForTraining in CharForTraining_Dict: raise Exception("TrainSNRFilter Logic error, should not be able to add char twice")
    CharForTraining_Dict[CharForTraining]={"PROVENANCE": [],"FILE":[]}#create empty list to populate with files associated with this character
    if CharForTraining in BestMatchAllChars_CharDictionary:
        for FileName in BestMatchAllChars_CharDictionary[CharForTraining]:
            if len(CharForTraining_Dict[CharForTraining]["FILE"])==InstancesOfChar:
                break
            if FileName not in CharForTraining_Dict[CharForTraining]["FILE"]:
                CharForTraining_Dict[CharForTraining]["FILE"].append(FileName)
                CharForTraining_Dict[CharForTraining]["PROVENANCE"].append("3")#Arbitrary rating, higher is better
                #if we have enough insgtances of the character we can move to next
                
        #if we still don't have enough instances of character - the next best place is the dictionary
        #that only tests confidence for the alpha characters - so in this case we could have some noise
    if CharForTraining in BestCharactersOnly_CharDictionary:
        for FileName in BestCharactersOnly_CharDictionary[CharForTraining]:
            if len(CharForTraining_Dict[CharForTraining]["FILE"])==InstancesOfChar:
                break
            if FileName not in CharForTraining_Dict[CharForTraining]["FILE"]:
                CharForTraining_Dict[CharForTraining]["FILE"].append(FileName)
                CharForTraining_Dict[CharForTraining]["PROVENANCE"].append("2")#Arbitrary rating, higher is better
                #if we have enough insgtances of the character we can move to next
                
    
    if CharForTraining in RareChars_dict:
        for FileName in RareChars_dict[CharForTraining]:
            if len(CharForTraining_Dict[CharForTraining]["FILE"])==InstancesOfChar:
                break
            if FileName not in CharForTraining_Dict[CharForTraining]["FILE"]:
                CharForTraining_Dict[CharForTraining]["FILE"].append(FileName)
                CharForTraining_Dict[CharForTraining]["PROVENANCE"].append("1")#Arbitrary rating, higher is better
            #if we have enough insgtances of the character we can move to next
      

    #worst case scenario - have to take characters from the unfiltered set
    if CharForTraining in DictCharCount_Unfiltered:
        for FileName in DictCharCount_Unfiltered[CharForTraining]:
            if len(CharForTraining_Dict[CharForTraining]["FILE"])==InstancesOfChar:
                break
            if FileName not in CharForTraining_Dict[CharForTraining]["FILE"]:
                CharForTraining_Dict[CharForTraining]["FILE"].append(FileName)
                CharForTraining_Dict[CharForTraining]["PROVENANCE"].append("0")#Arbitrary rating, higher is better
        
    
    Subfolder=TrainingSetFolder+ CharForTraining + "__" + "".join(CharForTraining_Dict[CharForTraining]["PROVENANCE"])
    
    #create folder for each character instance
    try:#only create subfolder if character subset isnt empty
        if len(CharForTraining_Dict[CharForTraining]["FILE"])>0:
            _3DVisLabLib.DeleteFiles_RecreateFolder(Subfolder)
            #populate new folder with the files linked to this character
            #if files exist, copy to output folder
            for Filename in CharForTraining_Dict[CharForTraining]["FILE"]:
                #file should be in this format : '[826]চঘ১২৭০০৫৪' , convert to 'চঘ১২৭০০৫৪'
                Filename_RemoveBrackets=Filename.split("]")[-1]
                if os.path.exists(BaseSNR_Folder + "\\" + Filename + ".jpg"):
                    shutil.copy(BaseSNR_Folder + "\\" + Filename + ".jpg",Subfolder+ "\\" + Filename_RemoveBrackets + ".jpg")
                else:
                    print("Could not find file",BaseSNR_Folder + "\\" + Filename + ".jpg")

                #if files exist, copy to output folder
                if os.path.exists(BaseSNR_Folder + "\\" + Filename + ".s39"):
                    shutil.copy(BaseSNR_Folder + "\\" + Filename + ".s39",Subfolder+ "\\" + Filename_RemoveBrackets + ".s39")
                else:
                    print("Could not find file",BaseSNR_Folder + "\\" + Filename + ".s39")

    except:
        print("Error creating folder",TrainingSetFolder +  Subfolder,"\nskipping this character instance")
    
print("----Filter Info----")
print("MaxLength_of_SN",MaxLength_of_SN)
print("MinConfidence",MinConfidence)
print("LanguageFilter",MaxLength_of_SN)
print("RareChars_Unfiltered",RareChars_Unfiltered)
print("InstancesOfChar",InstancesOfChar)
print("FirstCharsOnly",FirstCharsOnly)
print("Training folders at",BaseSNR_Folder)
print("key 3: Matches from full SN confidence & length filter")
print("key 2: Matches from first characters filtered by confidence, rest of SN unfiltered. All filtered by length")
print("key 3: Rare characters, SN no length or confidence filter")

#create training sets now for each character, and anther folder with everything by
#rolling through the dictionary with the best examples we have 

okokok

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

