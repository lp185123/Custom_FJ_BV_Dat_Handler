"""spin through simulation results and collate all 4A and 4B scoring"""
import os
import win32clipboard
import copy
import _3DVisLabLib
class InfoStrings():
    List_known_4Asuffix=["_Genuine"]
    List_known_4Bsuffix=["_Damage"]#["Clearly","_Damage"]
    list_known_countries=["Belarus","Brazil","Czech","Hungary","Malaysia","Poland","Mexico","Russia","Turkey","UK"]
    list_known_GenerationTypes=["Minimum","Standard"]
    list_FinalCategories=["CIRCULATIONFIT","COUNTERFEIT","GEN","NEW","TELLERFIT","UNFIT","UNKNOWN","UNKNOWN2"]

def yesno(question):
    """Simple Yes/No Function."""
    prompt = f'{question} ? (y/n): '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n']:
        print(f'{ans} is invalid, please try again...')
        return yesno(question)
    if ans == 'y':
        return True
    return False

def copyToClipboard(text):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
    win32clipboard.CloseClipboard()


def GetAllFilesInFolder_Recursive(root):
    ListOfFiles=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            FullpathOfFile=(os.path.join(path, name))
            ListOfFiles.append(FullpathOfFile)
    return ListOfFiles

def GetList_Of_ImagesInList(ListOfFiles, ImageTypes=(".jPg", ".Png",".gif")):
    
    Image_FileNames=[]
    
    #list comprehension [function-of-item for item in some-list
    ImageTypes_ForceLower=[x.lower()  for x in ImageTypes]
    ImageTypes_ForceLower_Tuple=tuple(ImageTypes_ForceLower)
    
    for filename in ListOfFiles:
    #if a known image filetype - copy file
        if str.endswith(str.lower(filename),ImageTypes_ForceLower_Tuple):
            Image_FileNames.append(filename)
    
    return Image_FileNames


class SingleSimRes_Breakdown():
    def __init__(self):
        self.TotalNotes=None
        self.Total4A_notes=None
        self.Total4B_notes=None
        self.Total_NotCat4_notes=None
        self.DatFileUsed=None
        self.NameOfSimFile=None
        
        self.TotalNotes_PC=None
        self.Total4A_notes_PC=None
        self.Total4B_notes_PC=None
        self.Total_NotCat4_notes_PC=None

        self.Country=None
        self.GenerationType=None
        
        self.FinalCat=None
        self.AttemptName=None
        self.DictKey=None
        self.FinalCatDictKey=None

    def GetPercentages(self):
        self.TotalNotes_PC=self.GetPercentageOfCat(self.TotalNotes)
        self.Total4A_notes_PC=self.GetPercentageOfCat(self.Total4A_notes)
        self.Total4B_notes_PC=self.GetPercentageOfCat(self.Total4B_notes)
        self.Total_NotCat4_notes_PC=self.GetPercentageOfCat(self.Total_NotCat4_notes)

    def GetPercentageOfCat(self,InputNumber):
        return round(((int(InputNumber)/int(self.TotalNotes))*100),2)

if __name__ == "__main__":
    #user input
    OutputFOlder=r"E:\NCR\TEMP_SPRINT_OUTPUT\\"
    _3DVisLabLib.yesno("press y to delete output folder :  " + OutputFOlder)
    _3DVisLabLib.DeleteFiles_RecreateFolder(OutputFOlder)
    #ResultFolder = input("Please enter folder with sim results:")
    ResultFolder=r"C:\Users\LP185123\OneDrive - NCR Corporation\Desktop\SR_GEN_Sprint"
    input("analysing: " + ResultFolder)
    AllFiles=GetAllFilesInFolder_Recursive(ResultFolder)
    #simulation results end with this suffix
    TxtFiles=GetList_Of_ImagesInList(AllFiles,[".dat.txt"])
    if TxtFiles==0:
        raise Exception("No valid simulation results found in" + ResultFolder)
    
    #dict to hold filename and performance
    TxtFile_V_Result=dict()
    #list of totals
    List_Totalpcs=[]
    list_datnames=[]
    #try building string - use stringbuilder if this gets too big
    Str_Totalpcs=""
    #populate skipped files
    SkippedFile=[]
    #collate all results for test
    All_results_Dictionary=dict()
    #roll through files
    for Item in TxtFiles:
        if not "Sim_" in Item:
            SkippedFile.append(Item)
            continue

        #get final category - assume is in correct folder structure
        FinalCAt=None
        AttemptName=None
        for FinCatName in InfoStrings.list_FinalCategories:
            if FinCatName.lower() in Item.lower():
                if FinCatName.lower() in Item.split("\\")[-3].lower():
                    FinalCAt=FinCatName
                    AttemptName=Item.split("\\")[-2]
        if FinalCAt==None:
            continue
            pass
            #raise Exception("FinalCAt name not found or using wrong root folder" + Item)


        #ignore weird chars 
        my_file = open(Item,   errors="ignore")
        content = my_file.read()
        Delimited=content.split()
        #find first instance of total - warning! if this appears in the filename this might break
        TotalString=Delimited.index("Total")
        FIleNameText=Delimited.index("Statistics")
        Statistics_Header=Delimited.index('Information')



        #get block of statistics
        StatisticsBlock=[]
        for Singleline in range (Statistics_Header+2,TotalString-1):
            StatisticsBlock.append(Delimited[Singleline])
        #go through statistics block and try to isolate each category
        #will crop the first bit encoded category but we dont need that at the moment
        CategoryBreakdown_dict=dict()
        CatCount=0
        for OuterIndex in range(0,len(StatisticsBlock)):
            if (StatisticsBlock[OuterIndex]=="x"):#this hex marker is consistent per category breakdown element
                CatCount=CatCount+1
                CategoryBreakdown_dict[CatCount]=[]
                for InnerIndex in range(OuterIndex+1,len(StatisticsBlock)):#now find next x
                    CategoryBreakdown_dict[CatCount].append(StatisticsBlock[InnerIndex])
                    if (StatisticsBlock[InnerIndex]=="x") :#next category
                        OuterIndex=InnerIndex+1
                        break

        #use string matching to find category 4B
        Categories4B_dict=dict()
        for Cats in CategoryBreakdown_dict:
            for Line in CategoryBreakdown_dict[Cats]:
                for Cat4BTestString in InfoStrings.List_known_4Bsuffix:
                    if Cat4BTestString in Line:
                        Categories4B_dict[Cats]=CategoryBreakdown_dict[Cats]

        #get total for category 4B
        _4B_Total=0
        for _4B_Instance in Categories4B_dict:
            _4B_Total=_4B_Total+int(Categories4B_dict[_4B_Instance][0])

        FIleName_Dat=' '.join(Delimited[0:FIleNameText-1])
        
                #if not ".dat" in FIleName_Dat.lower():
                    #FIleName_Dat=Delimited[FIleNameText+1]=Delimited[FIleNameText+2]+Delimited[FIleNameText+3]
        #get ratio of 4A versus all notes
        _4A_vs_AllNotes=Delimited[TotalString+3]
        _4A_vs_AllNotes=_4A_vs_AllNotes.split("/")
        _4A_Total=_4A_vs_AllNotes[0].replace("(","")
        #prior knowledge for position of these values
        list_datnames.append(FIleName_Dat)

        _4A_TotalPC=Delimited[TotalString+2]
        Total_Notes=Delimited[TotalString+1]
        #TxtFile_V_Result[Item]=(_4A_TotalPC,Total_Notes,FIleName_Dat,Item)
        List_Totalpcs.append(_4A_TotalPC)
        Str_Totalpcs=Str_Totalpcs + str(_4A_TotalPC) + " "
        #print(TxtFile_V_Result[Item])

        #get all other categories - warning; might be circular logic if we are using this number later to check count
        Total_not_Cat4=int(Total_Notes)-int(_4A_Total)-int(_4B_Total)

        #self checks
        totalNote_check=0
        for Cats in CategoryBreakdown_dict:
            #get # of notes for each category
            totalNote_check=totalNote_check+int(CategoryBreakdown_dict[Cats][0])




        #get country - assume is in correct folder structure
        Country=None
        for CountryName in InfoStrings.list_known_countries:
            if CountryName.lower() in Item.lower():
                if CountryName.lower() in Item.split("\\")[-4].lower():
                    Country=CountryName
        if Country==None:
            raise Exception("Country name not found or using wrong root folder" + Item)


        #get type of generation
        GenerationType=None
        for GenName in InfoStrings.list_known_GenerationTypes:
            if GenName.lower() in Item.lower():
                if GenName.lower() in Item.split("\\")[-5].lower():
                    GenerationType=GenName
        if GenerationType==None:
            raise Exception("GenerationType name not found or using wrong root folder" + Item)
        
        #build info object
        ResultInfo_singleFile=SingleSimRes_Breakdown()
        ResultInfo_singleFile.TotalNotes=int(Total_Notes)
        ResultInfo_singleFile.Total_NotCat4_notes=int(Total_not_Cat4)
        ResultInfo_singleFile.Total4A_notes=int(_4A_Total)
        ResultInfo_singleFile.Total4B_notes=int(_4B_Total)
        ResultInfo_singleFile.DatFileUsed=FIleName_Dat
        ResultInfo_singleFile.NameOfSimFile=Item
        ResultInfo_singleFile.GetPercentages()
        ResultInfo_singleFile.Country=Country
        ResultInfo_singleFile.GenerationType=GenerationType
        ResultInfo_singleFile.AttemptName=AttemptName
        ResultInfo_singleFile.FinalCat=FinalCAt
        ResultInfo_singleFile.DictKey=(ResultInfo_singleFile.Country+ "_"+ResultInfo_singleFile.GenerationType+"_"+ResultInfo_singleFile.FinalCat+"_"+ResultInfo_singleFile.AttemptName)
        ResultInfo_singleFile.FinalCatDictKey=(ResultInfo_singleFile.Country+ "_"+ResultInfo_singleFile.GenerationType+"_"+ResultInfo_singleFile.FinalCat)
        #print("\n\n")
        #print(ResultInfo_singleFile.Total4A_notes_PC,ResultInfo_singleFile.Total4B_notes_PC,ResultInfo_singleFile.Total_NotCat4_notes_PC)
        #print(vars(ResultInfo_singleFile))

        print("Checking",Item)
        if Total_Notes!=str(totalNote_check):
            DoNothing=True
            
            raise Exception("ERROR self check total VS cat breakdown")
        #potentially circular test
        if 0!=int(Total_Notes)-int(_4B_Total)-int(_4A_Total)-int(Total_not_Cat4):
            DoNothing=True
            raise Exception("ERROR self check total VS cat breakdown")
        if not ".dat" in FIleName_Dat.lower():
            DoNothing=True
            raise Exception("ERROR self check not .dat in dat filename")


        #load into dictionary
        
        if not ResultInfo_singleFile.DictKey in All_results_Dictionary:
            All_results_Dictionary[ResultInfo_singleFile.DictKey]=[]
        All_results_Dictionary[ResultInfo_singleFile.DictKey].append(copy.deepcopy(ResultInfo_singleFile))

        
    #check people not doubled up
    #for ResultItem in All_results_Dictionary:
    #    All_results_Dictionary[]


    for ResultItem in All_results_Dictionary:
        #each resultitem will be of this format: 
        # COuntry - gentype, final cat, attempt name: eg
        #Malaysia_Minimum_GEN_attempt 1 - Automatic

        #create companion text file for someone to verify results
        #Companion_file=Item.lower().replace(".dat.txt",".dat.chk")
        #datfileused, simtestpath, notcat4a, cat4a, cat4b
        Total_4a=0
        Total_4b=0
        Total_Not4aNot4B=0
        TotalNotes=0
        Delimiter=","
        f = open(OutputFOlder + str(ResultItem) +".txt", "a")
        ResultInfo_singleFile=All_results_Dictionary[ResultItem][0]
        ResultInfo_singleFile_testID=ResultInfo_singleFile.Country+ Delimiter+ResultInfo_singleFile.GenerationType+Delimiter+ResultInfo_singleFile.FinalCat+Delimiter+ResultInfo_singleFile.AttemptName
        f.write(ResultInfo_singleFile_testID +"\n")
        for Indv_SimResultFIle in All_results_Dictionary[ResultItem]:
            Total_4a=Total_4a+Indv_SimResultFIle.Total4A_notes
            Total_4b=Total_4b+Indv_SimResultFIle.Total4B_notes
            Total_Not4aNot4B=Total_Not4aNot4B+Indv_SimResultFIle.Total_NotCat4_notes
            TotalNotes=TotalNotes+Indv_SimResultFIle.TotalNotes
            TestID=Indv_SimResultFIle.Country+ Delimiter+Indv_SimResultFIle.GenerationType+Delimiter+Indv_SimResultFIle.FinalCat+Delimiter+Indv_SimResultFIle.AttemptName
            if TestID!=ResultInfo_singleFile_testID:
                raise Exception(TestID + " is not equal to " + ResultInfo_singleFile_testID)
            f.write((str(Indv_SimResultFIle.DatFileUsed) +Delimiter + str(Indv_SimResultFIle.NameOfSimFile) + Delimiter +str(Indv_SimResultFIle.Total_NotCat4_notes) + Delimiter+ str(Indv_SimResultFIle.Total4A_notes)+ Delimiter +str(Indv_SimResultFIle.Total4B_notes) +"\n"))
        f.close()

        Total_4a_PC=round((Total_4a/TotalNotes)*100,2)
        Total_4b_PC=round((Total_4b/TotalNotes)*100,2)
        Total_Not4aNot4B_PC=round((Total_Not4aNot4B/TotalNotes)*100,2)

        if (Total_4a_PC + Total_4b_PC + Total_Not4aNot4B_PC) < 99.9:
            raise Exception("Bad summation")
        if (Total_4a_PC + Total_4b_PC + Total_Not4aNot4B_PC) > 100.1:
            raise Exception("Bad summation")

        print(TestID + Delimiter + str(Total_4a) + Delimiter + str(Total_4b) + Delimiter+ str(TotalNotes))


        f = open(OutputFOlder + str(ResultItem) +"    _4A_"+ str (Total_4a_PC) + "_4B_"+ str (Total_4b_PC) + "_UNKNOWN_"+ str (Total_Not4aNot4B_PC) + "__of_" + str(TotalNotes) + ".txt", "a")

        #check
        if TotalNotes!=(Total_4a+Total_4b+Total_Not4aNot4B):
            raise Exception("count not valid")

        #print (All_results_Dictionary[ResultItem][0].DictKey)

    Str_Totalpcs=Str_Totalpcs.replace("%","")
    copyToClipboard(Str_Totalpcs)
    print("Skipped files:",str(SkippedFile))
if len(list_datnames)!=len(List_Totalpcs):
    raise Exception("total % vs total dat names dont match")
#yesno("scores in clipboard, press y to populate with dat file names")

    
