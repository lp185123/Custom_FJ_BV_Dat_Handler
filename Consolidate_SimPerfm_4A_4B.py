"""spin through simulation results and collate all 4A and 4B scoring"""
import os
import win32clipboard

class InfoStrings():
    List_known_4Bsuffix=["Clearly","_Damage","_Genuine_"]
    list_known_countries=["Belarus","Brazil","Czech","Hungary","Malaysia","Poland","Mexico","Russia","Turkey","UK"]
    list_known_GenerationTypes=["Minimum","Standard"]
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

    def GetPercentages(self):
        self.TotalNotes_PC=self.GetPercentageOfCat(self.TotalNotes)
        self.Total4A_notes_PC=self.GetPercentageOfCat(self.Total4A_notes)
        self.Total4B_notes_PC=self.GetPercentageOfCat(self.Total4B_notes)
        self.Total_NotCat4_notes_PC=self.GetPercentageOfCat(self.Total_NotCat4_notes)

    def GetPercentageOfCat(self,InputNumber):
        return round(((int(InputNumber)/int(self.TotalNotes))*100),2)

if __name__ == "__main__":
    #user input
    ResultFolder = input("Please enter folder with sim results:")
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
    #roll through files
    for Item in TxtFiles:
        if not "Sim_" in Item:
            SkippedFile.append(Item)
            continue
        #ignore weird chars 
        my_file = open(Item,   errors="ignore")
        content = my_file.read()
        Delimited=content.split()
        #find first instance of total - warning! if this appears in the filename this might break
        TotalString=Delimited.index("Total")
        FIleNameText=Delimited.index("Name")
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

        FIleName_Dat=Delimited[FIleNameText+1]

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
        if Total_Notes!=str(totalNote_check):
            raise Exception("ERROR self check total VS cat breakdown")
        #potentially circular test
        if 0!=int(Total_Notes)-int(_4B_Total)-int(_4A_Total)-int(Total_not_Cat4):
            raise Exception("ERROR self check total VS cat breakdown")

        #get country - assume is in correct folder structure
        Country=None
        for CountryName in InfoStrings.list_known_countries:
            if CountryName.lower() in Item.lower():
                Country=CountryName
        
        #get type of generation
        GenerationType=None
        for GenName in InfoStrings.list_known_GenerationTypes:
            if GenName.lower() in Item.lower():
                GenerationType=GenName
        
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

        print(vars(ResultInfo_singleFile))





        


    Str_Totalpcs=Str_Totalpcs.replace("%","")
    copyToClipboard(Str_Totalpcs)
    print("Skipped files:",str(SkippedFile))
if len(list_datnames)!=len(List_Totalpcs):
    raise Exception("total % vs total dat names dont match")
yesno("scores in clipboard, press y to populate with dat file names")

    
