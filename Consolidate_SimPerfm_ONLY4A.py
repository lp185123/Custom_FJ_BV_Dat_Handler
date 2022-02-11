"""spin through simulation results and collate all scores"""
import os
import win32clipboard

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

if __name__ == "__main__":
    #user input
    ResultFolder = input("Please enter folder with sim results:")
    ResultFolder=r"C:\Users\LP185123\OneDrive - NCR Corporation\Desktop\SR_GEN_Sprint\StandardDC\Malaysia\Unprocessed_SimResults\attempt 4 Unfit"
    AllFiles=GetAllFilesInFolder_Recursive(ResultFolder)
    #simulation results end with this suffix
    TxtFiles=GetList_Of_ImagesInList(AllFiles,[".dat.txt"])
    if TxtFiles==0:
        raise Exception("No valid simulation results found in" + ResultFolder)
    
    #dict to hold filename and performance
    TxtFile_V_Result=dict()
    #list of totals
    List_Totalpcs=[]
    #try building string - use stringbuilder if this gets too big
    Str_Totalpcs=""
    #roll through files
    for Item in TxtFiles:
        #ignore weird chars 
        my_file = open(Item,   errors="ignore")
        content = my_file.read()
        Delimited=content.split()
        #find first instance of total - warning! if this appears in the filename this might break
        TotalString=Delimited.index("Total")
        #prior knowledge for position of these values
        TotalPC=Delimited[TotalString+2]
        Total=Delimited[TotalString+1]
        TxtFile_V_Result[Item]=(TotalPC,Total)
        List_Totalpcs.append(TotalPC)
        Str_Totalpcs=Str_Totalpcs + str(TotalPC) + " "
        print(TxtFile_V_Result[Item])

    Str_Totalpcs=Str_Totalpcs.replace("%","")
    copyToClipboard(Str_Totalpcs)
        

    
