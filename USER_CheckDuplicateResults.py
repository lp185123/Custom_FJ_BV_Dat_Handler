
from typing import List
import _3DVisLabLib
import json
import os

#get folder from user
DefaultFOlder=r"C:\Working\FindIMage_In_Dat\MatchImages\Duplicates"
Duplicates_Folder = input("Please enter images folder: Default is" + DefaultFOlder)
if len(Duplicates_Folder)==0:
    Duplicates_Folder = DefaultFOlder

#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(Duplicates_Folder)

#filter out non images
ListAllJsons=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,[".txt"])

print(ListAllJsons)

#for each json file, create dictionary and store all .dats with same name
#relying on overwriting and removing duplicates

DatsDuplicated_dict=dict()

for JsonFile in ListAllJsons:
    data=None
    with open (JsonFile, "r") as myfile:
        data=myfile.read()
    if data is not None:
        splitdata=data.split(",")
        FilePath1=splitdata[0]
        FilePath2=splitdata[2]

        FilePath1=FilePath1.replace("[","")
        FilePath1=FilePath1.replace("]","")
        FilePath1=FilePath1.replace("'","")
        FilePath1=FilePath1.replace("//","\\")

        FilePath2=FilePath2.replace("[","")
        FilePath2=FilePath2.replace("]","")
        FilePath2=FilePath2.replace("'","")
        FilePath2=FilePath2.replace("//","\\")

        if FilePath2[0]==" ":
            FilePath2=FilePath2[1:]
        if FilePath1[0]==" ":
            FilePath1=FilePath1[1:]

        if os.path.exists(FilePath1) and os.path.exists(FilePath2):
            DatsDuplicated_dict[FilePath1]=False
            DatsDuplicated_dict[FilePath2]=False
        else:
            print("Warning!! file not reachable")
            print(FilePath1)
            print(FilePath2)
                
#clean up dicitonary - application specific
for Element in DatsDuplicated_dict:
    if "Test" in Element:
        DatsDuplicated_dict[Element]=True
#Check dats not to delete
for Element in DatsDuplicated_dict:
    if r"E:\NCR\SR_Generations\Sprint\Russia\DC\ForStd_Gen\ForGen\Gen" in Element:
        print("will not be deleting:", Element)
        if DatsDuplicated_dict[Element]==True:
            raise Exception("mismatch - double check logic")
#check dats to delete
def DatsToDelete(Delete=False):
    for Element in DatsDuplicated_dict:
        if r"E:\NCR\SR_Generations\Sprint\Russia\DC\ForStd_Gen\ForGen\Gen" in Element:
            #print("will not be deleting:", Element)
            if DatsDuplicated_dict[Element]==True:
                raise Exception("mismatch - double check logic")
        else:
            print("FOR DELETION", Element)
            if Delete==True:
                os.remove(Element)
                print("Deleted")

DatsToDelete(False)

if _3DVisLabLib.yesno("OK to continue with deletion?")==True:
    DatsToDelete(True)
