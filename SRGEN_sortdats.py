import _3DVisLabLib
import os
import shutil

InputPath=r"E:\NCR\SR_Generations\Sprint\USD_Retest\MM8 (Minimum) - Curated"
OutputFolder=r"E:\NCR\SR_Generations\Sprint\USD_Retest\20220131_ReleaseVer2.3.0\Data"


#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputPath)
#Get pickle file - warning will just take first one 
ListAllObj_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".dat"])

for Indexer,dat in enumerate(ListAllObj_files):
    Fullfilepath=dat.split("\\")
    filepath=Fullfilepath[-1].split("_")
    DenomiPath="Denomi."
    # if Fullfilepath[8]=="001":DenomiPath=DenomiPath+"1"
    # if Fullfilepath[8]=="002":DenomiPath=DenomiPath+"2"
    # if Fullfilepath[8]=="003":DenomiPath=DenomiPath+"3"
    # if Fullfilepath[8]=="004":DenomiPath=DenomiPath+"4"
    # if Fullfilepath[8]=="005":DenomiPath=DenomiPath+"5"
    # if Fullfilepath[8]=="006":DenomiPath=DenomiPath+"6"
    # if Fullfilepath[8]=="007":DenomiPath=DenomiPath+"7"
    # if Fullfilepath[8]=="008":DenomiPath=DenomiPath+"8"
    # if Fullfilepath[8]=="009":DenomiPath=DenomiPath+"9"
    location_of_NotID=-3
    if "1 OCD_GenA" in Fullfilepath[location_of_NotID]:DenomiPath=DenomiPath+"1"
    if "5 NCD_GenB"in Fullfilepath[location_of_NotID]:DenomiPath=DenomiPath+"2"
    if "5 XCD_GenC"in Fullfilepath[location_of_NotID]:DenomiPath=DenomiPath+"3"
    if "10 XCD_GenC"in Fullfilepath[location_of_NotID]:DenomiPath=DenomiPath+"4"
    if "20 XCD_GenC"in Fullfilepath[location_of_NotID]:DenomiPath=DenomiPath+"5"
    if "50 XCD_GenC"in Fullfilepath[location_of_NotID]:DenomiPath=DenomiPath+"6"
    if "100 XCD_GenC"in Fullfilepath[location_of_NotID]:DenomiPath=DenomiPath+"7"


    if DenomiPath=="Denomi.":continue
    # if filepath[0]=="10":DenomiPath=DenomiPath+"10"
    # if filepath[0]=="11":DenomiPath=DenomiPath+"11"
    OrientationPath="Direction"
    location_of_Orientation=-2
    if Fullfilepath[location_of_Orientation]=="A":OrientationPath=OrientationPath+"A"
    if Fullfilepath[location_of_Orientation]=="B":OrientationPath=OrientationPath+"B"
    if Fullfilepath[location_of_Orientation]=="C":OrientationPath=OrientationPath+"C"
    if Fullfilepath[location_of_Orientation]=="D":OrientationPath=OrientationPath+"D"
    if OrientationPath=="Direction":continue
    FinalPath=OutputFolder+ "\\" +DenomiPath + "\\Basedata\\" +OrientationPath +"\\0001\\"
    isDirectory = os.path.isdir(FinalPath)
    if isDirectory==False:
        raise Exception(FinalPath + " path does not exist!! Cannot proceed")

    FinalPathAndDat=FinalPath+"\\" +str(Indexer) + dat.split("\\")[-1]

    src = dat
    dst =  FinalPathAndDat
    shutil.copyfile(src, dst)
    print("Copying",src,"\nto\n",dst)
    print("---")

