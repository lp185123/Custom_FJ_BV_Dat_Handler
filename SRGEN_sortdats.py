import _3DVisLabLib
import os
import shutil

InputPath=r"E:\NCR\SR_Generations\Sprint\Russia\DC\ForStd_Gen\Gen_backup"
OutputFolder=r"E:\NCR\SR_Generations\Sprint\Russia\StandardGen\20220131_ReleaseVer2.3.0\Data"


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

    if "001" in Fullfilepath[8]:DenomiPath=DenomiPath+"1"
    if "002"in Fullfilepath[8]:DenomiPath=DenomiPath+"2"
    if "003"in Fullfilepath[8]:DenomiPath=DenomiPath+"3"
    if "004"in Fullfilepath[8]:DenomiPath=DenomiPath+"4"
    if "005"in Fullfilepath[8]:DenomiPath=DenomiPath+"5"
    if "006"in Fullfilepath[8]:DenomiPath=DenomiPath+"6"
    if "007"in Fullfilepath[8]:DenomiPath=DenomiPath+"7"
    if "008"in Fullfilepath[8]:DenomiPath=DenomiPath+"8"
    if "009"in Fullfilepath[8]:DenomiPath=DenomiPath+"9"


    if DenomiPath=="Denomi.":continue
    # if filepath[0]=="10":DenomiPath=DenomiPath+"10"
    # if filepath[0]=="11":DenomiPath=DenomiPath+"11"
    OrientationPath="Direction"
    if Fullfilepath[9]=="A":OrientationPath=OrientationPath+"A"
    if Fullfilepath[9]=="B":OrientationPath=OrientationPath+"B"
    if Fullfilepath[9]=="C":OrientationPath=OrientationPath+"C"
    if Fullfilepath[9]=="D":OrientationPath=OrientationPath+"D"
    if OrientationPath=="Direction":continue
    FinalPath=OutputFolder+ "\\" +DenomiPath + "\\Genuine\\Field\\" +OrientationPath +"\\"
    isDirectory = os.path.isdir(FinalPath)
    if isDirectory==False:
        raise Exception(FinalPath + " path does not exist!! Cannt proceed")

    FinalPathAndDat=FinalPath+"\\" +str(Indexer) + dat.split("\\")[-1]

    src = dat
    dst =  FinalPathAndDat
    shutil.copyfile(src, dst)
    print("Copying",src,"to",dst)

