import _3DVisLabLib
import os
import shutil
InputPath=r"E:\NCR\Currencies\AUS\Aus_SR_DC"
OutputFolder=r"E:\NCR\SR_Generations\Australia\Generated\AUD"


#get all files in input folder
InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputPath)
#Get pickle file - warning will just take first one
ListAllObj_files=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles,ImageTypes=[".dat"])



for dat in ListAllObj_files:
    Fullfilepath=dat.split("\\")
    filepath=Fullfilepath[-1].split("_")
    DenomiPath="Denomi."
    if filepath[0]=="01":DenomiPath=DenomiPath+"1"
    if filepath[0]=="02":DenomiPath=DenomiPath+"2"
    if filepath[0]=="03":DenomiPath=DenomiPath+"3"
    if filepath[0]=="04":DenomiPath=DenomiPath+"4"
    if filepath[0]=="05":DenomiPath=DenomiPath+"5"
    if filepath[0]=="06":DenomiPath=DenomiPath+"6"
    if filepath[0]=="07":DenomiPath=DenomiPath+"7"
    if filepath[0]=="08":DenomiPath=DenomiPath+"8"
    if filepath[0]=="09":DenomiPath=DenomiPath+"9"
    if filepath[0]=="10":DenomiPath=DenomiPath+"10"
    if filepath[0]=="11":DenomiPath=DenomiPath+"11"
    OrientationPath="Direction"
    if filepath[2]=="A":OrientationPath=OrientationPath+"A"
    if filepath[2]=="B":OrientationPath=OrientationPath+"B"
    if filepath[2]=="C":OrientationPath=OrientationPath+"C"
    if filepath[2]=="D":OrientationPath=OrientationPath+"D"

    FinalPath=OutputFolder+ "\\" +DenomiPath + "\\Genuine\\Field\\" +OrientationPath +"\\"
    isDirectory = os.path.isdir(FinalPath)
    if isDirectory==False:
        raise Exception(FinalPath + " path does not exist!! Cannt proceed")

    FinalPathAndDat=FinalPath+"\\" + dat.split("\\")[-1]

    src = dat
    dst =  FinalPathAndDat
    shutil.copyfile(src, dst)
    print("Copying",src,"to",dst)

