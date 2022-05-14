import shutil
import _3DVisLabLib
import os

InputFolder=r"E:\NCR\Currencies\Bangladesh_SR2800\Bangladesh\S39\UnfilteredS39\CorrectedJustA"

Allfiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(InputFolder)

ImageFiles_Not_A=[x for x in Allfiles if not "_A_" in x and ".jpg" in x]

FilesToDelete=[]
for ImageFile in ImageFiles_Not_A:
    _ID=(ImageFile.split("\\")[-1]).split("_")[0]
    _Path="\\".join(ImageFile.split("\\")[0:-1]) +"\\"
    OutputFileName=_ID+"_s39.s39"
    FilesToDelete.append(_Path+OutputFileName)

FinalFilesToDlete=FilesToDelete + ImageFiles_Not_A

for File in FinalFilesToDlete:
    os.remove(File)
plop=1