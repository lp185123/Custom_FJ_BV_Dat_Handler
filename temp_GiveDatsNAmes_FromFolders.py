import _3DVisLabLib
import shutil

OutputFolder=r"E:\NCR\Currencies\Bangladesh_SR2800\Bangladesh\SR DC\MM1\TEST_SETS\RenamedDatsForTesting\\"
#get all dats in folder
Folder2check=r"E:\NCR\Currencies\Bangladesh_SR2800\Bangladesh\SR DC\MM1\TEST_SETS"
print("Looking in",Folder2check,"for .dat files")
List_all_Files=_3DVisLabLib. GetAllFilesInFolder_Recursive(Folder2check)
#filter out non .dats
List_all_Dats=_3DVisLabLib.GetList_Of_ImagesInList(List_all_Files,".dat")
print(len(List_all_Dats),".dat files found")

#filter out specific types of folders
FilteredDats=[]
for Indexer, Datfile in enumerate(List_all_Dats):
    #subfolder position of denomination code
    if (Datfile.split("\\")[-4]) in "D1D2D3D4D5D6D7D8":
        #should be correct folder structure now
        #print(Datfile.split("\\")[-4])
        #print(Datfile)
        NoSpaces=Datfile.replace(" ","")
        RenamedFile=((NoSpaces.split("\\")[-4]) + "_CAT_" + (NoSpaces.split("\\")[-3]) +"_"+ (NoSpaces.split("\\")[-2]) + "__" + str(Indexer) +".dat")
        
        Denom=Datfile.split("\\")[-4]
        OutputRenamedDat=OutputFolder+RenamedFile

        print(OutputRenamedDat)
        shutil.copy(Datfile,OutputRenamedDat)