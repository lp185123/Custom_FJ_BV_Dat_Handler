###Compare snr reads between two services


import Snr_test_fitness as SNRTools
import _3DVisLabLib
import re
import random
import os
import json
import pickle
import shutil
# Set environment variables
os.environ["PYTHONUTF8"] = "on"

class TracetoSource_Class():
    def __init__(self):
        self.SourceDat=None
        self.SourceDat_Record=None
        self.SourceExtractedImage=None
        self.ProcessedImage=None


def CleanUpExternalOCR(InputOCR):
    #clean up external OCR which could contain non alphanumeric characters and spaces etc
    CleanedUpSnr=""
    #remove non alphanumeric chars
    #CleanedUpSnr = re.sub(r'[^a-zA-Z0-9]', '', InputOCR)
    #remove spaces
    CleanedUpSnr=InputOCR.replace(" ","")
    #should be left with continous string of alphanumeric characters
    return CleanedUpSnr

class CheckSN_Answers():
    #if NoTemplateSNR_CloudOCR_Only is true, it will not attempt to compare template SNR and CLoud OCR (by failing gracefully)
    #if it is false, the system will compare template SNR and Cloud SNR
    def __init__(self,NoTemplateSNR_CloudOCR_Only=False,Language=None,FilterConfidenceInput=0.1):
        #Language gets filtered in "getexternalOCR", if for a character the language matches, the char will be filtered by the confidence and replace with "%"
        self.NoTemplateSNR_CloudOCR_Only=NoTemplateSNR_CloudOCR_Only
        self.BaseSNR_Folder = input("Please enter images folder: Default is C:\Working\FindIMage_In_Dat\OutputTestSNR\CollimatedOutput")
        if len(self.BaseSNR_Folder)==0:
            self.BaseSNR_Folder = r"C:\Working\FindIMage_In_Dat\OutputTestSNR\CollimatedOutput"
        #self.BaseSNR_Folder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\ToBeRead_Collimated"
        #self.BaseSNR_Folder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\ToBeRead_Single"
        #self.BaseSNR_Folder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\India"
        #answers from external OCR verification
        self.AnswersFolder = input("Please enter answers folder: Default is C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\CloudOCR")
        if len(self.AnswersFolder)==0:
            self.AnswersFolder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\CloudOCR"
        #self.AnswersFolder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\SNR_Answers_Collimated"
        #self.AnswersFolder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\SNR_Answers_Single"



        #delete old extracted s39 and jpgs if they exist - these can be regenerated so will not lose work
        DeleteExistingItems=None
        for Item in _3DVisLabLib.GetAllFilesInFolder_Recursive(self.AnswersFolder):
            if ".s39" in Item.lower() or ".jpg" in Item.lower():
                if DeleteExistingItems is None:
                    if _3DVisLabLib.yesno("s39 and jpg files found in answers folders? Delete and regenerate?"):
                        DeleteExistingItems=True
                    else:
                        DeleteExistingItems=False
                if DeleteExistingItems==True:
                    print("Deleting",Item)
                    TryAgain=True
                    while TryAgain and os.path.exists(Item):
                        try:
                            os.remove(Item)
                        except:
                            print("Error deleting",Item," :try again?")
                            TryAgain=_3DVisLabLib.yesno("")



        self.Collimated=None#set boolean to single images or collimated images which handle answers differently
        self.SingleImages=None#set this boolean correctly after asserting if working on single or collimated images
        self.ImageVS_TemplateSNR_dict=None#each image (whether single or collimated) will have list of SNR
        self.ImageVS_ExternalSNR_dict=None
        self.CollimatedImageVsImageLink_dict=None
        #populate variables - CollimatedImageVsImageLink_dict is populated if using collimated images - and allows us to 
        #trace back to find original image used to create columns. THis is NONE if using single images with embedded SNR
        self.ImageVS_TemplateSNR_dict,self.Collimated,self.Fielding,self.CollimatedImageVsImageLink_dict=self.BuildTemplate_SNR_Info()#automatically create alpanumeric fielding for SNR
        self.ImageVS_ExternalSNR_dict=self.GetExternalOCR_Answers(InputLanguage=Language,FilterConfidence=FilterConfidenceInput)
        #make sure dictionaries align
        if self.CheckBoth_setsAnswers_align(self.ImageVS_TemplateSNR_dict,self.ImageVS_ExternalSNR_dict)==True:
            MatchResults_dict=self.Check_TemplateAndExternal_OCR(self.ImageVS_TemplateSNR_dict,self.ImageVS_ExternalSNR_dict)
            #MatchResults_dict[filename]=[ResultsObjectList,ImageVS_TemplateSNR_dict[filename],ExternalOCR_key,InternalImage_key]       
            #should have dictionary of results and other info such as image paths 
            TotalPass=0
            TotalFail=0
            for MatchResult in MatchResults_dict:
                for SingleResult in MatchResults_dict[MatchResult][0]:
                    if SingleResult.Pass==True:
                        TotalPass=TotalPass+1 
                    else:
                        TotalFail=TotalFail+1

            print("Efficiency:",round((TotalPass/(TotalPass+TotalFail))*100),"% match")

            #this is not quite the right place for this function - but for the upcoming report we can also try and find the file which links back to the source DAT records;
            self.SourceTraceDict=self.Find_Sources_Dats_images(self.CollimatedImageVsImageLink_dict)
            #build report - pass in matchresults but if using collimated images wwe need self.CollimatedImageVsImageLink_dict as well
            #so we can trace where images came from 
            self.BuildReport(MatchResults_dict,self.CollimatedImageVsImageLink_dict,self.SourceTraceDict)


    def Find_Sources_Dats_images(self,CollimatedImageVsImageLink_dict):
        ##Try and find the source dat records of the images, and also the unprocessed images
        ImagesBaseFilePath_dict=dict()
        #churn through our link back to processed images (which can also be unprocessed - but a copy of the original)
        #if all is in order, these should all be in the same place - if they are not then something weird is going on or
        #user has copied and pasted stuff incorrectly
        #if one filepath test is true, find the file there which links those images back to the source images, and from there we can
        #trace back to the dat records
        #this is a labourious way to go about tracing the source but will do for now
        for Item in CollimatedImageVsImageLink_dict:
            for ImgItem in CollimatedImageVsImageLink_dict[Item]:
                SplitFilePath=ImgItem.split("\\")[0:-1]
                ReJoinFilePath=""
                for FilePathSection in SplitFilePath:
                    ReJoinFilePath=ReJoinFilePath+FilePathSection +"\\"
                ImagesBaseFilePath_dict[ReJoinFilePath]=ReJoinFilePath
        if len(ImagesBaseFilePath_dict)!=1:
            print("ERROR!!! ImagesBaseFilePath_dict:")
            print(ImagesBaseFilePath_dict)
            print("ERROR! Two filepaths found when tracing back to source images & dats\n can continue but with reduced utility")
            if _3DVisLabLib.yesno("Continue?")==True:
                return None
            else:
                raise Exception("User declined to continue process")
        #at this point we have a valid filepath - lets try and get the tracing json files
         #load file which links images back to DAT files and record
        FilePath=next(iter(ImagesBaseFilePath_dict))#gte first key in dictionary - there should only be one
        Link_Img2DatRecord_data=None
        Link_Img2SourceImg_data=None
        Link_Img2DatRecord_file=FilePath + "TraceImg_to_DatRecord.json"#TODO magic numbers - need to have these as variables pointing to one source
        Link_Img2SourceImg_file=FilePath + "SingleImg_to_ExtractionLinker.json"
        if os.path.exists(Link_Img2DatRecord_file):
            print("Reading file which links images back to DAT records",Link_Img2DatRecord_file)
            with open(Link_Img2DatRecord_file) as json_file:
                Link_Img2DatRecord_data = json.load(json_file)

        if os.path.exists(Link_Img2SourceImg_file):
            print("Reading file which links images back to DAT records",Link_Img2SourceImg_file)
            with open(Link_Img2SourceImg_file) as json_file:
                Link_Img2SourceImg_data = json.load(json_file)

        if Link_Img2DatRecord_data is None or Link_Img2SourceImg_data is None:
            print("Cannot find files which trace images back to source dat records - tool utility will be reduced")
            if _3DVisLabLib.yesno("Continue?")==False:
                raise Exception("user declined to continue")
            else:
                return None

        #should now have full traceability!!
        #create a dictionary that links up 
        SourceTraceDict=dict()
        for SourceImage in Link_Img2SourceImg_data:
            TracetoSource=TracetoSource_Class()
            #can get source image linked to processed image straight away
            TracetoSource.SourceExtractedImage=SourceImage
            TracetoSource.ProcessedImage=Link_Img2SourceImg_data[SourceImage]
            #now need to get dat & record
            if TracetoSource.SourceExtractedImage in Link_Img2DatRecord_data:
                TracetoSource.SourceDat=Link_Img2DatRecord_data[TracetoSource.SourceExtractedImage][0]
                TracetoSource.SourceDat_Record=Link_Img2DatRecord_data[TracetoSource.SourceExtractedImage][1]
            else:
                print("ERROR!! Could not find", TracetoSource.SourceExtractedImage,"in source linking dictionary!!!")
                raise Exception("cannot proceed incase files have been mixed up",Link_Img2DatRecord_file,"Delete this file to proceed without\nbeing able to link back to dat")
                
            #we will use the processed image as the key as that is the working data in this module
            SourceTraceDict[TracetoSource.ProcessedImage]=TracetoSource

        #check same length - if they are not then the results will be invalid if providing a subset of records 
        if len(SourceTraceDict)!=len(Link_Img2SourceImg_data):
            print("WARNING!! SourceTraceDict and Link_Img2SourceImg_data different sizes",len(SourceTraceDict),len(Link_Img2SourceImg_data))
            if _3DVisLabLib.yesno("Continue? Will not be able to link back to Dat records")==True:
                return None
            else:
                raise Exception("user declined to continue")
       
       #return the populated dictionary to trace images
        return SourceTraceDict
        


    def BuildReport(self,MatchResults_dict,CollimatedImageVsImageLink_dict,SourceTraceDict):
        #build html report
        buildhtml=[]
        #header of html
        buildhtml.append("<!DOCTYPE html>")
        buildhtml.append("<html>")
        buildhtml.append("<body>")
        buildhtml.append("<h2> Analysis Folder: " + self.BaseSNR_Folder+  "</h2>")

        
        SingleResult_ColImgTrace=None#need this if we have collimated images 
        TotalPass=0
        TotalFail=0
        for MatchResult in MatchResults_dict:

            if CollimatedImageVsImageLink_dict is not None:
                if MatchResult in CollimatedImageVsImageLink_dict.keys():
                    SingleResult_ColImgTrace=CollimatedImageVsImageLink_dict[MatchResult]
                else:
                    SingleResult_ColImgTrace=None
                    #raise Exception("BuildReport, could not find MatchResult in CollimatedImageVsImageLink_dict - logic error")

            for IntIndexer, SingleResult in enumerate(MatchResults_dict[MatchResult][0]):
                if SingleResult.Pass==True:
                    TotalPass=TotalPass+1
                else:
                     #we might have a dictionary to link back to source dats and images - if so use that
                    TraceObject=None
                    if SourceTraceDict is not None:
                        if SingleResult_ColImgTrace[IntIndexer] in SourceTraceDict:
                            TraceObject=SourceTraceDict[SingleResult_ColImgTrace[IntIndexer]]
                        else:
                            raise Exception("Error!! Cannot find file in source dictionary json file which was found - delete this file as it is invalid or corrupt")

                    #add to html
                    #if an image - embed it into HTML
                    #if we have collimated images we have extra complexity to trace the image provenace
                    if SingleResult_ColImgTrace is not None:
                        #collimated image - get corresponding dictionary with image link
                        buildhtml.append("<img src=" + '"' + SingleResult_ColImgTrace[IntIndexer] +'"' + ">")
                        #copy failed image?
                        ImageDelimiter=SingleResult_ColImgTrace[IntIndexer].split("\\")
                        ImageFileNameOnly=ImageDelimiter[-1]
                        try:

                            #use tracer object if we have it to get original extracted image
                            if TraceObject is not None:
                                if self.NoTemplateSNR_CloudOCR_Only==False:
                                    shutil.copyfile(TraceObject.SourceExtractedImage, self.AnswersFolder + "\\" + ImageFileNameOnly)
                                else:
                                    #warning - will be unicode so might have lots of strange characters - can sort this out in "clean up external snr" function
                                    SavePath=self.AnswersFolder + "\\" + str(SingleResult.ExternalSNR) + ".jpg"
                                    SavePath_UniqueID=1
                                    while os.path.exists(SavePath):
                                        #if file exists - start adding unique ID at end - but might already exist so continue in loop
                                        print("WARNING: File Already Exists - will be overwritten",SavePath)
                                        SavePath=self.AnswersFolder + "\\" + str(SingleResult.ExternalSNR)  + "[" + str(SavePath_UniqueID) + "]" + ".jpg"
                                        SavePath_UniqueID=SavePath_UniqueID+1

                                    #double check filename is valid
                                    if SingleResult.ExternalSNR is not None:
                                        if str(SingleResult.ExternalSNR) !="":
                                            shutil.copyfile(TraceObject.SourceExtractedImage, SavePath)
                                    #if we have s39 files and we are training the snr for the first time (not checking results)
                                    #we can copy the s39 files and name them as the SN from the external OCR tool
                                    if TraceObject.SourceDat is not None:
                                        if ".s39" in TraceObject.SourceDat.lower():
                                            #TempString=self.AnswersFolder + "\\" + SingleResult.ExternalSNR + ".S39"
                                            TempString=str(SingleResult.ExternalSNR) #.replace("O","0")
                                            #print("TAKE OUT TEMP FIX FOR JOSH os to 0s")
                                            shutil.copyfile(TraceObject.SourceDat, self.AnswersFolder + "\\" +TempString + ".S39")
                            #otherwise use the single processed files
                            else:
                                #if pre-snr training - rename the output fiesl with the external OCR
                                #otherwise keep the same filename so we can still trace back to dats
                                if self.NoTemplateSNR_CloudOCR_Only==False:
                                    shutil.copyfile(SingleResult_ColImgTrace[IntIndexer], self.AnswersFolder + "\\" + ImageFileNameOnly)
                                else:
                                    #warning - will be unicode so might have lots of strange characters - can sort this out in "clean up external snr" function
                                    shutil.copyfile(SingleResult_ColImgTrace[IntIndexer], self.AnswersFolder + "\\" + str(SingleResult.ExternalSNR) +".jpg")
                        except Exception as e:
                            print("Error with image")
                            print(MatchResult)
                            print(e)
                    else:

                        if ".jpg" in MatchResult.lower():
                            buildhtml.append("<img src=" + '"' + MatchResult +'"' + ">")
                        else:
                            buildhtml.append("<h2>" + MatchResult+  "</h2>")
                    #basic info
                    #print(vars(SingleResult))
                    buildhtml.append("""<h2 style="color:DodgerBlue;font-family:arial;">""" + "TemplateOCR_____: " +SingleResult.TemplateSNR +  "</h2>")
                    buildhtml.append("""<h2 style="color:Tomato;font-family:arial">""" +      "FieldedCloudOCR__:" + SingleResult.RepairedExternalOCR +  "</h2>")
                    buildhtml.append("<h3>" + "AutoFielding:" + str(SingleResult.ExpectedFielding) +  "</h2>")
                    buildhtml.append("<h3>" + "     Details:" + SingleResult.InfoString +  "</h2>")
                    buildhtml.append("<h3>" + "       Error:" + SingleResult.Error +  "</h2>")
                    buildhtml.append("<h3>" + "Raw CloudOCR:" + str(SingleResult.ExternalSNR) +  "</h2>")
                    #if we have trace object - can get dat file and record number as well
                    if TraceObject is not None:
                        buildhtml.append("<h3>" + "Source Trace (warning:may be offset by 1 in viewer): " + str(TraceObject.SourceDat) + " record " + str(TraceObject.SourceDat_Record) +  "</h2>")
                    #sum fails
                    TotalFail=TotalFail+1

        #put this stuff at the top
        buildhtml.append("<h2> SNR instances: " + str(TotalPass+TotalFail)+  "</h2>")
        buildhtml.append("<h2> Matches : " + str(round((TotalPass/(TotalPass+TotalFail))*100))+  "% </h2>")
        buildhtml.append("<h2> Total pass: " + str(TotalPass)+  "</h2>")
        buildhtml.append("<h2> Total Fail: " + str(TotalFail)+  "</h2>")
        #end of html
        buildhtml.append("</body>")
        buildhtml.append("</html>")
        #save out report
        with open(self.AnswersFolder + "\\" + "Report.html", 'w',encoding="utf-8") as my_list_file:
            file_content = "\n".join(buildhtml)
            my_list_file.write(file_content)

    def BuildTemplate_SNR_Info(self):
        #fielding can come in two modes - SNR is embedded in single SNR image or can be in text files with multiple instaces
        #of snr for collimated images
        InputFiles_fielding=_3DVisLabLib.GetAllFilesInFolder_Recursive(self.BaseSNR_Folder)
        ListAllTextFiles=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles_fielding,ImageTypes=([".json"]))#name of function misnomer
        ListAllImageFiles=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles_fielding)#get all images

        #create dictionary of image key vs list of snr value(s) (even if single answer)
        OutputDictionary=dict()
        #collimated images need some extra data to handle single image provenance - otherwise
        #its difficult to find mismatch image for output report
        OutputDictionary_CollimatedImageLinks=dict()
        print("automatic fielding for SNR - using folder (nested)",self.BaseSNR_Folder)
        #if no images - nothing can proceed
        if len(ListAllImageFiles)==0:
            raise Exception("no images found in folder", self.BaseSNR_Folder)

        #check that each Json is valid
        #we could check by image partner but that could introduce error if we skip over files for string handling logic fault - just grab all JSONs except for 
        #one special case and let the program crash out otherwise
        ListAllTextFiles_Cleaned=[]
        for Json in ListAllTextFiles:#TODO this is basically a magic number - make this a centralised variable rather than hardcoding it everywhere
            if (not "TraceImg_to_DatRecord.json" in Json) and (not "SingleImg_to_ExtractionLinker.json" in Json):
                ListAllTextFiles_Cleaned.append(Json)

        #if text files are in folders - may be collimated answers
        if len(ListAllTextFiles_Cleaned)>0:# text files found
            print("json files found in target folder",self.BaseSNR_Folder)
            TemplateSNRs=[]
            for TemplateCollimatedSNRs in ListAllTextFiles_Cleaned:
                print("Reading",TemplateCollimatedSNRs)
                with open(TemplateCollimatedSNRs) as json_file:
                    data = json.load(json_file)
                #roll through deserialised json file dict
                lines=[]
                lines_ColImgLink=[]
                for Indexer, Element in enumerate(data):
                    lines.append(data[str(Indexer)][0])#0 is the template SNR read result
                    lines_ColImgLink.append(data[str(Indexer)][1])#1 is the SINGLE image linked to this result - not collimated image
                OutputDictionary[TemplateCollimatedSNRs]=lines
                OutputDictionary_CollimatedImageLinks[TemplateCollimatedSNRs]=lines_ColImgLink
                for Snr in lines:
                    TemplateSNRs.append(Snr)

            print(len(TemplateSNRs),"template SNR found within text files")
            if len(TemplateSNRs)==0:
                print("No text files with template SNR found, defaulting to embedded SNR in single images")
            else:
                #return collimated images = true with fielding found and OutputDictionary_CollimatedImageLinks image to snr link
                return(OutputDictionary,True,SNRTools.GenerateSN_Fielding(TemplateSNRs),OutputDictionary_CollimatedImageLinks)
        
        #if not collimated answers - template SNR may be embedded in image filenames
        ListEmbeddedFormatSNR=[]
        for imgfilepath in ListAllImageFiles:
            print("Reading",imgfilepath)
            if ("[" in imgfilepath) and ("]" in imgfilepath):#bookends for embedded SNR#TODO warning magic letter!! make this a common variable or function
                ListEmbeddedFormatSNR.append(imgfilepath)
                OutputDictionary[imgfilepath]=[imgfilepath]
            else:
                OutputDictionary[imgfilepath]=[""]

        print(len(ListEmbeddedFormatSNR),"possible template SNR found embedded in images")
        if len(ListEmbeddedFormatSNR)==0:
            if self.NoTemplateSNR_CloudOCR_Only==False:
                raise Exception("no embedded SNR in images found (format = xx[SNR]xx", self.BaseSNR_Folder)
            else:
                #images are not expected to have any SNR embedded in the files
                pass
        #return non-collimated images = false with fielding found , and empty OutputDictionary_CollimatedImageLinks
        return(OutputDictionary,False,SNRTools.GenerateSN_Fielding(ListEmbeddedFormatSNR),OutputDictionary_CollimatedImageLinks)

        #shouldnt get here 
        raise Exception("BuildTemplate_SNR_Info incomplete logic")

  

    def GetExternalOCR_Answers(self,InputLanguage=None,FilterConfidence=0.1):
        #first try and get character analysis file - this may not exist
        CharacAnalysis_database=None
        try:
            with open(self.AnswersFolder+"\\CharAnalysis.ca", 'rb') as pickle_file:
                CharacAnalysis_database = pickle.load(pickle_file)
        except:
            print("Could not load Character analysis database",self.AnswersFolder+"\\CharAnalysis.ca")
            print("this error can be ignored")

        #answers folder from external OCR service should have same filename as 
        #input text with txt prefix
        InputFiles_ExternalOCR=_3DVisLabLib.GetAllFilesInFolder_Recursive(self.AnswersFolder)
        List_ExternalOCRTextFiles=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles_ExternalOCR,([".txt"]))#name of function misnomer
        #create dictionary of image key vs list of snr value(s) (even if single answer)
        OutputDictionary=dict()
        CharacterDictionary=dict()
        TotalCharAnalysis_Dict=dict()
        CountSingleAnswers=0
        print("Folder of SNR Answers - using folder (nested)",self.AnswersFolder)
        #if no text - nothing can proceed
        if len(List_ExternalOCRTextFiles)==0:
            raise Exception("no images found in folder", self.BaseSNR_Folder)

        #if text files are in folders - may be collimated answers
        FileCounter=0
        if len(List_ExternalOCRTextFiles)>0:# text files found
            for OCRtext in List_ExternalOCRTextFiles:
                print("Reading",OCRtext)
                with open(OCRtext,errors="ignore",encoding="utf-8") as f:#TODO potentially hiding characters - can account for lack of matching
                    lines = f.read()
                    DelimitedLines=lines.split("DISPATCH")#TODO make this common
                    Cleanedlines=[]
                    FilteredLines=[]
                    AnalysisDB_Indexer=0#trace character positions if we have multiple lines of different lengths and delimiter text
                    #can get empty lines with delimiter
                    for Index, Singleline in enumerate(DelimitedLines):
                        FileCounter=FileCounter+1
                        #delimiter may give us an empty final line which can throw up an error for alignment
                        #and may also legimiately get empty lines back from external OCR
                        if (Index< len(DelimitedLines)-1) or (len(DelimitedLines)==1):
                            CleanedLine=Singleline#CleanUpExternalOCR(Singleline)
                            Cleanedlines.append(CleanedLine)
                            CountSingleAnswers=CountSingleAnswers+1
                        


                            #at this point can filter characters according to metrics from external OCR
                            #try and make this robust if we have manually modified file system
                            if CharacAnalysis_database is not None:
                                #check that file exists in database - even if it does its not a guarantee to be a match
                                if OCRtext in CharacAnalysis_database:
                                    FilteredLine=[]

                                    #associated these items with a filename which can then be used as a look-up
                                    Ext_Confidences=[]
                                    Ext_Char=[]
                                    Ext_Language=[]
                                    Ext_BoundingBox=[]
                                    Ext_AssociatedImage_KeyTex=[OCRtext]

                                    for CHA_INDEX, Cha in enumerate(CleanedLine):
                                        #[OCRtext] is the text file with external OCR [2] is the dictionary with index of char as key, with tuple containing info such as 
                                        #language, bounding box, confidence, and character
                                        CharAnalysisChar=CharacAnalysis_database[OCRtext][2][CHA_INDEX+AnalysisDB_Indexer+1][0]
                                        Confidence=CharacAnalysis_database[OCRtext][2][CHA_INDEX+AnalysisDB_Indexer+1][1]
                                        BoundingBox=CharacAnalysis_database[OCRtext][2][CHA_INDEX+AnalysisDB_Indexer+1][2]
                                        Language=CharacAnalysis_database[OCRtext][2][CHA_INDEX+AnalysisDB_Indexer+1][3]

                                        Ext_Char.append(CharAnalysisChar)
                                        Ext_Confidences.append(Confidence)
                                        Ext_Language.append(Language)
                                        #have trouble persisting this with pickle or json so break it out manually
                                        Ext_BoundingBox.append({"vertices[0].x":BoundingBox.vertices[0].x,
                                        "vertices[0].y":BoundingBox.vertices[0].y,
                                        "vertices[1].x":BoundingBox.vertices[1].x,
                                        "vertices[1].y":BoundingBox.vertices[1].y,
                                        "vertices[2].x":BoundingBox.vertices[2].x,
                                        "vertices[2].y":BoundingBox.vertices[2].y,
                                        "vertices[3].x":BoundingBox.vertices[3].x,
                                        "vertices[3].y":BoundingBox.vertices[3].y})


                                        if Cha!=CharAnalysisChar:
                                            print("error with getting char confidences- delete .ca file ")
                                        else:

                                            if InputLanguage is not None:
                                                if Confidence>FilterConfidence and InputLanguage in Language:
                                                    #print("char",CharAnalysisChar,"pass confidence for language",InputLanguage,round(Confidence,3))
                                                    FilteredLine.append(CharAnalysisChar)

                                                    #if character does not exist - add it
                                                    if CharAnalysisChar not in CharacterDictionary:
                                                        CharacterDictionary[CharAnalysisChar]=0
                                                    CharacterDictionary[CharAnalysisChar]=CharacterDictionary[CharAnalysisChar]+1

                                                elif InputLanguage in Language:
                                                    print("char",CharAnalysisChar,"filtered for confidence",InputLanguage,round(Confidence,3),round(FilterConfidence,3))
                                                    FilteredLine.append("%")
                                                else:
                                                    print("Filtering out non-input language char",CharAnalysisChar)
                                                    
                                                    pass
                                            else:
                                                print("no filtering applied to char",InputLanguage,Confidence)
                                                #if character does not exist - add it
                                                if CharAnalysisChar not in CharacterDictionary:
                                                    CharacterDictionary[CharAnalysisChar]=0
                                                CharacterDictionary[CharAnalysisChar]=CharacterDictionary[CharAnalysisChar]+1
                                                FilteredLine.append(CharAnalysisChar)
                                    FilteredLine=''.join(FilteredLine)
                                    #FilteredLines.append(FilteredLine)
                                    FileTrackerName="[" + str(FileCounter) +"]" + str(FilteredLine)
                                    FilteredLines.append(FileTrackerName)
                                    

                                    #save into character analysis dictionary which holds all data needed to filter out in a subsequent stage
                                    TotalCharAnalysis_Dict[FileTrackerName]={"CONFIDENCE":Ext_Confidences,
                                    "CHARACTER":Ext_Char,"LANGUAGE":Ext_Language,"BOUNDINGBOX":Ext_BoundingBox,"TXTFILE_TO_FIND_IMAGE":Ext_AssociatedImage_KeyTex}
                                    #if len(FilteredLine)!=len(CleanedLine):
                                    #    print("len(FilteredLine)!=len(CleanedLine)")
                                    #    raise Exception("problem with character analysis DB, delete .ca file to prevent code running")
                        #need to get character index, which is complicated by length of delimiter
                        AnalysisDB_Indexer=AnalysisDB_Indexer+len(CleanedLine)+len("DISPATCH")

                    #if there is a characvter analysis database - this will be the filtered line with question marks for low confidence chars
                    if CharacAnalysis_database is  None:
                        OutputDictionary[OCRtext]=Cleanedlines#Cleanedlines
                    else:
                        OutputDictionary[OCRtext]=FilteredLines#FilteredLines
            
            print(len(OutputDictionary),"answer files found")
            print(CountSingleAnswers,"External SNR answers found")
        #save out character dictionary if it exists
        if TotalCharAnalysis_Dict is not None:
            _3DVisLabLib.JSON_Save(self.AnswersFolder +"\\FoundCharDic.fc",TotalCharAnalysis_Dict)

        return OutputDictionary

    
                            
    def GetPath_WO_filename(self,InputDictionary):
        ExternalOCR_Filepath=random.choice(list(InputDictionary.keys()))
        ExternalOCR_Filepath=ExternalOCR_Filepath.split("\\")[:-1]
        RebuildPathList=[]
        for Item in ExternalOCR_Filepath:
            RebuildPathList.append(Item + "\\")
        ExternalOCR_PathString=''.join(RebuildPathList)
        return ExternalOCR_PathString
    
    def GenerateAligned_Filepath(self,CompleteFilePath,NewPathString,Extension):
        #take filename and extension off CompleteFilePath and add to NewPathString
        FileNameNoPath=CompleteFilePath.split("\\")[-1]
        FileNameNoExtension=FileNameNoPath.split(".")[0]
        ExternalOCR_key=NewPathString + FileNameNoExtension + Extension
        return ExternalOCR_key

    def CheckBoth_setsAnswers_align(self, ImageVS_TemplateSNR_dict,ImageVS_ExternalOCR_dict):
        ###Check that filenames and # of snr reads align between template reads and external OCR
        #get filepath of external SNR
        #have to remove last element of filepath (filename)
        ExternalOCR_PathString=self.GetPath_WO_filename(ImageVS_ExternalOCR_dict)

        if len(ImageVS_TemplateSNR_dict)==0 or len(ImageVS_ExternalOCR_dict)==0:
            print("File alignment failed - dictionary of input images or dictionary of external answers have zero length")
            print("\n","ImageVS_TemplateSNR_dict",len(ImageVS_TemplateSNR_dict),"ImageVS_ExternalOCR_dict",len(ImageVS_ExternalOCR_dict))
            return False

        #answer files will have the same filenames, potentially different extensions and will have different filepaths
        #need to check that files align
        for filename in ImageVS_TemplateSNR_dict:
            #get last part of filename so will be "0_SNR_Answers.txt"
            ExternalOCR_key=self.GenerateAligned_Filepath(filename,ExternalOCR_PathString,".txt")
            if ExternalOCR_key in ImageVS_ExternalOCR_dict:
                #alignment is good between dictionaries - we can therefore assume files are also aligned
                #between template OCR and external OCR
                if len(ImageVS_TemplateSNR_dict[filename])!=len(ImageVS_ExternalOCR_dict[ExternalOCR_key]):
                    print("\n\nfailed alignment \n[",filename,"] \n[",ExternalOCR_key,"]")
                    print("items for files do not have matching OCR instances")
                    print(len(ImageVS_TemplateSNR_dict[filename]),len(ImageVS_ExternalOCR_dict[ExternalOCR_key]))
                    return False
            else:
                #check expected files exist to help user, Template output is always master and expects external input to align
                print("\n\nCould not align Template SNR with External SNR")
                print("failed alignment \n[",filename,"] \n[",ExternalOCR_key,"]")

                if not os.path.isfile(ExternalOCR_key):
                    print("\n Filecheck: ",ExternalOCR_key,"not found!")
                if not os.path.isfile(filename):
                    print("\nFilecheck: ",filename,"not found!")
                print("\n please check input/output folders are correct")

                #return False

        return True
    
    def Check_TemplateAndExternal_OCR(self, ImageVS_TemplateSNR_dict,ImageVS_ExternalOCR_dict):
        ###Assume that all checks have been processed succesfully
        #get filepath of external SNR
        #have to remove last element of filepath (filename)
        ExternalOCR_PathString=self.GetPath_WO_filename(ImageVS_ExternalOCR_dict)
        InternalImage_PathString=self.GetPath_WO_filename(ImageVS_TemplateSNR_dict)
        #answer files will have the same filenames, potentially different extensions and will have different filepaths
        #create dictionaries to hold matching results using snr match class
        MatchResults_dict=dict()
        for filename in ImageVS_TemplateSNR_dict:
            print("Checking template VS external SNR:\n",filename)
            #get last part of filename so will be "0_SNR_Answers.txt"
            ExternalOCR_key=self.GenerateAligned_Filepath(filename,ExternalOCR_PathString,".txt")
            InternalImage_key=self.GenerateAligned_Filepath(filename,InternalImage_PathString,".jpg")
            TemplateSNR_list=ImageVS_TemplateSNR_dict[filename]
            ExternalSNR_list=ImageVS_ExternalOCR_dict[ExternalOCR_key]
            #Should have two lists of SNR/OCR results - can now loop through and check the SNRs match
            ResultsObjectList=[]
            for Index,Item in enumerate(TemplateSNR_list):
                ResultsObjectList.append(CheckSNR_Reads(TemplateSNR_list[Index],ExternalSNR_list[Index],self.Fielding,"NO ID"))
            #pack results into the dictionary alongside the image, template SNR and external SNR files for later analysis

            for CheckElement in ResultsObjectList:
                try:
                    if CheckElement.Pass is None:
                        raise Exception("Pass element has not been populated for OCR check card")
                    if CheckElement.Pass==True or CheckElement.Pass==False:
                        pass
                except:
                    raise Exception("Error with OCR read card - see logs")

            
            MatchResults_dict[filename]=[ResultsObjectList,ImageVS_TemplateSNR_dict[filename],ExternalOCR_key,InternalImage_key]
            

        return MatchResults_dict


class OCR_analysisCard_Dummy():
    def __init__(self) -> None:
        self.TemplateSNR=None
        self.ExternalSNR=None
        self.ExpectedFielding=None
        self.Error=None
        self.InfoString=None
        self.Pass=None
        self.RepairedExternalOCR=None
        self.AnalysisId=None

def CheckSNR_Reads(TemplateSNR,ExternalSNR,Fielding,AnalysisID):
    ###pass in internal and external SNR and check for match
    #Fielding will be NONE or generated externally


    Known_SNR_string=None
        #extract snr - by previous process will be bookended by "[" and "]"
#try:
    if (not "[" in TemplateSNR) or (not "]" in TemplateSNR):
        DUmmyResult=OCR_analysisCard_Dummy()
        DUmmyResult.TemplateSNR="NONE FOUND"
        DUmmyResult.ExternalSNR=ExternalSNR
        DUmmyResult.ExpectedFielding="NONE AVAILABLE"
        DUmmyResult.Error="NA"
        DUmmyResult.InfoString="NA"
        DUmmyResult.Pass=False
        DUmmyResult.RepairedExternalOCR="NA"
        DUmmyResult.AnalysisId=AnalysisID

        return DUmmyResult
        raise Exception("Template SNR not formatted correctly []")
    Get_SNR_string=TemplateSNR.split("[")#delimit
    Get_SNR_string=Get_SNR_string[-1]#get last element of delimited string
    Get_SNR_string=Get_SNR_string.split("]")#delimit
    Get_SNR_string=Get_SNR_string[0]
    if (Get_SNR_string is not None):
        if (len(Get_SNR_string))>0:
            Known_SNR_string=Get_SNR_string
            #print(vars(SNRTools.CompareOCR_Reads(Known_SNR_string,ExternalSNR,Fielding)))
            return(SNRTools.CompareOCR_Reads(Known_SNR_string,ExternalSNR,Fielding,AnalysisID))
#except Exception as e: 
    #print("error extracting known snr string from file " )
    #print(repr(e))

    return None




# #may want to get fielding from another source if using collimated answers
# Fielding=SNRTools.GenerateSN_Fielding(InputFiles_fielding)
# print("Fielding found:",Fielding)

# Example_CollimatedOCR_Respose="PRESIDENTE DO BANCO CENTRAL DO BRAS\
# A 7435055188 A\
# A 7435055188 A\
# PRESIDENTE\
# DISPATCH\
# 88ISSOSELLU A 7435055188 A\
# DISPATCH\
# 888ISSOSELLU EA 7435055188 A\
# DISPATCH\
# A 7435055188 A\
# 88 ISSO SELLE\
# DISPATCH\
# UzZb9b06628 H\
# A 8799046422 A\
# DISPATCH\
# UZZL9b06628H A 8799045422 A\
# DISPATCH\
# A 8799046422 A\
# A 8799046422 A\
# DISPATCH"

# #if collimated - check have corrected number of "dispatch" or whatever delimiter is
# #then remove spaces between each delimited section





# SNR_Dict=dict()
# for Index,Item in enumerate(ListAllImages):
#     Known_SNR_string=None
#         #extract snr - by previous process will be bookended by "[" and "]"
#     try:
#         if not "[" in Item:
#             continue
#         if not "]" in Item:
#             continue
#         Get_SNR_string=Item.split("[")#delimit
#         Get_SNR_string=Get_SNR_string[-1]#get last element of delimited string
#         Get_SNR_string=Get_SNR_string.split("]")#delimit
#         Get_SNR_string=Get_SNR_string[0]
#         if (Get_SNR_string is not None):
#             if (len(Get_SNR_string))>0:
#                 Known_SNR_string=Get_SNR_string
#                 #load image file path and SNR as key and value
#                 SNR_Dict[Item]=Known_SNR_string

#                 print(vars(SNRTools.CompareOCR_Reads(Known_SNR_string,"zzzz",Fielding)))

#     except Exception as e: 
#         print("error extracting known snr string from file ",Item )
#         print(repr(e))

# # #print(SNRTools.CompareOCR_Reads()

