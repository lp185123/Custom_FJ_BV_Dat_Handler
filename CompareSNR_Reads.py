###Compare snr reads between two services


import Snr_test_fitness as SNRTools
import _3DVisLabLib
import re
import random
import os

class CheckSN_Answers():
    def __init__(self):

        #BaseSNR_Folder = input("Please enter folder for analysis:")
        self.BaseSNR_Folder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\ToBeRead_Collimated"
        #self.BaseSNR_Folder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\ToBeRead_Single"
        #self.BaseSNR_Folder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\India"
        #answers from external OCR verification
        self.AnswersFolder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\SNR_Answers"
        #self.AnswersFolder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\SNR_Answers_Single"

        self.Collimated=None#set boolean to single images or collimated images which handle answers differently
        self.SingleImages=None#set this boolean correctly after asserting if working on single or collimated images
        self.ImageVS_TemplateSNR_dict=None#each image (whether single or collimated) will have list of SNR
        self.ImageVS_ExternalSNR_dict=None
        #populate variables
        self.ImageVS_TemplateSNR_dict,self.Collimated,self.Fielding=self.Build_SNR_Info()#automatically create alpanumeric fielding for SNR
        self.ImageVS_ExternalSNR_dict=self.GetExternalOCR_Answers()
        #make sure dictionaries align
        if self.CheckBoth_setsAnswers_align(self.ImageVS_TemplateSNR_dict,self.ImageVS_ExternalSNR_dict)==True:
            self.Check_TemplateAndExternal_OCR(self.ImageVS_TemplateSNR_dict,self.ImageVS_ExternalSNR_dict)


    def Build_SNR_Info(self):
        #fielding can come in two modes - SNR is embedded in single SNR image or can be in text files with multiple instaces
        #of snr for collimated images
        InputFiles_fielding=_3DVisLabLib.GetAllFilesInFolder_Recursive(self.BaseSNR_Folder)
        ListAllTextFiles=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles_fielding,(".txt"))#name of function misnomer
        ListAllImageFiles=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles_fielding)#get all images

        #create dictionary of image key vs list of snr value(s) (even if single answer)
        OutputDictionary=dict()

        print("automatic fielding for SNR - using folder (nested)",self.BaseSNR_Folder)
        #if no images - nothing can proceed
        if len(ListAllImageFiles)==0:
            raise Exception("no images found in folder", self.BaseSNR_Folder)

        #if text files are in folders - may be collimated answers
        if len(ListAllTextFiles)>0:# text files found
            print("text files found in target folder",self.BaseSNR_Folder)
            TemplateSNRs=[]
            for TemplateCollimatedSNRs in ListAllTextFiles:
                print("Reading",TemplateCollimatedSNRs)
                with open(TemplateCollimatedSNRs) as f:
                    lines = f.readlines()
                    OutputDictionary[TemplateCollimatedSNRs]=lines
                    for Snr in lines:
                        TemplateSNRs.append(Snr)
            print(len(TemplateSNRs),"template SNR found within text files")
            if len(TemplateSNRs)==0:
                print("No text files with template SNR found, defaulting to embedded SNR in single images")
            else:
                #return collimated images = true with fielding found
                return(OutputDictionary,True,SNRTools.GenerateSN_Fielding(TemplateSNRs))
        
        #if not collimated answers - template SNR may be embedded in image filenames
        ListEmbeddedFormatSNR=[]
        for imgfilepath in ListAllImageFiles:
            print("Reading",imgfilepath)
            if ("[" in imgfilepath) and ("[" in imgfilepath):#bookends for embedded SNR#TODO warning magic letter!! make this a common variable or function
                ListEmbeddedFormatSNR.append(imgfilepath)
                OutputDictionary[imgfilepath]=[imgfilepath]

        print(len(ListEmbeddedFormatSNR),"possible template SNR found embedded in images")
        if len(ListEmbeddedFormatSNR)==0:
            raise Exception("no embedded SNR in images found (format = xx[SNR]xx", self.BaseSNR_Folder)
        #return collimated images = false with fielding found 
        return(OutputDictionary,False,SNRTools.GenerateSN_Fielding(ListEmbeddedFormatSNR))

        #shouldnt get here 
        return None, None, None

    def CleanUpExternalOCR(self,InputOCR):
        #clean up external OCR which could contain non alphanumeric characters and spaces etc
        CleanedUpSnr=""
        #remove non alphanumeric chars
        CleanedUpSnr = re.sub(r'[^a-zA-Z0-9]', '', InputOCR)
        #remove spaces
        CleanedUpSnr=CleanedUpSnr.replace(" ","")
        #should be left with continous string of alphanumeric characters
        return CleanedUpSnr

    def GetExternalOCR_Answers(self):
        #answers folder from external OCR service should have same filename as 
        #input images with txt prefix
        InputFiles_ExternalOCR=_3DVisLabLib.GetAllFilesInFolder_Recursive(self.AnswersFolder)
        List_ExternalOCRTextFiles=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles_ExternalOCR,(".txt"))#name of function misnomer
        #create dictionary of image key vs list of snr value(s) (even if single answer)
        OutputDictionary=dict()
        CountSingleAnswers=0
        print("Folder of SNR Answers - using folder (nested)",self.AnswersFolder)
        #if no images - nothing can proceed
        if len(List_ExternalOCRTextFiles)==0:
            raise Exception("no images found in folder", self.BaseSNR_Folder)

        #if text files are in folders - may be collimated answers
        if len(List_ExternalOCRTextFiles)>0:# text files found
            for OCRtext in List_ExternalOCRTextFiles:
                print("Reading",OCRtext)
                with open(OCRtext,errors="ignore") as f:#TODO potentially hiding characters - can account for lack of matching
                    lines = f.read()
                    DelimitedLines=lines.split("DISPATCH")#TODO make this common
                    Cleanedlines=[]
                    #can get empty lines with delimiter
                    for Singleline in DelimitedLines:
                        if len(Singleline)>3:
                            CleanedLine=self.CleanUpExternalOCR(Singleline)
                            Cleanedlines.append(CleanedLine)
                            CountSingleAnswers=CountSingleAnswers+1
                    OutputDictionary[OCRtext]=Cleanedlines
            
            print(len(OutputDictionary),"answer files found")
            print(CountSingleAnswers,"External SNR answers found")
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

                return False

        return True
    
    def Check_TemplateAndExternal_OCR(self, ImageVS_TemplateSNR_dict,ImageVS_ExternalOCR_dict):
        ###Assume that all checks have been processed succesfully
        #get filepath of external SNR
        #have to remove last element of filepath (filename)
        ExternalOCR_PathString=self.GetPath_WO_filename(ImageVS_ExternalOCR_dict)
        #answer files will have the same filenames, potentially different extensions and will have different filepaths
        #create dictionaries to hold matching results and mismatching results when we check both sets of SNR
        Match_dict=dict()
        Mismatch_dct=dict()
        for filename in ImageVS_TemplateSNR_dict:
            #get last part of filename so will be "0_SNR_Answers.txt"
            ExternalOCR_key=self.GenerateAligned_Filepath(filename,ExternalOCR_PathString,".txt")
            TemplateSNR_list=ImageVS_TemplateSNR_dict[filename]
            ExternalSNR_list=ImageVS_ExternalOCR_dict[ExternalOCR_key]
            #Should have two lists of SNR/OCR results - can now loop through and check the SNRs match
            for Index,Item in enumerate(TemplateSNR_list):
                print("Checking template VS external SNR:\n",filename,"\n",ExternalOCR_key)
                self.CheckSNR_Reads(TemplateSNR_list[Index],ExternalSNR_list[Index],self.Fielding)


        
    def CheckSNR_Reads(self,TemplateSNR,ExternalSNR,Fielding):
        ###pass in internal and external SNR and check for match
        #Fielding will be NONE or generated externally
            Known_SNR_string=None
                #extract snr - by previous process will be bookended by "[" and "]"
            try:
                if (not "[" in TemplateSNR) or (not "]" in TemplateSNR):
                    raise Exception("Template SNR not formatted correctly []")
                Get_SNR_string=TemplateSNR.split("[")#delimit
                Get_SNR_string=Get_SNR_string[-1]#get last element of delimited string
                Get_SNR_string=Get_SNR_string.split("]")#delimit
                Get_SNR_string=Get_SNR_string[0]
                if (Get_SNR_string is not None):
                    if (len(Get_SNR_string))>0:
                        Known_SNR_string=Get_SNR_string
                        print(vars(SNRTools.CompareOCR_Reads(Known_SNR_string,ExternalSNR,Fielding)))
                        return(SNRTools.CompareOCR_Reads(Known_SNR_string,ExternalSNR,Fielding))
            except Exception as e: 
                print("error extracting known snr string from file " )
                print(repr(e))




testSnr=CheckSN_Answers()

print("fin")
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

