###Compare snr reads between two services


import Snr_test_fitness as SNRTools
import _3DVisLabLib

class CheckSN_Answers():
    def __init__(self):

        #BaseSNR_Folder = input("Please enter folder for analysis:")
        self.BaseSNR_Folder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\ToBeRead_Collimated"
        #self.BaseSNR_Folder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\ToBeRead_Single"
        #answers from external OCR verification
        self.AnswersFolder=r"C:\Working\FindIMage_In_Dat\OutputTestSNR\TestProcess\SNR_Answers"


        #self.InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(self.BaseSNR_Folder)
        #self.ListAllTextFiles=_3DVisLabLib.GetList_Of_ImagesInList(self.InputFiles)
        
        #point at single images with serial read in filename, or
        #can point at collimated folder which will have 3 files, in form:
        #0.jpg  0_GRAYSCALE.jpg  0_SNR_ANSWERS.txt
        #self.ListAllTextFiles=_3DVisLabLib.GetList_Of_ImagesInList(self.InputFiles)#collimated files (can have duplicate grayscale image) and answers
        #self.ListAllTemplateSNR=_3DVisLabLib.GetList_Of_ImagesInList(self.ListAllTextFiles,(".txt"))#possible answers text file 

        self.SingleImages=None#set this boolean correctly after asserting if working on single or collimated images
        self.Fielding=self.GetFielding()#automatically create alpanumeric fielding for SNR
        print("SNR Fielding found:",self.Fielding)

    def GetFielding(self):
        #fielding can come in two modes - SNR is embedded in single SNR image or can be in text files with multiple instaces
        #of snr for collimated images
        InputFiles_fielding=_3DVisLabLib.GetAllFilesInFolder_Recursive(self.BaseSNR_Folder)
        ListAllTextFiles=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles_fielding,(".txt"))#name of function misnomer
        ListAllImageFiles=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles_fielding)#get all images

        #if no images - nothing can proceed
        if len(ListAllImageFiles)==0:
            raise Exception("no images found in folder", self.BaseSNR_Folder)
        if len(ListAllTextFiles)>0:# text files found
            print("text files found in target folder",self.BaseSNR_Folder)
            TemplateSNRs=[]
            for TemplateCollimatedSNRs in ListAllTextFiles:
                with open(TemplateCollimatedSNRs) as f:
                    lines = f.readlines()
                    for Snr in lines:
                        TemplateSNRs.append(Snr)
            print(len(TemplateSNRs),"template SNR found within text files")
            if len(TemplateSNRs)==0:
                print("No text files with template SNR found, defaulting to embedded SNR in single images")
            else:
                Fielding=SNRTools.GenerateSN_Fielding(TemplateSNRs)
                return Fielding
        
        return None



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

# #print(SNRTools.CompareOCR_Reads()

