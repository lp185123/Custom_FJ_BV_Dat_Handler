import CompareSNR_Reads

#create images/media relabeled with the externally read SN
#example of language: '[language_code: "bn"\n] - find all codes on google vision supported lanuages or set to "None"
#set confidence to 0.0 if you want to accept everything - maximum is "1.0" - this is for quick check only to see filenames in output folder,
#the entire character set is saved in another file so refiltering
testSnr=CompareSNR_Reads.CheckSN_Answers(NoTemplateSNR_CloudOCR_Only=True,Language=None,FilterConfidenceInput=0.0)


#testSnr=CompareSNR_Reads.CheckSN_Answers(NoTemplateSNR_CloudOCR_Only=True,Language='[language_code: "en"\n]',FilterConfidenceInput=0.0)
print("fin")