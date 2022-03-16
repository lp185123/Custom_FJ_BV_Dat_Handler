###Tile S39 images into a column broken up with delimiter to aid
###analysis of OCR services
import _3DVisLabLib
import cv2
import numpy
import json
import random
import copy
import Snr_test_fitness
def CreateDelimiterImage(ListAllImages,DelimiterText,Xbuffer):
    #get first image - we can assume all images have been checked and loaded at this point
    CheckImg = list(ListAllImages.values())[0]
    ImageX=int(CheckImg.shape[1])
    ImageY=int(CheckImg.shape[0])
    #create colour image, convert to grayscale if we need to
    #make image twice the height to avoid cramming text against border
    blank_image = numpy.ones((ImageY*2,ImageX), dtype = numpy.uint8)#
    blank_image=blank_image*255#make all channels white

    #print delimiter text onto blank image- make sure it fits incase we get weird combinations of image size and string length
    TextScale=_3DVisLabLib.get_optimal_font_scale(DelimiterText,ImageX,ImageY-Xbuffer)#warning: this does not assume the same font
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_image, DelimiterText, (int(Xbuffer/2),ImageY), font, TextScale, (0, 0, 0), 2, cv2.LINE_AA)    

    DelimiterImage=blank_image

    return DelimiterImage,ImageX,ImageY

def CreateDelimited_Column(ColumnHeight,DelimiterImage,ImageY,ImageX):
    ColDim_Y=DelimiterImage.shape[0]
    NewY=0
    ListY_S39=[]
    ListY_Delimiter=[]
    for Index in range(ColumnHeight):
        #add height of image and height of delimiter
        ListY_S39.append(NewY)
        NewY=NewY+ImageY
        ListY_Delimiter.append(NewY)
        NewY=NewY+ColDim_Y

    #now have a list of Y positions to arrange images
    #create blank white image
    blank_image = (numpy.ones((NewY,ImageX), dtype = numpy.uint8))*255

    #load in delimiter image according to Y positions generated earlier
    for Ypos in ListY_Delimiter:
        blank_image[Ypos:Ypos+ColDim_Y:,:]=DelimiterImage
    BlankColWithDelimiters_Img=blank_image
    return ListY_S39, BlankColWithDelimiters_Img

def TileImages_with_delimiterImage(DelimiterImage,ImgVsPath,ImageY,ImageX,ColumnSize,OutputFolder):
    #create columns of images with delimiter

    ListAllImages=list(ImgVsPath.keys())
    #roll through all images
    #Calculate Column dimensions
    ColDim_Y=DelimiterImage.shape[0]#delimiter may be different height from s39 images to accomodate text

    # #roll through column size so we can visualise whats happening in case this gets more complicated
    # NewY=0
    # ListY_S39=[]
    # ListY_Delimiter=[]
    # for Index in range(ColumnSize):
    #     #add height of image and height of delimiter
    #     ListY_S39.append(NewY)
    #     NewY=NewY+ImageY
    #     ListY_Delimiter.append(NewY)
    #     NewY=NewY+ColDim_Y
        
    # #now have a list of Y positions to arrange images
    # #create blank white image
    # blank_image = (numpy.ones((NewY,ImageX,3), dtype = numpy.uint8))*255

    # #load in delimiter image according to Y positions generated earlier
    # for Ypos in ListY_Delimiter:
    #     blank_image[Ypos:Ypos+ColDim_Y:,:]=self.DelimiterImage
    # self.BlankColWithDelimiters_Img=blank_image

    #will have a white image with the delimited texts with pitch=height of s39 images

    #write in S39 images in sets of column size parmeter
    #get slice of input filepaths





    #return a dictionary of image filenames vs json filename, answerfile and image
    ImgPath_VS_ImageAndAnswer=dict()
    for Counter, Image in enumerate(ListAllImages):
        
        if Counter%20==0 and Counter>0:
            pass
            #print("Processed image", Counter, "of",len(ListAllImages))

        #generate column image - might need variable height
        if Counter%ColumnSize==0:
            if (len(ListAllImages)-Counter)< ColumnSize:
                ListY_S39,BlankColWithDelimiters_Img=CreateDelimited_Column(len(ListAllImages)-Counter,DelimiterImage,ImageY,ImageX)
            else:
                ListY_S39,BlankColWithDelimiters_Img=CreateDelimited_Column(ColumnSize,DelimiterImage,ImageY,ImageX)

            OutputColumn=BlankColWithDelimiters_Img.copy()
            SnrAnswersDict=dict()

        if Counter%ColumnSize==0:#take modulus
            SNRAnswersList=[]
            ImagesToEmbed=ListAllImages[Counter:Counter+ColumnSize]
            #roll through and space out into blank image with delimiter images
            for Index, ImgFilePath in enumerate(ImagesToEmbed):
                #load image
                LoadImage = ImgVsPath[ImgFilePath]#cv2.imread(ImgFilePath,cv2.IMREAD_GRAYSCALE)
                #use Y positions generated to position images into master image
                Yposition=ListY_S39[Index]
                #place into master image
                OutputColumn[Yposition:Yposition+ImageY,:]=LoadImage

                #extract SNR from filename if it exists
                #get delimited string
                #if no "[]" exist - images dont hvae serial number read embedded - which is valid depending on application
                if "[" in ImgFilePath:#no snr - but could hvae this in the filename so have to be careful
                    print("No SNR in file",ImgFilePath,"dummying out SNR")

                    Get_SNR_string=ImgFilePath.split("[")#delimit
                    Get_SNR_string=Get_SNR_string[-1]#get last element of delimited string
                    Get_SNR_string=Get_SNR_string.split("]")#delimit
                    Get_SNR_string=Get_SNR_string[0]
                    if Get_SNR_string is not None:
                        if len(Get_SNR_string)>1:#TODO magic number
                            #keep consistent format of SNR read string
                            Get_SNR_string="[" + Get_SNR_string +"]"
                        else:
                            Get_SNR_string=""
                    else:
                        Get_SNR_string=""
                else:
                    Get_SNR_string="NO SNR EMBEDDED IN IMAGE"
                SNRAnswersList.append(Get_SNR_string + ImgFilePath)
                SnrAnswersDict[Index]=(Get_SNR_string,ImgFilePath)
            #save out delimited master image ready for OCR 
            # cv2.imwrite(OutputFolder +"\\" + str(Counter) + ".jpg" ,OutputColumn)
            # gray_img = cv2.cvtColor(OutputColumn,cv2.COLOR_BGR2GRAY)

            # with open(OutputFolder +"\\" + str(Counter) + "" + ".json", 'w') as outfile:
            #     json.dump(SnrAnswersDict, outfile)
            
            ImgPath_VS_ImageAndAnswer[OutputFolder +"\\" + str(Counter) + ".jpg"]=(OutputFolder +"\\" + str(Counter) + "" + ".json",copy.deepcopy(SnrAnswersDict),OutputColumn.copy())


    return ImgPath_VS_ImageAndAnswer

class TileImage:
    def __init__(self,DelimiterText,InputFolder,OutputFolder,ColumnSize,Xbuffer,GenParams):
        #delimiter image is the same size as s39 image, but blank
        #with composited delimiter text to allow automatic segmentation of OCR results if 
        #tiling is used
        self.DelimiterText=DelimiterText
        self.InputFolder=InputFolder
        self.OutputFolder=OutputFolder
        if GenParams is not None:
            print("saved state found - using col size of",GenParams.ImageColumnSize)
            self.ColumnSize=GenParams.ImageColumnSize#how long we want column of stacked images (not including delimiters)
        else:
            self.ColumnSize=ColumnSize
        self.ImageX=None
        self.ImageY=None
        self.Xbuffer=Xbuffer
        self.DelimiterImage=None
        self.BlankColWithDelimiters_Img=None
        #organise files
        self.ListAllImages=dict()#populated by GetAllImages
        self.GetAllImages()#populate ListAllImages
        self.CheckImageFiles()#ensure no mixing up of images - or at least all same dimension
        self.DelimiterImage,self.ImageX,self.ImageY=CreateDelimiterImage(self.ListAllImages,self.DelimiterText,self.Xbuffer)#use valid image file from set to create delimiter image
        ImgPath_VS_ImageAndAnswer=TileImages_with_delimiterImage(self.DelimiterImage,self.ListAllImages,self.ImageY,self.ImageX,self.ColumnSize,self.OutputFolder)
        #save out images and answer files
        for Index, CollumnItem in enumerate(ImgPath_VS_ImageAndAnswer):
            cv2.imwrite(CollumnItem,ImgPath_VS_ImageAndAnswer[CollumnItem][2])#0 is the column image
            with open(ImgPath_VS_ImageAndAnswer[CollumnItem][0], 'w') as outfile:
                json.dump(ImgPath_VS_ImageAndAnswer[CollumnItem][1], outfile)#1 is the answer file

    def GetAllImages(self):
        #get all files in input folder
        print("getting all images from,",str(self.InputFolder))
        InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(self.InputFolder)
        #filter out non images
        ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)
        #package up in format expected by collimating function
        self.ListAllImages=dict()
        #format expected is dictionary with filepath as key and loaded image as value
        for imgpath in ListAllImages:
            TestImage=cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
            self.ListAllImages[imgpath]=TestImage.copy()

        print(len(self.ListAllImages), "images found")
    
    def CheckImageFiles(self):
        #check all images are same size
        Error=0
        List_to_Delete=[]
        CheckImg=None
        CheckImg_path=None
        dict_imageshapes=dict()
        for img in self.ListAllImages:
            CheckImg_path=img
            CheckImg=self.ListAllImages[img]
            #print("first image set shape check,",CheckImg,"shape",CheckImg.shape)
            break

        for img in self.ListAllImages:
            if not(self.ListAllImages[img].shape in dict_imageshapes):
                dict_imageshapes[self.ListAllImages[img].shape]=[]
            dict_imageshapes[self.ListAllImages[img].shape].append(img)

        #have a dictionary of imgae shapes (dimensions)
        #if more than 1 we have a problem
        if len(dict_imageshapes)==1:
            print("All images in same format - can proceed")
            print("Collimated images saved in:",self.OutputFolder)
            return
        
        #print image sizes found
        print("WARNING: Multiple image sizes found - not compatible with current development")
        ImgCount=0
        ShapeKey=None
        for Shapeformat in dict_imageshapes.keys():
            print("Image dimensions:",str(Shapeformat),len(dict_imageshapes[Shapeformat]),"images found")
            if len(dict_imageshapes[Shapeformat])>ImgCount:
                ImgCount=len(dict_imageshapes[Shapeformat])
                ShapeKey=Shapeformat
        print("Image format with largest image instance:",ShapeKey)
        
        Check=_3DVisLabLib.yesno("Proceed with largest image set and ignore others?")
        if Check==False:
            raise Exception("Image size mismatch in folder, do not proceed ", str(img) )
        else:
            print("Reloading subset of images")
            #remove images that dont match size/dims
            self.ListAllImages=dict()#clear list of images
            #rebuild list of images
            for ImgItem in dict_imageshapes[ShapeKey]:
                TestImage=cv2.imread(ImgItem,cv2.IMREAD_GRAYSCALE)
                self.ListAllImages[ImgItem]=TestImage.copy()
            print("Recursive image check")
            self.CheckImageFiles()
        return

    