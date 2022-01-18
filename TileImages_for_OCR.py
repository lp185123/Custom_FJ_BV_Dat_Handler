###Tile S39 images into a column broken up with delimiter to aid
###analysis of OCR services
import _3DVisLabLib
import cv2
import numpy


class TileImage:
    def __init__(self,DelimiterText,InputFolder,OutputFolder,ColumnSize):
        #delimiter image is the same size as s39 image, but blank
        #with composited delimiter text to allow automatic segmentation of OCR results if 
        #tiling is used
        self.DelimiterText=DelimiterText
        self.InputFolder=InputFolder
        self.OutputFolder=OutputFolder
        self.ColumnSize=ColumnSize#how long we want column of stacked images (not including delimiters)

        self.ImageX=None
        self.ImageY=None
        self.Xbuffer=20
        self.DelimiterImage=None
        self.BlankColWithDelimiters_Img=None
        #organise files
        self.ListAllImages=None#populated by GetAllImages
        self.GetAllImages()#populate ListAllImages
        self.CheckImageFiles()#ensure no mixing up of images - or at least all same dimension
        self.CreateDelimiterImage()#use valid image file from set to create delimiter image
        self.TileImages_with_delimiterImage()
        
    def GetAllImages(self):
        #get all files in input folder
        print("getting all images from,",str(self.InputFolder))
        InputFiles=_3DVisLabLib.GetAllFilesInFolder_Recursive(self.InputFolder)
        #filter out non images
        self.ListAllImages=_3DVisLabLib.GetList_Of_ImagesInList(InputFiles)
        print(len(self.ListAllImages), "images found")
    
    def CheckImageFiles(self):
        #check all images are same size
        CheckImg = cv2.imread(self.ListAllImages[0])
        for img in self.ListAllImages:
            Img = cv2.imread(img)
            if Img.shape != CheckImg.shape:
                raise Exception("Image size mismatch ", str(self.ListAllImages[0]), str(img) )

    def CreateDelimiterImage(self):
        #get first image - we can assume all images have been checked and loaded at this point
        CheckImg = cv2.imread(self.ListAllImages[0])
        self.ImageX=int(CheckImg.shape[1])
        self.ImageY=int(CheckImg.shape[0])
        #create colour image, convert to grayscale if we need to
        #make image twice the height to avoid cramming text against border
        blank_image = numpy.ones((self.ImageY*2,self.ImageX,3), dtype = numpy.uint8)#
        blank_image=blank_image*255#make all channels white

        #print delimiter text onto blank image- make sure it fits incase we get weird combinations of image size and string length
        TextScale=_3DVisLabLib.get_optimal_font_scale(self.DelimiterText,self.ImageX,self.ImageY-self.Xbuffer)#warning: this does not assume the same font
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(blank_image, self.DelimiterText, (int(self.Xbuffer/2),self.ImageY), font, TextScale, (0, 0, 0), 2, cv2.LINE_AA)    

        self.DelimiterImage=blank_image
        #_3DVisLabLib.ImageViewer_Quick_no_resize(blank_image,0,True,True)
    def TileImages_with_delimiterImage(self):
        #create columns of images with delimiter

        #roll through all images
        #Calculate Column dimensions
        ColDim_Y=self.DelimiterImage.shape[0]#delimiter may be different height from s39 images to accomodate text

        #roll through column size so we can visualise whats happening in case this gets more complicated
        NewY=0
        ListY_S39=[]
        ListY_Delimiter=[]
        for Index in range(self.ColumnSize):
            #add height of image and height of delimiter
            ListY_S39.append(NewY)
            NewY=NewY+self.ImageY
            ListY_Delimiter.append(NewY)
            NewY=NewY+ColDim_Y
            
        #now have a list of Y positions to arrange images
        #create blank white image
        blank_image = (numpy.ones((NewY,self.ImageX,3), dtype = numpy.uint8))*255

        #load in delimiter image according to Y positions generated earlier
        for Ypos in ListY_Delimiter:
            blank_image[Ypos:Ypos+ColDim_Y:,:]=self.DelimiterImage
        self.BlankColWithDelimiters_Img=blank_image

        #will have a white image with the delimited texts with pitch=height of s39 images

        #write in S39 images in sets of column size parmeter
        #get slice of input filepaths
        for Counter, Image in enumerate(self.ListAllImages):
           
            if Counter%20==0 and Counter>0:
                print("Processed image", Counter, "of",len(self.ListAllImages))
            OutputColumn=self.BlankColWithDelimiters_Img.copy()
            if Counter%self.ColumnSize==0:#take modulus
                SNRAnswersList=[]
                ImagesToEmbed=self.ListAllImages[Counter:Counter+self.ColumnSize]
                #roll through and space out into blank image with delimiter images
                for Index, ImgFilePath in enumerate(ImagesToEmbed):
                    #load image
                    LoadImage = cv2.imread(ImgFilePath)
                    #use Y positions generated to position images into master image
                    Yposition=ListY_S39[Index]
                    #place into master image
                    OutputColumn[Yposition:Yposition+self.ImageY,:,:]=LoadImage

                    #extract SNR from filename if it exists
                    #get delimited string
                    Get_SNR_string=ImgFilePath.split("[")#delimit
                    Get_SNR_string=Get_SNR_string[-1]#get last element of delimited string
                    Get_SNR_string=Get_SNR_string.split("]")#delimit
                    Get_SNR_string=Get_SNR_string[0]
                    if Get_SNR_string is not None:
                        if len(Get_SNR_string)>5:#TODO magic number
                            #keep consistent format of SNR read string
                            Get_SNR_string="[" + Get_SNR_string +"]"
                        else:
                            Get_SNR_string="NO_SNR"
                    else:
                        Get_SNR_string="NO_SNR"
                    SNRAnswersList.append(Get_SNR_string + ImgFilePath)
                #TODO lets put the delimiter text in the text file as well


                #save out delimited master image ready for OCR 
                cv2.imwrite(self.OutputFolder +"\\" + str(Counter) + ".jpg" ,OutputColumn)
                gray_img = cv2.cvtColor(OutputColumn,cv2.COLOR_BGR2GRAY)
                cv2.imwrite(self.OutputFolder +"\\" + str(Counter) + "_GRAYSCALE.jpg" ,gray_img)

                #write out SNR answer file
                with open(self.OutputFolder +"\\" + str(Counter) + "" + ".txt", 'w') as f:
                    for item in SNRAnswersList:
                        f.write("%s\n" % item)





#TestTileImage=TileImage("DISPATCH",
#r"C:\Working\FindIMage_In_Dat\TestSNs",
#r"C:\Working\FindIMage_In_Dat\OutputTestSNR\CollimatedOutput",7)



    