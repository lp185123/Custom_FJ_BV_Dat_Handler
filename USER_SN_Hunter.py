import os
#import ManyMuchS39
import cv2
import BV_DatReader_Lib
import _3DVisLabLib
import enum
import numpy as np
import copy
import random
import time
import datetime
import sys, os, shutil, binascii, math
import datetime
import time
import psutil
import multiprocessing

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []#global object to handle recording mouse position
cropping = False
Global_Image=None#
GlobalMousePos=None#need to be global to handle capturing mouse on opencv UI
ImgView_Resize=1.3#HD image doesnt fit on screen

LookUpTable=dict()

class ImageExtractor:
    HEIGHT_OFFSET = 4
    DATA_WIDTH = 8
    HEADER_WIDTH = 48
    SIZE_OFFSET = 0
    TYPE_OFFSET = 1
    WAVELENGTH_OFFSET = 2
    WIDTH_OFFSET = 3
    SRU_HEADER_WIDTH = 64
    DEBUG = False

    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    CONDITIONS = ['Genuine', 'Suspect', 'Counterfeit', 'Damage', 'Cat1', 'Unknown Category', 'Unknown Category', 'Unknown Category', 'Damage (Tape)', 'Tear', 'Hole', 'Damaged (Soil)', 'Damage (Folded Corner)', 'DSP Validation Result Delay', 'Long Edge Too Short', 'Unknown Category']
    ERRORS = {
        'ae' : 'Not recognised (AE Error)',
        'ce' : 'Not recognised (CE Error)'
    }

    # Wavelengths
    WAVE_DICTIONARY = {
        '00000000' : 'None',
        '00000001' : 'A1',
        '00000002' : 'A2',
        '00000003' : 'A3',
        '00000004' : 'A4',
        '00000011' : 'B',
        '00000021' : 'C',
        '00000022' : 'D',
        '00000031' : 'E1',
        '00000032' : 'E2',
        '00000041' : 'F1',
        '00000042' : 'F2',
        '00000051' : 'UV1',
        '00000052' : 'UV2',
        '00000061' : 'Reserve 1',
        '00000071' : 'Reserve 2'
    }

    # Block types that are images
    IMAGE_TYPES = [1, 2, 3, 4, 5, 13, 14, 15, 18, 19]

    # Description of each block type
    BLOCK_TYPES = {
        0 : 'Prefix - Fixed data',
        1 : 'Image Correction Data1',
        2 : 'Image Correction Data2',
        3 : 'GBVE MM8 image',
        4 : 'GBVE MM1 image',
        5 : 'GBVE MM1 side image',
        6 : 'GBVE Magnetic 25dpi',
        7 : 'GBVE thickness',
        8 : 'Reserved(spot UV)',
        9 : 'Reserved(laser)',
        10 : 'Reserved',
        11 : 'Reserved',
        12 : 'Image Correction Data3',
        13 : 'SRU MM8 image',
        14 : 'SRU MM1 image',
        15 : 'SRU MM1 side image',
        16 : 'SRU Magnetic',
        17 : 'SRU thickness',
        18 : 'SRU SNR image1',
        19 : 'SRU SNR image2'
    }

    def littleEndianHexToInt (hexDataLittleEndian):
        return int(ImageExtractor.littleEndianHexToBE(hexDataLittleEndian), 16)

    def littleEndianHexToBE (hexDataLittleEndian):
        hexDataBigEndian = ''
        x = 0
        while x < len(hexDataLittleEndian):
            hexDataBigEndian = hexDataLittleEndian[x : x + 2] + hexDataBigEndian
            x += 2
        return hexDataBigEndian

    def __init__(self, file):

        self.file = file

        self.notes = [];

        try:
            with open(file, 'rb') as openedFile:
                self.hexstring = openedFile.read().hex()
        except FileNotFoundError:
            if ImageExtractor.DEBUG:
                print('File named ' + file + ' not found')
            return None
        except:
            if ImageExtractor.DEBUG:
                print('An error occurred in opening the file')
            return None

        # We skip past the data in the header so we don't waste time on it
        self.fileHeader = 0
        # We check if this file begins with "SRU", in which case it has the
        # SRU header
        self.header = ""
        if self.hexstring[0:6] == '535255':
            self.fileHeader = ImageExtractor.SRU_HEADER_WIDTH
            self.header = self.hexstring[0:ImageExtractor.SRU_HEADER_WIDTH]
        self.index = self.fileHeader

        # We define the limit so the parsing can stop once the file is read
        self.limit = len(self.hexstring)

        self.notes = []

        foundBlock = True
        id = 1
        while foundBlock:
            foundBlock = self.getNextBlock()

    # Returns a dictionary with only the images
    # that match a given wavelength

    # You may specify 'None' to find images without wavelengths
    def getByWavelength(self, waveDesignation):
        imagesWithWavelength = {}
        dictId = 1
        for note in self.notes:
            for image in note.images:
                if image.waveDesignation == waveDesignation and image.type in ImageExtractor.BLOCK_TYPES:

                    imagesWithWavelength[dictId] = image
                    dictId += 1

        return imagesWithWavelength

    def getByType(self, imageType):
        imagesWithType = {}
        dictId = 1
        for note in self.notes:
            for image in note.images:
                if ImageExtractor.BLOCK_TYPES[image.type] == imageType and imageType in ImageExtractor.BLOCK_TYPES.values():

                    imagesWithType[dictId] = image
                    dictId += 1

        return imagesWithType

    def filter(self, imageType, waveDesignation):
        filteredImages = {}
        dictId = 1
        for note in self.notes:
            for image in note.images:
                if ImageExtractor.BLOCK_TYPES[image.type] == imageType and image.waveDesignation == waveDesignation and imageType in ImageExtractor.BLOCK_TYPES.values():

                    filteredImages[dictId] = image
                    dictId += 1

        return filteredImages

    def getHex(self, offset, dataWidth):
        return self.hexstring[self.index + offset * dataWidth :
            self.index + (offset * dataWidth) + dataWidth]

    def isFinished(self):
        return self.index + ImageExtractor.HEADER_WIDTH > self.limit

    def getNextBlock(self):
        if self.isFinished():
            return False
        blockSize = ImageExtractor.littleEndianHexToInt(
            self.getHex(ImageExtractor.SIZE_OFFSET,ImageExtractor.DATA_WIDTH)
        ) * 2
        blockType = ImageExtractor.littleEndianHexToInt(
            self.getHex(ImageExtractor.TYPE_OFFSET,ImageExtractor.DATA_WIDTH)
        )
        self.index += blockSize
        if blockType == 1:
            self.notes.append(ExtractedNote(self.hexstring[self.index-blockSize:self.index], len(self.notes) + 1, self.index - blockSize, self))

        if blockType == 0:
            self.header = self.header + self.hexstring[self.index-blockSize:self.index]

        if blockType > 0 and blockType < 20:
            self.notes[len(self.notes) - 1].blockSize += blockSize

            self.notes[len(self.notes) - 1].images.append(
                ExtractedBlock(
                    self.hexstring[self.index-blockSize:self.index],
                    blockType,
                    self.index - blockSize,
                    blockSize / 2,
                    self.notes[len(self.notes) - 1]
                )
            )

        return True

    # Saves a clean and dirty version of the .dat
    # according to a provided list of ids
    def clean (self, arrayOfNoteIds, cleanPath = '', dirtyPath = ''):

        if cleanPath == '':
            cleanPath = self.file[0:-4] + '_clean.dat'

        if dirtyPath == '':
            dirtyPath = self.file[0:-4] + '_dirty.dat'

        arrayOfNoteIds.sort(reverse=True)

        badNotes = []
        goodNotes = []

        for i in range(0,len(self.notes)):
            addedImage = False
            # We iterate through the whole list to allow multiple of the same
            # image to be added
            for noteId in arrayOfNoteIds:
                if i == noteId:
                    addedImage = True
                    badNotes.append(self.notes[noteId])
            if addedImage == False:
                goodNotes.append(self.notes[i])

        clean = self.header
        for note in goodNotes:
            clean += note.print()

        dirty = self.header
        for note in badNotes:
            dirty += note.print()

        with open(cleanPath, "bw+") as output:
            output.write(binascii.unhexlify(clean))
        with open(dirtyPath, "bw+") as output:
            output.write(binascii.unhexlify(dirty))

class ExtractedNote:

    def __init__(self, data, id, offset, extractor):
        self.extractor = extractor
        self.blockSize = 0
        self.offset = offset
        self.id = id
        self.images = []
        if int(data[49:50], 16) < 4:
            # self.recognition = True

            # Everything inside this conditional only applies if the note
            # is recognised
            # self.countryCode = data[52:54]
            # self.denominationCode = data[54:56]
            self.condition = ImageExtractor.CONDITIONS[int(data[50:51], 16)]
            # The Serial Number Support value is stored in a single bit, the
            # first bit in this hex value, so we can evaluate it by checking if
            # this hex value is less than 8 (if it is, the bit is 0)

            # This value seems to be a flag that is set when it is NOT supported
            # rather than when it is; so if the bit is 0, we set snrSupport to
            # True
            # if int(data[51:52], 16) < 8:
            #     self.snrSupport = True
            # else:
            #     self.snrSupport = False

            # The issue code uses the remaining 3 bits (the first bit is used
            # for snrSupport, see above). We can evalulate these bits by finding
            # the remainder of the hex value after dividing by 8
            # self.issueCode = int(data[51:52], 16) % 8
        else:
            #self.recognition = False
            # Everything inside this conditional only applies if the note
            # wasn't recognised
            #self.notRecognitionErrorDetail = int(data[50:52], 16)

            self.condition = "Not Recognised"

            # if data[52:54] in ImageExtractor.ERRORS:
            #     self.condition = ImageExtractor.ERRORS[data[52:54]]
            # else:
            #     self.condition = data[52:54] + " Error"

        # The orientation code uses the remaining 2 bits (the first 2 bits are
        # used for recognition). We can evalulate these bits by finding the
        # remainder of the hex value after dividing by 8
        self.orientation = ImageExtractor.ALPHABET[int(data[17:18], 16) % 4]

        # The Parity value is stored in a single bit, the first bit in this hex
        # value, so we can evaluate it by checking if this hex value is less
        # than 8 (if it is, the bit is 0)

        # We are presently speculating that 0 = False / No; 1 = True / Yes
        # if int(data[16:17], 16) < 8:
        #     self.parity = False
        # else:
        #     self.parity = True

        # The generation code uses the remaining 3 bits (the first bit is used
        # for parity, see above). We can evalulate these bits by finding
        # the remainder of the hex value after dividing by 8
        self.generation = ImageExtractor.ALPHABET[int(data[16:17], 16) % 8]


    def print (self):
        return self.extractor.hexstring[self.offset:self.offset + self.blockSize]

class ExtractedBlock:

    def __init__(self, data, type, offset, recordSize, note):
        self.note = note
        self.type = type
        self.data = data

        if type in ImageExtractor.IMAGE_TYPES:
            wavelength = ImageExtractor.littleEndianHexToBE(
                data[ImageExtractor.WAVELENGTH_OFFSET * ImageExtractor.DATA_WIDTH :
                    (ImageExtractor.WAVELENGTH_OFFSET * ImageExtractor.DATA_WIDTH) +
                    ImageExtractor.DATA_WIDTH]
            )
        else:
            wavelength = '00000000'


        if wavelength in ImageExtractor.WAVE_DICTIONARY:
            self.waveDesignation = ImageExtractor.WAVE_DICTIONARY[wavelength]
        else:
            # Set to 'None' when we don't recognise the wavelength
            self.waveDesignation = ImageExtractor.WAVE_DICTIONARY['00000000']

        self.info = ''

        if self.type in ImageExtractor.BLOCK_TYPES:
            self.info = ImageExtractor.BLOCK_TYPES[self.type]
        else:
            self.info = 'Unexpected block type'
            if ImageExtractor.DEBUG:
                print('Unexpected block type: ' + self.type)

        if type in ImageExtractor.IMAGE_TYPES:

            # In pixels, 1 pixel = 1 bytes
            self.width = ImageExtractor.littleEndianHexToInt(
                data[ImageExtractor.WIDTH_OFFSET * ImageExtractor.DATA_WIDTH :
                    (ImageExtractor.WIDTH_OFFSET * ImageExtractor.DATA_WIDTH) +
                    ImageExtractor.DATA_WIDTH]
            )
            self.height = ImageExtractor.littleEndianHexToInt(
                data[ImageExtractor.HEIGHT_OFFSET * ImageExtractor.DATA_WIDTH :
                    (ImageExtractor.HEIGHT_OFFSET * ImageExtractor.DATA_WIDTH) +
                    ImageExtractor.DATA_WIDTH]
            )
        else:
            self.width = 0
            self.height = 0

        # The offets that define exactly where the image begins and ends
        self.offsetStart = (offset + ImageExtractor.HEADER_WIDTH) / 2
        self.offsetEnd = (
            offset + ImageExtractor.HEADER_WIDTH + ((self.width * 2) * self.height)
        ) / 2


        # This is the size of the whole block
        self.recordSize = recordSize

class ImageMerger:
    def __init__(self):

        self.shouldPrintProgress = True
        self.shouldMergeExtra = True
        self.directory = '.\\'
        self.searchText = ''
        self.arrayOfNoteCounts = []
        self.files = []
        self.outputDirectory = '.\\'

    def start(self):
        if len(self.arrayOfNoteCounts) == 0:
            self.shouldMergeExtra = True
        if self.shouldPrintProgress:
            print('Beginning merge')
            if not (self.directory == '.\\'):
                print('Using directory: ' + self.directory)
            if not (self.searchText == ''):
                print('Limiting to files with "' + self.searchText + '" in the name')
            if len(self.arrayOfNoteCounts) == 0:
                print('Merging all images into one file')
            else:
                print('Merging dats into into sizes: ' + str(self.arrayOfNoteCounts))
                if self.shouldMergeExtra:
                    print('Excess images will be merged')
                else:
                    print('Excess images will be ignored')

        self.uniqueID = 1
        if len(self.files) == 0:
            self.files = self.getFiles(self.directory, self.searchText)
        self.buildMerges()

    def getFiles(self, directory, searchText):
        allFiles = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                if name[-4:len(name)] == '.dat' and searchText in name:
                    allFiles.append(os.path.join(root, name))
        allFiles.sort()
        return allFiles

    def buildMerges(self):
        self.noteCountIndex = 0
        self.contents = ''
        self.totalMergeCount = 0
        filesCompleted = 0
        for individualFile in self.files:
            self.image = ImageExtractor(individualFile)
            self.getRequiredNotes()
            self.noteIndex = 0
            mergeCount = 0
            while (len(self.image.notes) - mergeCount > 0):
                count = self.merge()
                mergeCount += count
                self.totalMergeCount += count
                if self.requiredCount == self.totalMergeCount:
                    self.outputMerge()
                    self.contents = ''
                    self.noteCountIndex += 1
                    self.totalMergeCount = 0
                self.getRequiredNotes()
                if (not self.shouldMergeExtra) and (self.requiredCount == -1):
                    print("100%")
                    return
            if (self.shouldPrintProgress):
                filesCompleted += 1
                print(str(filesCompleted / len(self.files) * 100) + "%")

        if ((self.shouldMergeExtra) and (not (self.contents == ''))):
            self.outputMerge()

    def getRequiredNotes(self):
        if self.noteCountIndex < len(self.arrayOfNoteCounts):
            self.requiredCount = self.arrayOfNoteCounts[self.noteCountIndex]
        else:
            self.requiredCount = -1

    def merge(self):
        count = 0
        while (self.noteIndex < len(self.image.notes)) and ((self.requiredCount == -1) or (count < (self.requiredCount - self.totalMergeCount))) and ((self.requiredCount == -1) or (not (self.noteCountIndex + 1 == len(self.arrayOfNoteCounts) and (self.requiredCount == count)))):
            self.contents += self.image.notes[self.noteIndex].print()
            count += 1
            self.noteIndex += 1
        return count

    def outputMerge(self):
        with open(self.outputDirectory + datetime.now().strftime("%Y%m%d%H%M%S") + str(self.uniqueID) + '_' + str(self.totalMergeCount) + '.dat', "bw+") as output:
            output.write(binascii.unhexlify(self.image.header + self.contents))
        self.uniqueID += 1

class S39Maker:
    def __init__(self):

        self.shouldPrintProgress = True
        self.directory = '.\\'
        self.files = []
        self.outputDirectory = '.\\s39\\'
        self.wave = 'red' # or green or blue
        self.side = 'front' # or back
        self.validation = '80080103'
        self.width = 336
        self.height = 88
        self.Rotate180=False
        self.x = 0
        self.y = 0
        self.images = []
        self.mm8 = []

    def start(self):
        if self.shouldPrintProgress:
            #print('Extracting S39 from all MM8 in same directory or lower')
            if not (self.directory == '.\\'):
                print('Using directory: ' + self.directory)
        if len(self.files) == 0:
            self.files = self.getFiles(self.directory)
        for file in self.files:
            imageExtractor = ImageExtractor(file)
            mm8Images = imageExtractor.getByType('SRU MM8 image')
            if len(mm8Images) > 0:
                self.mm8.append(MM8(imageExtractor, file))
                #self.extractS39()

    def ensureDirectory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def intToLittleEndianHex(self, number):
        hexNumber = hex(number)[2:]
        while len(hexNumber) < 8:
            hexNumber = '0' + hexNumber
        littleHex = ''
        for x in range(0, len(hexNumber), 2):
            littleHex = hexNumber[x:x+2] + littleHex
        return littleHex

    def intToBigEndianHex(self, number):
        hexNumber = hex(number)[2:]
        while len(hexNumber) < 4:
            hexNumber = '0' + hexNumber
        return hexNumber

    def extractS39(self):
        global LookUpTable
        for mm8 in self.mm8:
            imageExtractor = mm8.imageExtractor
            file = mm8.file
            s39 = self.validation
            firstBlock = imageExtractor.getByType('Image Correction Data1')
            s39 += firstBlock[1].data[56:1448]
            s39 += self.intToBigEndianHex(self.width) + self.intToBigEndianHex(self.height)
            s39 += firstBlock[1].data[1456:]
            if self.wave == 'red':
                if self.side == 'front':
                    wave = 'F1'
                    waveDesignation = '41'
                elif self.side == 'back':
                    wave = 'F2'
                    waveDesignation = '42'
            elif self.wave == 'green':
                if self.side == 'front':
                    wave = 'C'
                    waveDesignation = '21'
                elif self.side == 'back':
                    wave = 'D'
                    waveDesignation = '22'
            elif self.wave == 'blue':
                if self.side == 'front':
                    wave = 'E1'
                    waveDesignation = '31'
                elif self.side == 'back':
                    wave = 'E2'
                    waveDesignation = '32'

            snr = imageExtractor.filter('SRU MM8 image', wave)
            point = 48 + (self.y * 1632 * 4)
            point += self.x * 4
            #print(len(s39))
            image = ''
            image_list=[]
            t1_start = time.perf_counter()
            for y in range(0, self.height):
                if self.y + y < 640:
                    for x in range(0, self.width):
                        if self.x + x < 1632:
                            point=int(point)#liell update - not sure why point became floating
                            HexChunk=snr[1].data[point + (x * 4):point + (x * 4) + 4]

                            try:#liell update - try to create lookup table to speed up process
                                correctedPixel=LookUpTable[HexChunk]
                            except:
                                grayscale16 = int(ImageExtractor.littleEndianHexToInt(HexChunk )/ 16)
                                correctedPixel = hex(grayscale16)[2:]
                                correctedPixel = correctedPixel[-2:]
                                while len(correctedPixel) < 2:
                                    correctedPixel = '0' + correctedPixel

                                LookUpTable[HexChunk]=correctedPixel
                            
                            #image += correctedPixel
                            image_list.append(correctedPixel)
                    point += 1632 * 4
            
            print("Rotate180",self.Rotate180)
            if self.Rotate180==True:
                image=''.join(list(reversed(image_list)))#liell update - if we want to use B or C orientaton
            else:
                image=''.join(image_list)

            self.images.append(image)
            s39 += image
            remaining = 66320 - len(s39)

            for x in range(0, remaining, 4):
                s39 += '0000'

            secondBlock = imageExtractor.getByType('Image Correction Data2')
            thirdBlock = imageExtractor.getByType('Image Correction Data3')

            hexHeight = hex(self.height)

            #       size    type (0x12 = 18 = SNR Image Type 1)                                                                                        data storage type
            s39 += '1880000012000000' + waveDesignation + '000000' + self.intToLittleEndianHex(self.width) + self.intToLittleEndianHex(self.height) + '01000000'

            s39 += secondBlock[1].data[48:]
            s39 += thirdBlock[1].data[48:4834]
            s39 += waveDesignation
            s39 += thirdBlock[1].data[4836:]

            T1_finish=round(time.perf_counter()-t1_start,3)
            print("Time per Snunt=",T1_finish)
            print(len(LookUpTable))

            directories = file.split('\\')
            self.ensureDirectory(self.outputDirectory);
            with open(self.outputDirectory + directories[len(directories) - 1][:-4] + '.s39', "bw+") as output:
                output.write(binascii.unhexlify(s39))
            #print(self.outputDirectory + directories[len(directories) - 1][:-4] + '.s39')

    def getFiles(self, directory):
        allFiles = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                if name[-4:len(name)] == '.dat':
                    allFiles.append(os.path.join(root, name))
        allFiles.sort()
        return allFiles

class MM8:
    def __init__(self, imageExtractor, file):
        self.imageExtractor = imageExtractor
        self.file = file

class S39Extractor:
    def __init__(self):

        self.shouldPrintProgress = True
        self.directory = '.\\'
        self.files = []
        self.images = []

    def start(self):
        if len(self.files) == 0:
            self.files = self.getFiles(self.directory)
            if self.shouldPrintProgress:
                print('Extracting images from ' + self.directory)
        elif self.shouldPrintProgress:
            print('Extracting images from all S39 in same directory or lower')

        for file in self.files:
            try:
                with open(file, 'rb') as openedFile:
                    s39 = openedFile.read().hex()
            except FileNotFoundError:
                if ImageExtractor.DEBUG:
                    print('File named ' + file + ' not found')
                return None
            except:
                if ImageExtractor.DEBUG:
                    print('An error occurred in opening the file')
                return None

            width = int(s39[1400:1404], 16)
            height = int(s39[1404:1408], 16)
            self.images.append(S39Image(width, height, s39[3376:3376 + (width * height * 2)]))

    def getFiles(self, directory):
        allFiles = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                if name[-4:len(name)] == '.s39':
                    allFiles.append(os.path.join(root, name))
            allFiles.sort()
        return allFiles

class S39Image:
    def __init__(self, width, height, image):

        self.width = width
        self.height = height
        self.image = image



























class SNunter_UserParams():
    def __init__(self) -> None:
        self.InputFolder=r"C:\Working\FindIMage_In_Dat\Input"
        self.OutputFolder=r"C:\Working\FindIMage_In_Dat\Output"
        self.s39_shouldPrintProgress = False
        self.s39_directory = '.\\'#repopulated later
        self.s39_outputDirectory = '.\\s39\\'#repopulated later
        self.s39_wave = 'colour' # or green or blue
        self.s39_side = 'front' # or back
        self.s39_validation = '80080103'
        self.s39_width = 336
        self.s39_height = 88
        self.s39_x = 681+358#519+336+40+40+40#add these together
        self.s39_y = 101#keep scroll point at 320 (weird coordinate systems)
        self.ColourBackGroundF=None
        self.ColourBackGroundB=None
        self.ColourBackGroundF_flood=None
        self.ColourBackGroundB_flood=None
        self.WorkingImage=None
        self.FlipHorizontal=False
        self.SearchTemplate_col=None
        self.SearchTemplate_bw=None
        self.SearchTemplate_localArea_bw=None
        self.UserSN_ROI=None
        self.CircularAoI_WithNoise=None
        self.CircularAoI_Mask_WhiteCircle=None
        self.CircularAoI_Mask_BlackCircle=None
        self.RotateSeries_SquareLocalArea_ROI=None
        self.RotateSeries_SquareLocalArea=[]
        self.RotateSeries_MaskOfSN=[]
        self.LocalAreaBuffer=50#add to area once user has selected serial number
        self.RotationRange_deg=45#rotation range - bear in mind first note may be badly skewed already
        self.RotationStepSize=1#how many steps to cover range of rotation
        self.MM8_fullResX=1632#1632
        self.MM8_fullResY=640#640
        self.PatternSearchDivideImg=3#resize image - might help search as template matching seems to die if we make image too big
        self.PatternSearchBlur=5#a little blur can help matching
        self.SubSetOfData=9999
        self.GetFloodFillImg=False
        self.MidPoint_user_SN_XY=None
        self.FloodFillUpdiff=(5, 5,5, 5)#strength of flood fill tolerance, 3 colour channels and unknown channel
        #multiprocess params
        self.FreeMemoryBuffer_pc = 30  # how much ~memory % should be reserved while multiprocessing
        # set this to "1" to force inline processing, otherwise to limit cores set to the cores you wish to use then add one (as system will remove one for safety regardless)
        self.MemoryError_ReduceLoad = (True,11)  # fix memory errors (multiprocess makes copies of everything) (Activation,N+1 cores to use -EG use 4 cores = (True,5))


def FloodFill(Inputimage,SNunter_UserParams_Loaded_int):
    SeedPoint=(50,20)
    CheckAreaColour=Inputimage[SeedPoint[0]:SeedPoint[0]+10,SeedPoint[1]:SeedPoint[1]+10]
    #get edge
    Gradimage=GetGradientImage(Inputimage)
    #dilate
    # Taking a matrix of size 5 as the kernel
    #kernel = np.ones((3,3), np.uint8)
    #image = cv2.dilate(image, kernel, iterations=1)
    
    #print("need to flood-fill to handle pattern matches at note boundary (different backgrounds)")
    #couple of blurs - this image is only for user visualisation
    Inputimage = cv2.medianBlur(Inputimage, 3)

    #add to other image to create impassable boundries for the flood fill
    image=cv2.add(Inputimage,Gradimage)

    #blur = cv2.bilateralFilter(image,9,75,75)
    #https://stackoverflow.com/questions/60197665/opencv-how-to-use-floodfill-with-rgb-image
    # 0 1 2 here is essentially a "magic number" - at moment as array is uint8 cant put in a special number like -1
    #so have to just use an unlikely colour to appear
    cv2.floodFill(image, None, seedPoint=SeedPoint, newVal=(1, 2, 3), loDiff=(10, 10, 10, 10), upDiff=SNunter_UserParams_Loaded_int.FloodFillUpdiff)

    #we want to keep the flooded area but not keep the gradient image parts - so use as a pseudo mask
    for _X in range (0,image.shape[0]):
        for _Y in range (0,image.shape[1]):
            #must be a better way of doing this
            #matching code from floodfill line above
                if image[_X,_Y,0]==1 and image[_X,_Y,1]==2 and image[_X,_Y,2]==3 :
                    Inputimage[_X,_Y,:]=[0,255,0]

    #circle for us to check that the seed is in correct location
    cv2.circle(Inputimage, SeedPoint, 10, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
    return Inputimage

def kmeans_color_quantization(image, clusters=8, rounds=1):#
    #https://stackoverflow.com/questions/60197665/opencv-how-to-use-floodfill-with-rgb-image
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

def GetContours(image):
    blur = cv2.pyrMeanShiftFiltering(image, 11, 21)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
    cnt = contours[4]
    cv2.drawContours(image, contours, 0, (0,255,0), 3)
    return image

def GetGradientImage(Image):
    blur = cv2.pyrMeanShiftFiltering(Image, 11, 21)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
     #gradient image
    ksize=3
    gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    # combine the gradient representations into a single image
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    combinedRGB=cv2.cvtColor(combined,cv2.COLOR_GRAY2RGB)
    #contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
    #cnt = contours[4]
    #cv2.drawContours(Image, contours, 0, (0,255,0), 3)
    return combinedRGB
    
def FindRectangles(image):
    blur = cv2.pyrMeanShiftFiltering(image, 11, 21)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
   
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12),2)
    return image

def RotateImage(InputImage,RotationDeg):
    #set point of rotation to centre of image - can offset if we need to
    #less verbose method would be to use imutils library
    M = cv2.getRotationMatrix2D((int((InputImage.shape[1])/2), int((InputImage.shape[0])/2)), RotationDeg, 1.0)
    rotated = cv2.warpAffine(InputImage, M, (InputImage.shape[1], InputImage.shape[0]))
    return rotated

def click_and_crop(event, x, y, flags, param):
    #this is specifically for the s39 area selection and will need to be modified
    #for other applications
    #https://pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/#:~:text=Anytime%20a%20mouse%20event%20happens,details%20to%20our%20click_and_crop%20function.
	# grab references to the global variables
    global refPt, cropping,GlobalMousePos,ImgView_Resize

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    
    #UI has to be shrunk to fit in window - working images are true size
    x=x*ImgView_Resize
    y=y*ImgView_Resize
    #if user is cropping - comply with s39 restriction for height & width being divisibly by 8
    if cropping==True:
        #get difference between start of area select and current position, then correct to be divisible by 8
        StartX=refPt[0][0]
        StartY=refPt[0][1]
        DiffX=x-StartX
        DiffY=y-StartY
        ErrorX=DiffX%8
        ErrorY=DiffY%8
        x=x-ErrorX
        y=y-ErrorY
    #set global variable
    GlobalMousePos=(int(x), int(y))


    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(int(x), int(y))]
        cropping = True
	# check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((int(x), int(y)))
        cropping = False

def SetImageParams(SNunter_UserParamsSide_int,FlipSide,FlipWave,FlipHoriz):
    #this can probably be done more easily using enums
    if FlipSide==True:
        if SNunter_UserParamsSide_int.s39_side=="front":
            SNunter_UserParamsSide_int.s39_side="back"
        else:  
            SNunter_UserParamsSide_int.s39_side="front"

    if FlipWave==True:
        if SNunter_UserParamsSide_int.s39_wave=="red":
            SNunter_UserParamsSide_int.s39_wave="green"
        elif SNunter_UserParamsSide_int.s39_wave=="green":
            SNunter_UserParamsSide_int.s39_wave="blue"
        elif SNunter_UserParamsSide_int.s39_wave=="blue":
            SNunter_UserParamsSide_int.s39_wave="colour"
        elif SNunter_UserParamsSide_int.s39_wave=="colour":
            SNunter_UserParamsSide_int.s39_wave="red"

    SNunter_UserParamsSide_int.FlipHorizontal=FlipHoriz
    
    return SNunter_UserParamsSide_int

def GetAllFilesInFolder_Recursive(root):
    ListOfFiles=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            FullpathOfFile=(os.path.join(path, name))
            ListOfFiles.append(FullpathOfFile)
    return ListOfFiles

def GetList_Of_ImagesInList(ListOfFiles, ImageTypes=(".jPg", ".Png",".gif")):
    
    Image_FileNames=[]
    
    #list comprehension [function-of-item for item in some-list
    ImageTypes_ForceLower=[x.lower()  for x in ImageTypes]
    ImageTypes_ForceLower_Tuple=tuple(ImageTypes_ForceLower)
    
    for filename in ListOfFiles:
    #if a known image filetype - copy file
        if str.endswith(str.lower(filename),ImageTypes_ForceLower_Tuple):
            Image_FileNames.append(filename)
    
    return Image_FileNames

def Get39Image(S39MakerObject,Side,wave,Width,Height,Xoffset,Yoffset,Validation,Rotate180_=False,FixMM8_AspectRatio=False):
    S39MakerObject.images=[]#clean out images list
    S39MakerObject.wave =wave
    S39MakerObject.side=Side
    S39MakerObject.validation=Validation
    S39MakerObject.x=Xoffset
    S39MakerObject.y=Yoffset
    S39MakerObject.width=Width
    S39MakerObject.height=Height
    S39MakerObject.Rotate180=Rotate180_
    #fire off the s39 maker
    S39MakerObject.extractS39()
    
    #need a dummy class for cross compatibility with other libraries
    FakedClass=BV_DatReader_Lib. DummyImageClass()
    FakedClass.offsetStart=0
    FakedClass.offsetEnd=len(S39MakerObject.images[0])
    FakedClass.width=S39MakerObject.width
    FakedClass.height=S39MakerObject.height
    filteredImages=dict()
    filteredImages["DummyNote"]=FakedClass
    #interpret hex mass as image 
    (OutputImage,dummy)=BV_DatReader_Lib.Image_from_Automatic_mode(filteredImages,"DummyNote",S39MakerObject.images[0],False)
    #mm8 data seems to be always squashed in Y 
    if FixMM8_AspectRatio==True:
        OutputImage=cv2.resize(OutputImage,(int(OutputImage.shape[1]),int(OutputImage.shape[0]*2)))
    return OutputImage

def GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_int):
    OutputImage=None
    if SNunter_UserParams_Loaded_int.s39_side == 'front':
        if SNunter_UserParams_Loaded_int.GetFloodFillImg==False:
            OutputImage=SNunter_UserParams_Loaded_int.ColourBackGroundF.copy()
        else:
            OutputImage=SNunter_UserParams_Loaded_int.ColourBackGroundF_flood.copy()
    else:
        if SNunter_UserParams_Loaded_int.GetFloodFillImg==False:
            OutputImage=SNunter_UserParams_Loaded_int.ColourBackGroundB.copy()
        else:
            OutputImage=SNunter_UserParams_Loaded_int.ColourBackGroundB_flood.copy()
    
    if SNunter_UserParams_Loaded_int.s39_wave == 'red':
        OutputImage[:,:,0]=OutputImage[:,:,2]
        OutputImage[:,:,1]=OutputImage[:,:,2]
    if SNunter_UserParams_Loaded_int.s39_wave == 'green':
        OutputImage[:,:,0]=OutputImage[:,:,1]
        OutputImage[:,:,2]=OutputImage[:,:,1]
    if SNunter_UserParams_Loaded_int.s39_wave == 'blue':
        OutputImage[:,:,1]=OutputImage[:,:,0]
        OutputImage[:,:,2]=OutputImage[:,:,0]
    
    if SNunter_UserParams_Loaded_int.FlipHorizontal==True:
        # Use Flip code 0 to flip vertically
        OutputImage = RotateImage(OutputImage,180)# cv2.flip(OutputImage, 0)

    

    return OutputImage

def SN_HuntLoop(SNunter_UserParams_Loaded,InputDat):
    global ImgView_Resize#HD image doesnt fit on screen

    #maybe find least skewed note? Either auto or by user selecting #TODO

    #extract s39 data as image in RGB for user - save as base for UI
    #populate input parameters for s39 extraction
    s39Maker = S39Maker()
    s39Maker.files=[InputDat]
    s39Maker.outputDirectory=SNunter_UserParams_Loaded.OutputFolder +"\\"
    s39Maker.start()
    #1632*320 is full mm8 image
    OutputImageR=Get39Image(s39Maker,'front','red',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    OutputImageG=Get39Image(s39Maker,'front','green',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    OutputImageB=Get39Image(s39Maker,'front','blue',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    #pull out the hex mass and convert to an image
    #create dummy dictionary for cross-compatibility with other processes
    ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
    ColourImg[:,:,0]=OutputImageB
    ColourImg[:,:,1]=OutputImageG
    ColourImg[:,:,2]=OutputImageR
    SNunter_UserParams_Loaded.ColourBackGroundF=ColourImg.copy()
    ColourImg=FloodFill(ColourImg,SNunter_UserParams_Loaded)
    SNunter_UserParams_Loaded.ColourBackGroundF_flood=ColourImg.copy()
    OutputImageR=Get39Image(s39Maker,'back','red',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    OutputImageG=Get39Image(s39Maker,'back','green',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    OutputImageB=Get39Image(s39Maker,'back','blue',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
    #pull out the hex mass and convert to an image
    ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
    ColourImg[:,:,0]=OutputImageB
    ColourImg[:,:,1]=OutputImageG
    ColourImg[:,:,2]=OutputImageR
    SNunter_UserParams_Loaded.ColourBackGroundB=ColourImg.copy()
    ColourImg=FloodFill(ColourImg,SNunter_UserParams_Loaded)
    SNunter_UserParams_Loaded.ColourBackGroundB_flood=ColourImg.copy()

    #enable global variables to be used
    global Global_Image
    global refPt
    global GlobalMousePos

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    Blur=1
    LaB=0
    
    print("Use keys 1/2/3 to cycle through viewmodes")
    print("select area in correct wave with mouse then press C to cut out Region of Interest")

    
    while True:
         #update image
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded)
        clone=Global_Image.copy()
        if len(refPt)==2:
            #smoothly reduce blurring if user has finished selecting area
            if Blur>1: Blur=Blur-1
            if LaB<SNunter_UserParams_Loaded.LocalAreaBuffer:LaB=LaB+8 
            if LaB>SNunter_UserParams_Loaded.LocalAreaBuffer:LaB=SNunter_UserParams_Loaded.LocalAreaBuffer
            #draw rectangle around area selected by user
            cv2.rectangle(Global_Image, refPt[0], refPt[1], (0, 255, 0), 2)
            #buffer rectangle (for grabbing more local features to help match template)
            #cv2.rectangle(Global_Image, (refPt[0][0]-LaB,refPt[0][1]-LaB), (refPt[1][0]+LaB,refPt[1][1]+LaB), (20, 150, 20), 1)
            #Draw Circle
            #get max dimension
            #DistanceDiagX=abs((refPt[0][0]-LaB)-(refPt[1][0]+LaB))
            MidPointX=int((refPt[0][0]+refPt[1][0])/2)
            MidPointY=int((refPt[0][1]+refPt[1][1])/2)
            Length=max(abs(refPt[0][0]-refPt[1][0]),abs(refPt[0][1]-refPt[1][1]))#assume X axis will be longes,
            cv2.circle(Global_Image, (MidPointX,MidPointY), int(Length/2), (20, 150, 20), 1)
        elif len(refPt)==1:
            LaB=0
            if Blur<35: Blur=Blur+2
            #cut out area of interest
            SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
            SNunter_UserParams_Loaded_temp.s39_wave = 'red'
            Global_Image_temp=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
            
            AoI=Global_Image_temp[refPt[0][1]:GlobalMousePos[1],refPt[0][0]:GlobalMousePos[0],:]
            #blur original image
            KernelSize=Blur
            kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing
            Global_Image = cv2.filter2D(Global_Image,-1,kernel)
            #place back AoI
            Global_Image[refPt[0][1]:GlobalMousePos[1],refPt[0][0]:GlobalMousePos[0],:]=AoI
            #draw on rectangle
            cv2.rectangle(Global_Image, refPt[0], GlobalMousePos, (100, 100, 100), 1)
        else:
            if Blur>1: Blur=Blur-2
            LaB=0
    
        # display the image and wait for a keypress
        Global_Image_view=cv2.resize(Global_Image,(int(Global_Image.shape[1]/ImgView_Resize),int(Global_Image.shape[0]/ImgView_Resize)))
        
        #Global_Image_view=FloodFill(Global_Image_view,(20,20))
        cv2.imshow("image", Global_Image_view)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            refPt=[]#clear out cropping rectangle
        # if the 'c' key is pressed, break from the loop
        if key == ord("1"):
            SNunter_UserParams_Loaded=SetImageParams(SNunter_UserParams_Loaded,True,False, SNunter_UserParams_Loaded.FlipHorizontal)
        if key == ord("2"):
            SNunter_UserParams_Loaded=SetImageParams(SNunter_UserParams_Loaded,False,True, SNunter_UserParams_Loaded.FlipHorizontal)
        if key == ord("3"):
            SNunter_UserParams_Loaded=SetImageParams(SNunter_UserParams_Loaded,False,False,not SNunter_UserParams_Loaded.FlipHorizontal)
        if key == ord("c"):
            if SNunter_UserParams_Loaded.s39_wave == 'colour':
                print("Please choose a wave using the keyboard - RGB is not compatible with .S39 extraction")
                
            else:
                if len(refPt) == 2:
                    Global_Image=clone.copy()
                    Global_Image=cv2.resize(Global_Image,(int(Global_Image.shape[1]/ImgView_Resize),int(Global_Image.shape[0]/ImgView_Resize)))
                    KernelSize=31
                    kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing
                    Global_Image_superblur = cv2.filter2D(Global_Image,-1,kernel)
                    cv2.imshow("image", Global_Image_superblur)
                    cv2.waitKey(1)#1 millisecond to refresh window
                    break
        if key==ord("4"):
            print("DEBUG - check flood function to remove background")
            SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
            SNunter_UserParams_Loaded_temp.s39_wave = 'colour'
            SNunter_UserParams_Loaded_temp.GetFloodFillImg=True
            Global_Image_Flood=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
            Global_Image_Flood=cv2.resize(Global_Image_Flood,(int(Global_Image_Flood.shape[1]/ImgView_Resize),int(Global_Image_Flood.shape[0]/ImgView_Resize)))
            cv2.imshow("image", Global_Image_Flood)
            cv2.waitKey(2000)
    # if there are two reference points, then crop the region of interest
    # from the image and display it
    if len(refPt) == 2:
        print("User parameter clipping - press any key to continue")
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded)
        roi_user = Global_Image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        SNunter_UserParams_Loaded.UserSN_ROI=(refPt)
        #cv2.imshow("imageclip", roi_user)
        #cv2.waitKey(0)
        #get pattern we will use to search other notes (full colour)
        print("Search clipping- press any key to continue")
        SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
        SNunter_UserParams_Loaded_temp.s39_wave = 'colour'
        SNunter_UserParams_Loaded_temp.GetFloodFillImg=True
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
        #Global_Image=FloodFill(Global_Image)
        roi_searchPattern = Global_Image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        #populate search template
        SNunter_UserParams_Loaded.SearchTemplate_col=roi_searchPattern#colour template for reference
        SNunter_UserParams_Loaded.SearchTemplate_bw=cv2.cvtColor(roi_searchPattern, cv2.COLOR_BGR2GRAY)#single channels template used for matching
        #cv2.imshow("imageclip", roi_searchPattern)
        #cv2.waitKey(0)
        #get a larger area of the note to help the template matcher
        #get the colour image again - double up the code incase we want to move this out
        SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
        SNunter_UserParams_Loaded_temp.s39_wave = 'colour'
        SNunter_UserParams_Loaded_temp.GetFloodFillImg=True
        Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
        #Global_Image=FloodFill(Global_Image)
        #use buffer parameter to increase size of selection to grab local features for template matching
        LaB=SNunter_UserParams_Loaded.LocalAreaBuffer
        SNunter_UserParams_Loaded.SearchTemplate_localArea_bw=cv2.cvtColor(Global_Image[refPt[0][1]-LaB:refPt[1][1]+LaB, refPt[0][0]-LaB:refPt[1][0]+LaB], cv2.COLOR_BGR2GRAY)
        #cv2.imshow("imageclip", SNunter_UserParams_Loaded.SearchTemplate_localArea_bw)
        #cv2.waitKey(0)

        

    else:
        raise Exception("No area selected - cannot proceed")

    #ready for checking through all MM8 data to find serial number matching
    #we must handle 4 note orientations and variations of angle
    #first - build template matching rotation series

    #create circular area cut - so have same blackspace during rotation - otherwise
    #if rectangular blackspace from rotation may affect template matching score

    #MOST OF THIS CAN BE REMOVED - ULTIMATELY DID NOT NEED MASKING SYSTEM

    MidPointX=int((refPt[0][0]+refPt[1][0])/2)
    MidPointY=int((refPt[0][1]+refPt[1][1])/2)

    SNunter_UserParams_Loaded.MidPoint_user_SN_XY=(MidPointX,MidPointY)
    Length=max(abs(refPt[0][0]-refPt[1][0]),abs(refPt[0][1]-refPt[1][1]))
    #create random noise image
    RandomNoiseImg=np.random.randint(255, size=(int(Length), int(Length),3),dtype="uint8")
    #cv2.imshow("imageclip", RandomNoiseImg)
    #cv2.waitKey(0)
    ##probably need circular mask
    Mask_circle=Global_Image[0:Length,0:Length,:]#use any donor image to avoid using Numpy library until we need it (keep size of exe down)
    Mask_circle[:,:,:]=0#set everything to 0
    #cv2.imshow("imageclip", Mask_circle)
    #cv2.waitKey(0)
    #create mask, of black image with white circle in the middle to be used as mask
    cv2.circle(Mask_circle,(int(Mask_circle.shape[0]/2),int(Mask_circle.shape[1]/2)),int(Length/2),(255, 255, 255),-1)
    SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle=Mask_circle
    #cv2.imshow("imageclip", Mask_circle)
    #cv2.waitKey(0)
    
    #create inverted version of mask
    Mask_circle_inverted= cv2.bitwise_not(Mask_circle)#set everything to 0
    SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle=Mask_circle_inverted
    #cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    #cv2.waitKey(0)

    #grab a square area of the image to be used for template matching
    #firstly blank out a rectangle in the image to mask out the SN
    SNunter_UserParams_Loaded_temp=copy.deepcopy(SNunter_UserParams_Loaded)
    SNunter_UserParams_Loaded_temp.s39_wave = 'colour'
    SNunter_UserParams_Loaded_temp.GetFloodFillImg=True
    Global_Image_blankSN=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
    #Global_Image=FloodFill(Global_Image)
    Global_Image_blankSN[:,:,:]=0#make black canvas
    Global_Image_blankSN[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]=255#create white strip where user rectangled SN
    #get coords from midpoint
    Y_up=int(MidPointY-Length/2)
    X_left=int(MidPointX-Length/2)
    SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea_ROI=(X_left,Y_up)
    SquareCutRegion_blankSNMask=Global_Image_blankSN[Y_up:Y_up+Length,X_left:X_left+Length,:]
    #cv2.imshow("imageclip", SquareCutRegion_blankSNMask)
    #cv2.waitKey(0)
    
    #get full colour image again in case we want to move code
    Global_Image=GetWorkingImage_FromParameters(SNunter_UserParams_Loaded_temp)
    #Global_Image=FloodFill(Global_Image)
    SquareCutRegion=Global_Image[Y_up:Y_up+Length,X_left:X_left+Length,:]
    #subtract noise
    #SquareCutRegion=cv2.subtract(SquareCutRegion,SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    #SquareCutRegion=cv2.subtract(SquareCutRegion,SquareCutRegion_blankSNMask)
    #cv2.imshow("imageclip", SquareCutRegion)
    #cv2.waitKey(0)
    #https://pyimagesearch.com/2021/01/20/opencv-rotate-image/


    #update circular masks with SN mask
    SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle=cv2.add(SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle,SquareCutRegion_blankSNMask)
    #cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    #cv2.waitKey(0)
    #update inverted mask
    SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle=cv2.bitwise_not(SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
    #cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle)
    #cv2.waitKey(0)

    #create noise around edges of square only and leave a circle to later add the rotation template
    MaskedNoise=cv2.subtract(RandomNoiseImg,SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle)
    #cv2.imshow("imageclip", MaskedNoise)
    #cv2.waitKey(0)

    #add together for composite image
    SNunter_UserParams_Loaded.CircularAoI_WithNoise=cv2.add(SquareCutRegion,MaskedNoise)
    #cv2.imshow("imageclip", SNunter_UserParams_Loaded.CircularAoI_WithNoise)
    #cv2.waitKey(0) 
  
    for RotateDeg in range (-SNunter_UserParams_Loaded.RotationRange_deg,SNunter_UserParams_Loaded.RotationRange_deg,SNunter_UserParams_Loaded.RotationStepSize):
        #RotatedImage=RotateImage(SNunter_UserParams_Loaded.SquareCutRegion,RotateDeg)
        #SubtractImg=cv2.subtract(RotatedImage,SNunter_UserParams_Loaded.CircularAoI_Mask_BlackCircle)
        #RotatedImage=RotateImage(SquareCutRegion,RotateDeg)
        RotatedImage_mask=RotateImage(SNunter_UserParams_Loaded.CircularAoI_Mask_WhiteCircle,RotateDeg)
        #RotatedImg_andMask=cv2.add(MaskedNoise,SubtractImg)
        #SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea.append(RotatedImage)
        SNunter_UserParams_Loaded.RotateSeries_MaskOfSN.append(RotatedImage_mask)
        #cv2.imshow("imageclip", RotatedImage)
        #cv2.waitKey(0)
        #cv2.imshow("imageclip", RotatedImage_mask)
        #cv2.waitKey(0)

        #second rotation technique
        #rotate the main image and crop a square from the center of selection region
        M = cv2.getRotationMatrix2D((MidPointX,MidPointY), RotateDeg, 1.0)
        rotated = cv2.warpAffine(Global_Image, M, (Global_Image.shape[1], Global_Image.shape[0]))
        SquareCutRegion_rot=rotated[Y_up:Y_up+Length,X_left:X_left+Length,:]
        SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea.append(SquareCutRegion_rot)
        #cv2.imshow("imageclip", SquareCutRegion_rot)
        #cv2.waitKey(0)

        #save out rotation sequence for debugging
        RotateMatchPAttern_filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(RotateDeg) + "_RotateMatchPattern.jpg"
        #cv2.imwrite(RotateMatchPAttern_filename,SquareCutRegion_rot)


    return SNunter_UserParams_Loaded

def PrepareMultiProcessing(SNunter_UserParams_Loaded,InputDats_list):
    #set up multiprocessing
    #PhysicalCores=psutil.cpu_count(logical=False)#number of physical cores
    Cores_Available = int(os.environ['NUMBER_OF_PROCESSORS'])#hyperthreaded cores - not compatible with some processes
    #final core count available
    CoresTouse=1
    #user may have restricted performance to overcome memory errors or to leave system capacity for other tasks
    if SNunter_UserParams_Loaded.MemoryError_ReduceLoad[0]==True and Cores_Available>1:
        CoresTouse=min(Cores_Available,SNunter_UserParams_Loaded.MemoryError_ReduceLoad[1])#if user has over-specified cores restrict to cores available
        print("THROTTLING BY USER - Memory protection: restricting cores to", CoresTouse, "or less, user option MemoryError_ReduceLoad")
    else:
        CoresTouse=Cores_Available
    #if no restriction by user , leave a core anyway
    processes=max(CoresTouse-1,1)#rule is thumb is to use number of logical cores minus 1, but always make sure this number >0. Its not a good idea to blast CPU at 100% as this can reduce performance as OS tries to balance the load
    #find how much memory single process uses (windows)
    Currentprocess = psutil.Process(os.getpid())
    SingleProcess_Memory=Currentprocess.memory_percent()
    SystemMemoryUsed=psutil.virtual_memory().percent
    FreeMemoryBuffer_pc=SNunter_UserParams_Loaded.FreeMemoryBuffer_pc#arbitrary free memory to leave
    MaxPossibleProcesses=max(math.floor((100-FreeMemoryBuffer_pc-SystemMemoryUsed)/SingleProcess_Memory),0)#can't be lower than zero
    print(MaxPossibleProcesses,"parallel processes possible at system capacity (leaving",FreeMemoryBuffer_pc,"% memory free)")
    print("Each process will use ~ ",round(psutil.virtual_memory().total*SingleProcess_Memory/100000000000,1),"gb")#convert bytes to gb
    #cannot proceed if we can't use even one core
    if processes<1 or MaxPossibleProcesses<1:
        print(("Multiprocess Configuration error! Less than 1 process possible - memory or logic error"))
        processes=1
        MaxPossibleProcesses=1
        print("Forcing processors =1, may cause memory error")
        #raise Exception("Multiprocess Configuration error! Less than 1 process possible - memory or logic error")
    #check system has enough memory - if not restrict cores used
    if processes>MaxPossibleProcesses:
        print("WARNING!! possible memory overflow - restricting number of processes from",processes,"to",MaxPossibleProcesses)
        processes=MaxPossibleProcesses

    ThreadsNeeded=len(InputDats_list)
    #calculate how many processes each CPU will get per cycle
    chunksize=processes*3#arbitrary setting to be able to get time feedback and not clog up system - but each reset of tasks can have memory overhead
    #how many jobs do we build up to pass off to the multiprocess pool, in this case in theory each core gets 3 stacked tasks
    ProcessesPerCycle=processes*chunksize#this might be bigger than amount of multiprocesses needed
    #randomise tasks - application specific and not required in this situation
    #SacrificialDictionary=copy.deepcopy(MatchImages.ImagesInMem_Pairing)
    #ImagesInMem_Pairing_ForThreading=dict()
    #while len(SacrificialDictionary)>0:
    #    RandomItem=random.choice(list(SacrificialDictionary.keys()))
    #    ImagesInMem_Pairing_ForThreading[RandomItem]=MatchImages.ImagesInMem_Pairing[RandomItem]
    #    del SacrificialDictionary[RandomItem]

    print("[Multiprocess start]","Taskstack per core:",chunksize,"  Taskpool size:",ProcessesPerCycle,"  Physical cores used:",processes,"   Image Threads:",ThreadsNeeded)


def SearchMM8_forSN_Location(SNunter_UserParams_Loaded,InputDats_list):



    

    #1 6 3 2 * 6 4 0 full res
    #roll through all dats in input list
     #start timer and time metrics
    listTimings=[]
    listCounts=[]
    listAvgTime=[]
    ProcessOnly_start = time.perf_counter()
    t1_start = None
    for DatIndex,DatFile in enumerate(InputDats_list):
        try:
            if DatIndex>0:
                listTimings.append(round(time.perf_counter()-t1_start,2))
                #don't need linear regression as process is static - keep in case we add something funky
                #slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(listTimings[1:-1])),listTimings[1:-1])
                TimePerProcess=sum(listTimings)/len(listTimings)
                JobsLeft=len(InputDats_list)-DatIndex
                print("Estimated time left for" + str(len(InputDats_list)-DatIndex)+"jobs:",str(datetime.timedelta(seconds=(TimePerProcess*JobsLeft))))
                print("Total time for",len(InputDats_list),"jobs:",str(datetime.timedelta(seconds=(TimePerProcess*len(InputDats_list)))))
                print("Time per Snunt:",str(datetime.timedelta(seconds=(TimePerProcess))))
            #start timer again
            t1_start = time.perf_counter()
        except:
            print("Timing code broken")
        #pull out entire mm8 image in colour
        LoadedImg_dict={}
        s39Maker = S39Maker()
        s39Maker.files=[]
        s39Maker.files=[DatFile]
        s39Maker.outputDirectory=SNunter_UserParams_Loaded.OutputFolder +"\\"
        s39Maker.start()
        #1632*320 is full mm8 image
        #1632*320 is full mm8 image
        OutputImageR=Get39Image(s39Maker,'front','red',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
        OutputImageG=Get39Image(s39Maker,'front','green',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
        OutputImageB=Get39Image(s39Maker,'front','blue',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
        #persist waves into dictionary
        LoadedImg_dict[('front','red')]=cv2.cvtColor(OutputImageR.copy(),cv2.COLOR_GRAY2RGB)
        LoadedImg_dict[('front','green')]=cv2.cvtColor(OutputImageG.copy(),cv2.COLOR_GRAY2RGB)
        LoadedImg_dict[('front','blue')]=cv2.cvtColor(OutputImageB.copy(),cv2.COLOR_GRAY2RGB)
        

        #pull out the hex mass and convert to an image
        #create dummy dictionary for cross-compatibility with other processes
        ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
        ColourImg[:,:,0]=OutputImageB
        ColourImg[:,:,1]=OutputImageG
        ColourImg[:,:,2]=OutputImageR
        ColourImg=FloodFill(ColourImg,SNunter_UserParams_Loaded)
        ColourBackGroundF=ColourImg#combine channels into colour image
        
        OutputImageR=Get39Image(s39Maker,'back','red',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
        OutputImageG=Get39Image(s39Maker,'back','green',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
        OutputImageB=Get39Image(s39Maker,'back','blue',SNunter_UserParams_Loaded.MM8_fullResX,SNunter_UserParams_Loaded.MM8_fullResY,0,0,"80080103",FixMM8_AspectRatio=True)
        #persist waves into dictionary
        LoadedImg_dict[('back','red')]=cv2.cvtColor(OutputImageR.copy(),cv2.COLOR_GRAY2RGB)
        LoadedImg_dict[('back','green')]=cv2.cvtColor(OutputImageG.copy(),cv2.COLOR_GRAY2RGB)
        LoadedImg_dict[('back','blue')]=cv2.cvtColor(OutputImageB.copy(),cv2.COLOR_GRAY2RGB)
        #pull out the hex mass and convert to an image
        ColourImg=cv2.cvtColor(OutputImageR,cv2.COLOR_GRAY2RGB)
        ColourImg[:,:,0]=OutputImageB
        ColourImg[:,:,1]=OutputImageG
        ColourImg[:,:,2]=OutputImageR
        ColourImg=FloodFill(ColourImg,SNunter_UserParams_Loaded)
        ColourBackGroundB=ColourImg#combine channels into colour image

        #stack all images - we need to do this due to some template matching methods compatible with masking
        #give us results more difficult to decipher if analysing each note orientation independantly
        #multiply Y by 4 (# of orientations)
        StackOrientations=np.zeros((ColourBackGroundB.shape[0],ColourBackGroundB.shape[1]*4,ColourBackGroundB.shape[2]),dtype='uint8')
        StackOrientations_UserWave=np.zeros((ColourBackGroundB.shape[0],ColourBackGroundB.shape[1]*4,ColourBackGroundB.shape[2]),dtype='uint8')
        
        
        Offset=0
        
        #add front (A)
        StackOrientations[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=ColourBackGroundF
        StackOrientations_UserWave[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=LoadedImg_dict[('front',SNunter_UserParams_Loaded.s39_wave)]

        #add front rotate(B)
        Offset=Offset+ColourBackGroundF.shape[1]
        StackOrientations[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=RotateImage(ColourBackGroundF,180)
        StackOrientations_UserWave[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=RotateImage(LoadedImg_dict[('front',SNunter_UserParams_Loaded.s39_wave)],180)
        
        #add back rotate(C)
        Offset=Offset+ColourBackGroundF.shape[1]
        StackOrientations[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=RotateImage(ColourBackGroundB,180)
        StackOrientations_UserWave[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=RotateImage(LoadedImg_dict[('back',SNunter_UserParams_Loaded.s39_wave)],180)
       
        #add back (D)
        Offset=Offset+ColourBackGroundF.shape[1]
        StackOrientations[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=ColourBackGroundB
        StackOrientations_UserWave[0:ColourBackGroundF.shape[0],0+Offset:ColourBackGroundF.shape[1]+Offset,:]=LoadedImg_dict[('back',SNunter_UserParams_Loaded.s39_wave)]

        #StackOrientations=ColourBackGroundF
        StackOrientations=cv2.resize(StackOrientations,(int(StackOrientations.shape[1]/SNunter_UserParams_Loaded.PatternSearchDivideImg),int(StackOrientations.shape[0]/SNunter_UserParams_Loaded.PatternSearchDivideImg)))
        #blurring might help template matching
        #NOTE this is already done in floodfill

        KernelSize=9
        kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing - maybe make smoother as such small images
        StackOrientations = cv2.filter2D(StackOrientations,-1,kernel)
        #cv2.imshow("imageclip", ColourBackGroundB)
        #StackOrientations=cv2.resize(ColourBackGroundF,(int(ColourBackGroundF.shape[0]/4),int(ColourBackGroundF.shape[1]/4)))
        cv2.imshow("image",cv2.resize(StackOrientations,(800,800)))
        cv2.waitKey(1)
        #cv2.waitKey(0)

        # Apply template Matching
        #roll through our list of the template and mask through a rotation range to handle skew of notes
        #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
        Latch_MaxValue=None
        Latch_SavedBestPattern_userROI=None
        Latch_MaxValueIndex=None
        Latch_MaxValueTopLeft=None
        Latch_MaxValueTopRight=None
        Latch_SavedPatternMatch=None
        Latch_SavedBestPattern=None
        Latch_PatternOffsetX_Single=None
        Orientation="EMPTY"
        #warning - a lot of these are not used so take out
        for RotateIndex, RotationStage in enumerate(SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea):
            ImgToProcess=StackOrientations.copy()
            RotatedTemplate=SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea[RotateIndex]
            RotatedMask=SNunter_UserParams_Loaded.RotateSeries_MaskOfSN[RotateIndex]
            RotatedTemplate=cv2.resize(RotatedTemplate,(int(RotatedTemplate.shape[1]/SNunter_UserParams_Loaded.PatternSearchDivideImg),int(RotatedTemplate.shape[0]/SNunter_UserParams_Loaded.PatternSearchDivideImg)))
            RotatedMask=cv2.resize(RotatedMask,(int(RotatedMask.shape[1]/SNunter_UserParams_Loaded.PatternSearchDivideImg),int(RotatedMask.shape[0]/SNunter_UserParams_Loaded.PatternSearchDivideImg)))
            #cv2.imshow("imageclipTemplate", RotatedTemplate)
            #cv2.waitKey(0)
            #cv2.imshow("imageclip", RotatedMask)
            #cv2.waitKey(0)
            ch, h,w = RotatedTemplate.shape[::-1]
            #only two methods accept masks
            #(cv.TM_SQDIFF == match_method or match_method == cv.TM_CCORR_NORMED)
            #normalising methods make it harder to find a true match - for instance if looking at wrong side of note 
            res = cv2.matchTemplate(StackOrientations,RotatedTemplate,cv2.TM_CCORR_NORMED,None,None)#warning: only two methods work with mask
            #so we can visualise result - don't normalise the result

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            InputVal=max_val#max or min result depends on method used in match template
            #print("max value",InputVal)
            #latch max/min values
            if RotateIndex==0 or InputVal>Latch_MaxValue:
                Latch_MaxValue=InputVal
                Latch_MaxValueIndex=RotateIndex

                #draw initial rectangle
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(ImgToProcess,top_left, bottom_right, 90, 1)

                #draw mid point for debug
                #WARNING: Remember this is scaled! We will have to scale it up when extracting the real image
                MidPointX=int((top_left[0]+bottom_right[0])/2)
                MidPointY=int((top_left[1]+bottom_right[1])/2)
                cv2.circle(ImgToProcess, (MidPointX,MidPointY), int(5), (0,0,255),2)
                
                Latch_MaxValueTopLeft=top_left
                Latch_MaxValueTopRight=bottom_right
                #latch image to save out
                Latch_SavedPatternMatch=ImgToProcess
                Latch_SavedBestRotationPattern=SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea[RotateIndex].copy()
                Div=SNunter_UserParams_Loaded.PatternSearchDivideImg
                Latch_SavedBestPattern=StackOrientations_UserWave[int(top_left[1]*Div):int(bottom_right[1]*Div),int(top_left[0]*Div):int(bottom_right[0]*Div),:]

                #get what orientation search pattern was found - current method relies on orientations A B C D being stacked horizontally
                StackWidth=StackOrientations_UserWave.shape[1]/4
                #get position - need to multiply by user parameter for shrinking search stack image
                PatternPos=int(Latch_MaxValueTopLeft[0]*SNunter_UserParams_Loaded.PatternSearchDivideImg)
                #get index of orientation (1=a,2=b etc)
                OrientationIndex=np.floor(PatternPos/StackWidth)+1
                #get x & y offset for non stacked image
                Latch_ForS39MakerPatternOffsetX=PatternPos-(StackWidth*(OrientationIndex-1))
                
                Orientation="ERROR"
                if OrientationIndex==1:Orientation="A"
                if OrientationIndex==2:Orientation="B"
                if OrientationIndex==3:Orientation="C"
                if OrientationIndex==4:Orientation="D"
                #create debug output image of match space
                res_output=res.copy()
                cv2.normalize(res_output, res_output, 0, 255, cv2.NORM_MINMAX, -1 )

                #offset original midpoint of SN during user framing
                refPt=SNunter_UserParams_Loaded.UserSN_ROI#tight framing of serial number by user
                SNunter_UserParams_Loaded.MidPoint_user_SN_XY#midpoint of framing in original image
                Width=abs(refPt[0][0]-refPt[1][0])
                Height=abs(refPt[0][1]-refPt[1][1])

                Width_scaled=int(Width/SNunter_UserParams_Loaded.PatternSearchDivideImg)
                Height_scaled=int(Height/SNunter_UserParams_Loaded.PatternSearchDivideImg)

                UserROI_ScaledX=int(MidPointX-(Width_scaled/2))
                UserROI_ScaledY=int(MidPointY-(Height_scaled/2))
                TopLeft_ScaledUserROI=(UserROI_ScaledX,UserROI_ScaledY)#NB use this for s39 output
                UserROI_ScaledX=int(MidPointX+(Width_scaled/2))
                UserROI_ScaledY=int(MidPointY+(Height_scaled/2))
                BottomRight_ScaledUserROI=(UserROI_ScaledX,UserROI_ScaledY)

                #scaled user selected region for illustration. Scaling this up will allow us to crop correct region
                cv2.circle(ImgToProcess, TopLeft_ScaledUserROI, int(5), (255,255,255),2)
                cv2.circle(ImgToProcess, BottomRight_ScaledUserROI, int(5), (255,255,255),2)
                #cut out user selected region - generally tight framing the SN
                Div=SNunter_UserParams_Loaded.PatternSearchDivideImg
                Latch_SavedBestPattern_userROI=StackOrientations_UserWave[int(TopLeft_ScaledUserROI[1]*Div):int(BottomRight_ScaledUserROI[1]*Div),int(TopLeft_ScaledUserROI[0]*Div):int(BottomRight_ScaledUserROI[0]*Div),:]
                
                #have s39 details in one place
                Latch_ForS39MakerPatternOffsetY=int(TopLeft_ScaledUserROI[1]*Div/2)
                #to get X offset - we have far left of pattern search - this can be larger than user ROI so need to calculate offset from the far left of local search to far left of user search
                #we have that elsewhere - this is not the best way to do things and needs tidied up #TODO
                Latch_ForS39MakerPatternOffsetX=int(Latch_ForS39MakerPatternOffsetX+(w*SNunter_UserParams_Loaded.PatternSearchDivideImg/2)-(Width/2))
                Latch_ForS39MakerPatternWidth=Width#this is pretty daft = apologies for whoever is looking at this code (probably me)
                Latch_ForS39MakerPatternHeight=Height


                #now have to 
                # #get auto expanded area used to find features for pattern recognition
                # LocalPatternTopLeft=SNunter_UserParams_Loaded.RotateSeries_SquareLocalArea_ROI
                # #we want to find the top left corner of the area framed by the user
                # #at the moment all regions have common center - but this could change if we 
                # #introduce secondary regions to handle difficult subject matter

                # #get offset between original top left of local search region and user search region
                # UserTopLeft=refPt[0]#if mouse is dragged from left to right
                # #WARNING! NOT HANDLING SKEW! S39 generator does not seem to handle this so for now leave it
                # OffsetX=(UserTopLeft[0]-LocalPatternTopLeft[0])#divide by scaling 
                # OffsetY=(UserTopLeft[1]-LocalPatternTopLeft[1])

                # Offset_SN_AOI_TopLeftX=top_left[0]+OffsetX
                # Offset_SN_AOI_TopLeftY=top_left[1]+OffsetY

                # Offset_SN_AOI_BottomRightX=Offset_SN_AOI_TopLeftX+Width
                # Offset_SN_AOI_BottomRightY=Offset_SN_AOI_TopLeftY+Height

                
            
            cv2.rectangle(ImgToProcess,Latch_MaxValueTopLeft, Latch_MaxValueTopRight, 255, 2)
            
            #cv2.imshow("imageclip", res)
            #cv2.waitKey(0)
            #cv2.imshow("imageclip",cv2.resize(ImgToProcess,(800,800)))
            #cv2.waitKey(0)

        
        #save debug images out to folder
        BestMatchImg_filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(DatIndex) + "_" + Orientation + "_BestMatch.jpg"
        SearchPattern_filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(DatIndex) + "_" + Orientation + "_SearchPattern.jpg"
        PatternMatch_filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(DatIndex) + "_" + Orientation + "_PatternMatch.jpg"
        Latch_SavedBestPattern_userROI_Filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(DatIndex) + "_" + Orientation + "_UserROI_PatternMatch.jpg"
        res_output_filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(DatIndex) + "_" + Orientation + "_ResMap.jpg"
        StackOrientations_UserWave_filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(DatIndex) + "_" + Orientation + "_userWave.jpg"
        

        cv2.imwrite(StackOrientations_UserWave_filename,StackOrientations_UserWave)
        cv2.imwrite(BestMatchImg_filename,Latch_SavedPatternMatch)
        cv2.imwrite(SearchPattern_filename,Latch_SavedBestPattern)
        cv2.imwrite(Latch_SavedBestPattern_userROI_Filename,Latch_SavedBestPattern_userROI)
        cv2.imwrite(PatternMatch_filename,Latch_SavedBestRotationPattern)
        cv2.imwrite(res_output_filename,res_output)

        #save out s39 for A and D - we will need special rules for B and C
        s39Maker = S39Maker()#new instance of s39 extractor
        s39Maker.files=[DatFile]
        s39Maker.outputDirectory=SNunter_UserParams_Loaded.OutputFolder +"\\"
        s39Maker.start()
        #populate s39 area - need different logic if note is in flipped orientation (B/C)
        RealS39_Image=None

        #explaination of inputs for get39image
        #input s39 maker object - we might use different versions
        #input wave - this is the wave the user selected at the region selection (r/g/b)
        #pattern width - this is the width of the user rectangle drawing initially. The image the user has drawn on
        #has had aspect ratio changed and is scaled to fit on screen - so has to be fiddled with
        #Similarly for height  - this is why we are dividing by 2 to undo aspect ratio fix
        #and same goes for x/y offset - this is area found by the search pattern - but the search pattern is larger generally than the
        #area we want - so need to calculate this offset then fix any scaling, and any aspect ratio
        
        if Orientation=="A":
            RealS39_Image=Get39Image(s39Maker,"front",SNunter_UserParams_Loaded.s39_wave,Latch_ForS39MakerPatternWidth,int(Latch_ForS39MakerPatternHeight/2),Latch_ForS39MakerPatternOffsetX,Latch_ForS39MakerPatternOffsetY,SNunter_UserParams_Loaded.s39_validation,FixMM8_AspectRatio=False)
            pass
        #_3DVisLabLib.ImageViewer_Quickv2(Get39Image(s39Maker,"back",SNunter_UserParams_Loaded.s39_wave,Latch_ForS39MakerPatternWidth,int(Latch_ForS39MakerPatternHeight/2),Latch_ForS39MakerPatternOffsetX,int(refPt[0][1]/2),SNunter_UserParams_Loaded.s39_validation,FixMM8_AspectRatio=False),0,True,True)
        if Orientation=="D":
            RealS39_Image=Get39Image(s39Maker,"back",SNunter_UserParams_Loaded.s39_wave,Latch_ForS39MakerPatternWidth,int(Latch_ForS39MakerPatternHeight/2),Latch_ForS39MakerPatternOffsetX,Latch_ForS39MakerPatternOffsetY,SNunter_UserParams_Loaded.s39_validation,FixMM8_AspectRatio=False)
            pass

        #orientations B and C require a rotation of 180 to match A and D orientation
        if Orientation=="B" or Orientation=="C":
            
            Latch_ForS39MakerPatternOffsetY=SNunter_UserParams_Loaded.MM8_fullResY-Latch_ForS39MakerPatternOffsetY-int(Latch_ForS39MakerPatternHeight/2)
            Latch_ForS39MakerPatternOffsetX=SNunter_UserParams_Loaded.MM8_fullResX-Latch_ForS39MakerPatternOffsetX-Latch_ForS39MakerPatternWidth
            #make rotation matrix
            #M = cv2.getRotationMatrix2D((int(SNunter_UserParams_Loaded.MM8_fullResX/2), int(SNunter_UserParams_Loaded.MM8_fullResY/2)), 180, 1.0)
            #make vector 
            #TopLeftVector=np.array()
            if Orientation=="B":
                RealS39_Image=Get39Image(s39Maker,"front",SNunter_UserParams_Loaded.s39_wave,Latch_ForS39MakerPatternWidth,int(Latch_ForS39MakerPatternHeight/2),Latch_ForS39MakerPatternOffsetX,Latch_ForS39MakerPatternOffsetY,SNunter_UserParams_Loaded.s39_validation,Rotate180_=True,FixMM8_AspectRatio=False)
            if Orientation=="C":
                RealS39_Image=Get39Image(s39Maker,"back",SNunter_UserParams_Loaded.s39_wave,Latch_ForS39MakerPatternWidth,int(Latch_ForS39MakerPatternHeight/2),Latch_ForS39MakerPatternOffsetX,Latch_ForS39MakerPatternOffsetY,SNunter_UserParams_Loaded.s39_validation,Rotate180_=True,FixMM8_AspectRatio=False)
                pass
            
        if RealS39_Image is not None:
            RealS39_Image_filename=SNunter_UserParams_Loaded.OutputFolder +"\\00" + str(DatIndex) + "_" + Orientation + "_S39ExtractionImg.jpg"
            cv2.imwrite(RealS39_Image_filename,RealS39_Image)

        

        
#instantiate user params class which we will load during user interactivity
SNunter_UserParams_toLoad=SNunter_UserParams()

#load in user folders
SNunter_UserParams_toLoad.InputFolder=r"D:\Bax\NCR_2022_03_29\NCR\Currencies\01_MM8_DC\SR_MALAYSIA_MM8_DC\KP00010010\MM8\5 MYR B"
#SNunter_UserParams_toLoad.InputFolder=r"E:\NCR\Currencies\Bangladesh_SR2800\Bangladesh\SR DC\MM8\1000\2008"
#set output folder
SNunter_UserParams_toLoad.OutputFolder=r"C:\Working\FindIMage_In_Dat\OutputFindPattern"
print("Please check output folders can be deleted:",SNunter_UserParams_toLoad.OutputFolder)
Response=_3DVisLabLib.yesno("Continue?")
if Response==True:
    #delete output folder
    _3DVisLabLib.DeleteFiles_RecreateFolder(SNunter_UserParams_toLoad.OutputFolder)


#get all files in folders recursively
print("Looking in",SNunter_UserParams_toLoad.InputFolder,"for .dat files")
List_all_Files=GetAllFilesInFolder_Recursive(SNunter_UserParams_toLoad.InputFolder)
#filter out non .dats
List_all_Dats=GetList_Of_ImagesInList(List_all_Files,".dat")
print(len(List_all_Dats),".dat files found")
#randomise dat files
#load images in random order for testing
print("Randomising input order, set of",int(min(SNunter_UserParams_toLoad.SubSetOfData,len(List_all_Dats))),"MM8 files")
randomdict=dict()
for Index, ImagePath in enumerate(List_all_Dats):
    randomdict[ImagePath]=Index
#user may have specified a subset of data
List_all_Dats=[]
while (len(List_all_Dats)<SNunter_UserParams_toLoad.SubSetOfData) and (len(randomdict)>0):
    randomchoice_img=random.choice(list(randomdict.keys()))
    List_all_Dats.append(randomchoice_img)
    del randomdict[randomchoice_img]

PrepareMultiProcessing(SNunter_UserParams_toLoad,List_all_Dats)
ssss
#UI to select part of note, then create details necessary to find SN in other notes 
SNunter_UserParams_Loaded=SN_HuntLoop(SNunter_UserParams_toLoad,List_all_Dats[0])#use first .dat file - even better if we can find least skewed one

#for each mm8 file try to find SN area - note can be in any orientation and true orientation must
#be recorded in the S39 file 
SearchMM8_forSN_Location(SNunter_UserParams_Loaded,List_all_Dats)