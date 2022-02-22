import sys, os, shutil, binascii
from datetime import datetime

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

        #The following code is only for when we have a gui

        # elif blockType in ImageExtractor.IMAGE_TYPES:
        #     self.notes[len(self.notes) - 1].images.append(
        #         ExtractedImage(
        #             self.hexstring[self.index-blockSize:self.index],
        #             blockType,
        #             self.index - blockSize,
        #             blockSize / 2,
        #             self.notes[len(self.notes) - 1]
        #         )
        #     )

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

        for noteId in arrayOfNoteIds:
            badNotes.append(self.notes.pop(noteId))

        clean = ""
        for note in self.notes:
            clean += note.print()

        dirty = ""
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

class ExtractedImage:

    def __init__(self, data, type, offset, recordSize, note):
        self.note = note
        self.type = type

        wavelength = ImageExtractor.littleEndianHexToBE(
            data[ImageExtractor.WAVELENGTH_OFFSET * ImageExtractor.DATA_WIDTH :
                (ImageExtractor.WAVELENGTH_OFFSET * ImageExtractor.DATA_WIDTH) +
                ImageExtractor.DATA_WIDTH]
        )

        if wavelength in ImageExtractor.WAVE_DICTIONARY:
            self.waveDesignation = ImageExtractor.WAVE_DICTIONARY[wavelength]
        else:
            # Set to 'None' when we don't recognise the wavelength
            self.waveDesignation = ImageExtractor.WAVE_DICTIONARY['00000000']
            if ImageExtractor.DEBUG:
                print('Unexpected wavelength detected: ' + wavelength)

        self.info = ''

        if self.type in ImageExtractor.BLOCK_TYPES:
            self.info = ImageExtractor.BLOCK_TYPES[self.type]
        else:
            self.info = 'Unexpected block type'
            if ImageExtractor.DEBUG:
                print('Unexpected block type: ' + self.type)

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
                    print('outputting')
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


# ---ImageMerger---
#
# merger = ImageMerger()
# merger.shouldPrintProgress = True
# merger.arrayOfNoteCounts = [2,3,2]
# merger.shouldMergeExtra = False
# merger.start()
#
# ---ImageExtractor---
#
# imageExtractor = ImageExtractor('0005.dat')
#
# noteIds = [0,1,2]
#
# imageExtractor.clean(noteIds, 'cool.dat', 'ugly.dat')
