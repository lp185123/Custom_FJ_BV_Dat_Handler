import sys, os, shutil, binascii, hashlib

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
        if self.hexstring[0:6] == '535255':
            self.fileHeader = ImageExtractor.SRU_HEADER_WIDTH
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

        if blockType == 4 or blockType == 14:
            md5_hash = hashlib.md5()
            md5_hash.update(binascii.unhexlify(self.hexstring[self.index-blockSize:self.index]))
            digest = md5_hash.hexdigest()
            self.notes[len(self.notes) - 1].hash = digest

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

class ExtractedNote:

    def __init__(self, data, id, offset, extractor):
        self.extractor = extractor
        self.blockSize = 0
        self.offset = offset
        self.id = id
        self.images = []
        self.hash = ''
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

        self.skew = int(data[1808:1810], 16) + int(data[1820:1822], 16) - 20

        #self.skew = int(data[32:34], 16);
        #print("Skew: " + str(self.skew))

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
        #return self.extractor.hexstring[self.offset:self.offset + 52] + "01" + self.extractor.hexstring[self.offset + 54:self.offset + self.blockSize]
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

class ImageHash:

    def __init__(self, hash, path, index):
        self.hash = hash
        self.path = path
        self.index = index

def ensureDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():

    answer = input("Would you like the tool to only record the first duplicate it finds in each dat? (y/n)\n")
    if answer == 'y' or answer == 'Y':
    	stopAtFirst = True
    else:
    	stopAtFirst = False

    allFiles = []
    for root, dirs, files in os.walk(r"E:\NCR\SR_Generations\Sprint\Russia\DC\ForStd_Gen\ForGen"):
        for name in files:
            if name[-4:len(name)] == '.dat':
                allFiles.append(os.path.join(root, name))

    allFiles.sort()

    imageHashes = []

    totalIndex = 0

    printable = 'Duplicates'

    DuplicateFiles_dict=dict()

    if stopAtFirst:
        printable = printable + ' (only showing first duplicate in each dat)'

    printable = printable + ':\n'

    for individualFile in allFiles:

        imageExtractor = ImageExtractor(individualFile)
        index = 0
        foundDuplicate = False
        for note in imageExtractor.notes:

            for imageHash in imageHashes:
                if imageHash.hash == note.hash:
                    if not (stopAtFirst and foundDuplicate):
                        printable = printable + '\n'
                        printable = printable + os.path.abspath(imageHash.path) + ', note ' + str(imageHash.index + 1) + '\n'
                        printable = printable + os.path.abspath(individualFile) + ', note ' + str(index + 1) + '\n'
                        DuplicateFiles_dict[imageHash.path]=imageHash.path
                        DuplicateFiles_dict[individualFile]=individualFile
                    foundDuplicate = True

            imageHashes.append(ImageHash(note.hash, individualFile, index))
            index = index + 1

        totalIndex += 1

        print(str(totalIndex / len(allFiles) * 100) + "%")
    
    print(printable)
    def ReadyToDelete(Delete):
        for DatFile in DuplicateFiles_dict:
            
            if DatFile.split("\\")[8]=="Test":
                print("to be removed ",DatFile)
                if Delete==True:
                    if os.path.exists(DatFile)==True:
                        os.remove(DatFile)



    ReadyToDelete(False)
    answer = input("press Y to start deleting duplicates (y/n)\n")
    if answer!="y":
        raise Exception("User stopped process")
    #start deleting files
    ReadyToDelete(True)
    
    with open('Duplicates.txt', "w+") as output:
        output.write(printable)
    print("\nDuplicates saved to Duplicates.txt")
    print("\nProcessing of DuplicateChecker complete")
    os.system('pause')

if __name__ == "__main__":
    #try:
    main()
    #except Exception as e:
    #    print(e)
    #    os.system('pause')
