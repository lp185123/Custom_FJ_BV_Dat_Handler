class ImageExtractor:
    HEIGHT_OFFSET = 4
    DATA_WIDTH = 8
    HEADER_WIDTH = 48
    SIZE_OFFSET = 0
    TYPE_OFFSET = 1
    WAVELENGTH_OFFSET = 2
    WIDTH_OFFSET = 3
    SRU_HEADER_WIDTH = 64
    SNR_START = 4392
    SNR_END = 4584

    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    CONDITIONS = ['Genuine', 'Unclear', 'Suspect Counterfeit', 'Damaged', 'Unknown Category', 'Unknown Category', 'Unknown Category', 'Unknown Category', 'Damage (Tape)', 'Tear', 'Hole', 'Damaged (Soil)', 'Damage (Folded Corner)', 'DSP Validation Result Delay', 'Long Edge Too Short', 'Unknown Category']
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

        try:
            with open(file, 'rb') as openedFile:
                self.hexstring = openedFile.read().hex()
        except FileNotFoundError:
            if ImageExtractor.DEBUG:
                print('File named ' + sys.argv[1] + ' not found')
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
            self.notes.append(ExtractedNote(self.hexstring[self.index-blockSize:self.index], len(self.notes) + 1))
        elif blockType == 12:
            # Each character of the SNR is stored up to 3 times
            # It looks like
            offset = ((self.index-blockSize) + ImageExtractor.SNR_START)
            limit = ((self.index-blockSize) + ImageExtractor.SNR_END)
            while offset < limit:
                possibilities = []
                # Loop through each
                for x in range(0,3):
                    if self.hexstring[offset:offset+4] != "0000":
                        possibilities.append(self.hexstring[offset:offset+4])
                    offset += 4

                # If the snr read of this character has any possible result
                if len(possibilities) >= 1:
                    if len(possibilities) >= 2:
                        # If first two possibilities disagree, remove them both and check 3rd
                        if possibilities[0] != possibilities[1]:
                            if len(possibilities) >= 3:
                                # Check if the 3rd value matches either of first two,
                                # and remove if not
                                if possibilities[2] != possibilities[1] and possibilities[2] != possibilities[0]:
                                    possibilities.remove(possibilities[2])
                            possibilities.remove(possibilities[1])
                            possibilities.remove(possibilities[0])
                        else:
                            # If the first two possibilities agree, we use their value
                            possibilities = [possibilities[0]]
                    # If we are still left with a possible character of the snr, add
                    # it to the current note
                    if len(possibilities):
                        self.notes[len(self.notes) - 1].snr += bytes.fromhex(possibilities[0]).decode('utf-16 be')
                    else:
                        # Add a question mark since validation is inconclusive
                        self.notes[len(self.notes) - 1].snr += '?'
                    # note that we don't add a question mark in the case that
                    # there were no interpreted values to begin with
        elif blockType in ImageExtractor.IMAGE_TYPES:
            self.notes[len(self.notes) - 1].images.append(
                ExtractedImage(
                    self.hexstring[self.index-blockSize:self.index],
                    blockType,
                    self.index - blockSize,
                    blockSize / 2,
                    self.notes[len(self.notes) - 1]
                )
            )

        return True

class ExtractedNote:

    def __init__(self, data, id):
        self.id = id
        self.snr = ""
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

            if data[52:54] in ImageExtractor.ERRORS:
                self.condition = ImageExtractor.ERRORS[data[52:54]]
            else:
                self.condition = data[52:54] + " Error"

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
            print('Unexpected wavelength detected: ' + wavelength)

        self.info = ''

        if self.type in ImageExtractor.BLOCK_TYPES:
            self.info = ImageExtractor.BLOCK_TYPES[self.type]
        else:
            self.info = 'Unexpected block type'
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

# # GBxx .DATs
# images = ImageExtractor('01_5_A.dat', False)
# filteredImages = images.getByType(4)
#
# # SRU .DATs
# images = ImageExtractor('0005.dat', True)
# images = ImageExtractor('SR0007_from_DaveS.dat', True)
# filteredImages = images.getByType('SRU MM8 image')

# Use this to only get images of particular wavelengths
# filteredImages = images.getByWavelength('A1')

# Use this to only get images of a particular type, refer to type list
# filteredImages = images.getByType(4)

# Filter by type and wavelength
# filteredImages = images.filter(8, 'A1')
#

# images = ImageExtractor('M8_211207_150339.dat', True)
# filteredImages = images.getByWavelength('A1')
#
# image = filteredImages[1]

# notes = ImageExtractor('SR0007_from_DaveS.dat').notes
# for note in notes:
#     print("Condition: " + note.condition + ", SNR: " + note.snr)
