import sys, os, shutil, binascii, math
from datetime import datetime
import time


    #1400#width
    #1404#height
    #3376 where image starts

# def main():

    #s39Extractor = S39Extractor()
    #s39Extractor.start()

    # s39Maker = S39Maker()
    # print('What wave would you like? (type "r", "g", or "b")')
    # message = input()
    # if message == 'r':
    #     s39Maker.wave = 'red'
    # elif message == 'g':
    #     s39Maker.wave = 'green'
    # else:
    #     s39Maker.wave = 'blue'
    # print('Using ' + s39Maker.wave)
    # print('Front or back? (type "f" or "b", or leave empty and use front')
    # message = input()
    # if message == 'b':
    #     s39Maker.side = 'back'
    # else:
    #     s39Maker.side = 'front'
    # print('Using ' + s39Maker.side)
    # s39Maker.validation = ''
    # while len(s39Maker.validation) != 8:
    #     print('Please enter the validation code you will use for these S39 (should be 8 hex digits long) or leave empty to use Euro:')
    #     s39Maker.validation = input()
    #     if s39Maker.validation == '':
    #         s39Maker.validation = '80080103'
    # print('Using ' + s39Maker.validation)
    # print('Please enter the x position of the rectangle:')
    # message = 'not a digit'
    # while not message.isdigit():
    #     message = input()
    # s39Maker.x = int(message)
    # print('Using ' + str(s39Maker.x) + ' as the x position')
    # print('Please enter the y position of the rectangle:')
    # message = 'not a digit'
    # while not message.isdigit():
    #     message = input()
    # s39Maker.y = int(message)
    # print('Using ' + str(s39Maker.y) + ' as the y position')
    # print('Please enter the width of the rectangle:')
    # message = 'not a digit'
    # while not message.isdigit():
    #     message = input()
    #     if message.isdigit():
    #         numMessage = int(message)
    #         if not numMessage % 8 == 0:
    #             print('The width must be divisible by 8. Try ' + str(numMessage - (numMessage % 8)) + ' or ' + str((numMessage - (numMessage % 8)) + 8))
    #             message = 'not a number'
    # s39Maker.width = int(message)
    # print('Using ' + str(s39Maker.width) + ' as the width')
    # print('Please enter the height of the rectangle:')
    # message = 'not a digit'
    # while not message.isdigit():
    #     message = input()
    #     if message.isdigit():
    #         numMessage = int(message)
    #         if not numMessage % 8 == 0:
    #             print('The height must be divisible by 8. Try ' + str(numMessage - (numMessage % 8)) + ' or ' + str((numMessage - (numMessage % 8)) + 8))
    #             message = 'not a number'
    # s39Maker.height = int(message)
    # print('Using ' + str(s39Maker.height) + ' as the height')
    #
    # s39Maker.start()
    # print('Process complete. You can find your S39 in the S39 folder.')

# if __name__ == "__main__":
#     try:
#         main()
#         os.system('pause')
#     except Exception as e:
#         print(e)
#         os.system('pause')

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
