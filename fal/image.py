import sys
import numpy as np

import struct

def align(x, a):
    return (((x - 1) // a) + 1) * a

class UnsupportedFileFormatError(Exception):
    pass

HEADER_FORMAT = '<2sIHHIIIIHHIIIIII'

class BMPImage:

    def __init__(self):
        self.__signature = 'BM'
        self.__size = 0
        self.__offset = 54
        self.__headerSize = 40
        self.__width = 0
        self.__height = 0
        self.__planes = 1
        self.__bpp = 24
        self.__compression = 0
        self.__sizeOfData = 0
        self.__horizontalRes = 2835
        self.__verticalRes = 2835
        self.__rawData = None

    def __loadBmpHeader(self, file):
        header = file.read(struct.calcsize(HEADER_FORMAT))
        (self.__signature, self.__size, ignored, ignored, self.__offset, self.__headerSize,
            self.__width, self.__height, self.__planes, self.__bpp, self.__compression,
            self.__sizeOfData, self.__horizontalRes, self.__verticalRes, ignored, ignored) =\
            struct.unpack(HEADER_FORMAT, header)
        if (self.__signature, self.__planes, self.__bpp, self.__compression) != ('BM', 1, 24, 0):
            raise UnsupportedFileFormatError
#        print 'Size of data: ' + str(self.__sizeOfData)

    def __loadBmpData(self, file):
        file.seek(self.__offset)
        self.__rawData = []
        for j in xrange(0, self.__height):
            line = file.read(align(self.__width * 3, 4))
            for i in xrange(0, self.__width * 3, 3):
                self.__rawData.append(struct.unpack_from('<BBB', line, i))

    def load(self, filename):
        f = sys.stdin
        if filename and isinstance(filename, str):
            f = open(filename, 'rb')
        with f as file:
            self.__loadBmpHeader(file)
            self.__loadBmpData(file)

    def __writeHeader(self, file):
        ignored = 0
        header = (
            self.__signature, self.__size, ignored, ignored, self.__offset, self.__headerSize,
            self.__width, self.__height, self.__planes, self.__bpp, self.__compression,
            self.__sizeOfData, self.__horizontalRes, self.__verticalRes, ignored, ignored
        )
        rawHeader = struct.pack(HEADER_FORMAT, *header)
        file.write(rawHeader)

    def __writeBmpData(self, file):
        padding = align(self.__width * 3, 4) - self.__width * 3
        print 'Padding: ' + str(padding)
        counter = 0
        for v in self.__rawData:
            r, g, b = v
            file.write(struct.pack('<BBB', r, g, b))
            counter += 1
            if counter == self.__width:
                counter = 0
                for i in xrange(0, padding):
                    file.write(struct.pack('<B', 0))

    def save(self, filename):
        with open(filename, 'wb') as file:
            self.__writeHeader(file)
            self.__writeBmpData(file)

    def getDimensions(self):
        return self.__width, self.__height

    def setDimensions(self, width, height):
        self.__width = width
        self.__height = height
        self.__sizeOfData = width * height * 3

    def getRawData(self):
        return self.__rawData

    def setRawData(self, newData):
        self.__rawData = newData

class CustomizableImage:

    def __init__(self):
        self.__width = 0
        self.__height = 0

        # (originalBlockSize, packedBlockSize, numberOfBlocks)
        self.__yDescription = None
        self.__cbDescription = None
        self.__crDescription = None

        self.__yData = None
        self.__cbData = None
        self.__crData = None

    def __readHeader(self, file):
        self.__width, self.__height = struct.unpack('<II', file.read(struct.calcsize('<II')))
        self.__yDescription = struct.unpack('<HHH', file.read(struct.calcsize('<HHH')))
        self.__cbDescription = struct.unpack('<HHH', file.read(struct.calcsize('<HHH')))
        self.__crDescription = struct.unpack('<HHH', file.read(struct.calcsize('<HHH')))

    def __readBlocks(self, file, description):
        blocks = []
        originalBlockSize = description[0]
        packedBlockSize = description[1]
        numberOfBlocks = description[2]
        blockLength = packedBlockSize * packedBlockSize
        pattern = '<' + 'h' * blockLength
        for i in xrange(0, numberOfBlocks):
            data = struct.unpack(pattern, file.read(struct.calcsize(pattern)))
            block = np.matrix(np.zeros((originalBlockSize, originalBlockSize)))
            block[0:packedBlockSize, 0:packedBlockSize] = np.matrix(data).reshape(packedBlockSize, packedBlockSize)
            blocks.append(block)
        return blocks

    @staticmethod
    def load(filename):
        f = sys.stdin
        if filename and isinstance(filename, str):
            f = open(filename, 'rb')
        with f as file:
            image = CustomizableImage()
            image.__readHeader(file)
            if image.__yDescription[2] > 0:
                image.__yData = image.__readBlocks(file, image.__yDescription)
            if image.__cbDescription[2] > 0:
                image.__cbData = image.__readBlocks(file, image.__cbDescription)
            if image.__crDescription[2] > 0:
                image.__crData = image.__readBlocks(file, image.__crDescription)
            return image

    def getYData(self):
        return self.__yData

    def getCbData(self):
        return self.__cbData

    def getCrData(self):
        return self.__crData

    def getDimensions(self):
        return self.__width, self.__height

    def setDimensions(self, width, height):
        self.__width = width
        self.__height = height

    def setDescriptions(self, yDescription, cbDescription, crDescription):
        self.__yDescription = yDescription
        self.__cbDescription = cbDescription
        self.__crDescription = crDescription

    def setData(self, yData, cbData, crData):
        yPackedBlockSize = self.__yDescription[1]
        self.__yData = map(lambda x: x[0:yPackedBlockSize, 0:yPackedBlockSize], yData)
        cbPackedBlockSize = self.__cbDescription[1]
        self.__cbData = map(lambda x: x[0:cbPackedBlockSize, 0:cbPackedBlockSize], cbData)
        crPackedBlockSize = self.__crDescription[1]
        self.__crData = map(lambda x: x[0:crPackedBlockSize, 0:crPackedBlockSize], crData)

    def __writeHeader(self, file):
        file.write(struct.pack('<II', self.__width, self.__height))
        file.write(struct.pack('<HHH', *self.__yDescription))
        file.write(struct.pack('<HHH', *self.__cbDescription))
        file.write(struct.pack('<HHH', *self.__crDescription))

    def __writeBlocks(self, file, blocks):
        for block in blocks:
            data = np.array(block).reshape(-1).tolist()
            file.write(struct.pack('<' + 'h' * len(data), *data))

    def save(self, filename):
        with open(filename, 'wb') as file:
            self.__writeHeader(file)
            if self.__yData:
                self.__writeBlocks(file, self.__yData)
            if self.__cbData:
                self.__writeBlocks(file, self.__cbData)
            if self.__crData:
                self.__writeBlocks(file, self.__crData)

