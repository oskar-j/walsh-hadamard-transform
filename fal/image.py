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

    def _load_bmp_header(self, file):
        header = file.read(struct.calcsize(HEADER_FORMAT))
        (self.__signature, self.__size, ignored, ignored, self.__offset, self.__headerSize,
            self.__width, self.__height, self.__planes, self.__bpp, self.__compression,
            self.__sizeOfData, self.__horizontalRes, self.__verticalRes, ignored, ignored) =\
            struct.unpack(HEADER_FORMAT, header)
        if (self.__signature, self.__planes, self.__bpp, self.__compression) != ('BM', 1, 24, 0):
            raise UnsupportedFileFormatError
#        print 'Size of data: ' + str(self.__sizeOfData)

    def _load_bmp_data(self, file):
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
            self._load_bmp_header(file)
            self._load_bmp_data(file)

    def _write_header(self, file):
        ignored = 0
        header = (
            self.__signature, self.__size, ignored, ignored, self.__offset, self.__headerSize,
            self.__width, self.__height, self.__planes, self.__bpp, self.__compression,
            self.__sizeOfData, self.__horizontalRes, self.__verticalRes, ignored, ignored
        )
        raw_header = struct.pack(HEADER_FORMAT, *header)
        file.write(raw_header)

    def _write_bmp_data(self, file):
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
            self._write_header(file)
            self._write_bmp_data(file)

    def get_dimensions(self):
        return self.__width, self.__height

    def set_dimensions(self, width, height):
        self.__width = width
        self.__height = height
        self.__sizeOfData = width * height * 3

    def get_raw_data(self):
        return self.__rawData

    def set_raw_data(self, new_data):
        self.__rawData = new_data


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

    def _read_header(self, file):
        self.__width, self.__height = struct.unpack('<II', file.read(struct.calcsize('<II')))
        self.__yDescription = struct.unpack('<HHH', file.read(struct.calcsize('<HHH')))
        self.__cbDescription = struct.unpack('<HHH', file.read(struct.calcsize('<HHH')))
        self.__crDescription = struct.unpack('<HHH', file.read(struct.calcsize('<HHH')))

    @staticmethod
    def _read_blocks(file, description):
        blocks = []
        original_block_size = description[0]
        packed_block_size = description[1]
        number_of_blocks = description[2]
        block_length = packed_block_size * packed_block_size
        pattern = '<' + 'h' * block_length
        for i in xrange(0, number_of_blocks):
            data = struct.unpack(pattern, file.read(struct.calcsize(pattern)))
            block = np.matrix(np.zeros((original_block_size, original_block_size)))
            block[0:packed_block_size, 0:packed_block_size] = np.matrix(data).reshape(packed_block_size,
                                                                                      packed_block_size)
            blocks.append(block)
        return blocks

    @staticmethod
    def load(filename):
        f = sys.stdin
        if filename and isinstance(filename, str):
            f = open(filename, 'rb')
        with f as file:
            image = CustomizableImage()
            image._read_header(file)
            if image.__yDescription[2] > 0:
                image.__yData = image._read_blocks(file, image.__yDescription)
            if image.__cbDescription[2] > 0:
                image.__cbData = image._read_blocks(file, image.__cbDescription)
            if image.__crDescription[2] > 0:
                image.__crData = image._read_blocks(file, image.__crDescription)
            return image

    def get_y_data(self):
        return self.__yData

    def get_cb_data(self):
        return self.__cbData

    def get_cr_data(self):
        return self.__crData

    def get_dimensions(self):
        return self.__width, self.__height

    def set_dimensions(self, width, height):
        self.__width = width
        self.__height = height

    def set_descriptions(self, y_description, cb_description, cr_description):
        self.__yDescription = y_description
        self.__cbDescription = cb_description
        self.__crDescription = cr_description

    def set_data(self, y_data, cb_data, cr_data):
        y_packed_block_size = self.__yDescription[1]
        self.__yData = map(lambda x: x[0:y_packed_block_size, 0:y_packed_block_size], y_data)
        cb_packed_block_size = self.__cbDescription[1]
        self.__cbData = map(lambda x: x[0:cb_packed_block_size, 0:cb_packed_block_size], cb_data)
        cr_packed_block_size = self.__crDescription[1]
        self.__crData = map(lambda x: x[0:cr_packed_block_size, 0:cr_packed_block_size], cr_data)

    def _write_header(self, file):
        file.write(struct.pack('<II', self.__width, self.__height))
        file.write(struct.pack('<HHH', *self.__yDescription))
        file.write(struct.pack('<HHH', *self.__cbDescription))
        file.write(struct.pack('<HHH', *self.__crDescription))

    @staticmethod
    def _write_blocks(file, blocks):
        for block in blocks:
            data = np.array(block).reshape(-1).tolist()
            file.write(struct.pack('<' + 'h' * len(data), *data))

    def save(self, filename):
        with open(filename, 'wb') as file:
            self._write_header(file)
            if self.__yData:
                self._write_blocks(file, self.__yData)
            if self.__cbData:
                self._write_blocks(file, self.__cbData)
            if self.__crData:
                self._write_blocks(file, self.__crData)

