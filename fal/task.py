import numpy as np
from colors import RgbColorModel, YCbCrColorModel
from image import BMPImage, CustomizableImage
from transforms import WalshHadamardTransform

class Task:

    def __init__(self):
        self.__input = None
        self.__output = None
        self.__action = None
        self.__coeff_removal = None

    def __getPaddingSize(self, x, a):
        return ((x - 1) // a + 1) * a - x

    def __slice(self, x, width, height, blockSize):
        x = np.matrix(x).reshape(height, width)
        print 'Shape: ' + str(x.shape)

        heightPadding = self.__getPaddingSize(height, blockSize)
        print 'Height pad: ' + str(heightPadding)
        x = np.concatenate((x, np.zeros((heightPadding, x.shape[1]))), 0)

        widthPadding = self.__getPaddingSize(width, blockSize)
        print 'Width pad: ' + str(widthPadding)
        x = np.concatenate((x, np.zeros((x.shape[0], widthPadding))), 1)

        blocks = []
        for row in np.vsplit(x, x.shape[0] / blockSize):
            blocks.extend(np.hsplit(row, row.shape[1] / blockSize))

        print 'Number of blocks: ' + str(len(blocks))
        print blocks[0].shape

        return blocks

    def __merge(self, blocks, width, height):
        blockHeight, blockWidth = blocks[0].shape
        blocksPerRow = (width - 1) // blockWidth + 1
        blocksPerColumn = len(blocks) / blocksPerRow
        rows = [np.hstack(blocks[i * blocksPerRow : (i + 1) * blocksPerRow])
               for i in xrange(0, blocksPerColumn)]
        return np.vstack(rows)[0:height, 0:width]

    def compress(self):
        print 'Action: compress'

        bmpImage = BMPImage()
        try:
            bmpImage.load(self.__input)
        except IOError as e:
            print 'IO Error: ' + str(e)
            return

        width, height = bmpImage.getDimensions()
        data = bmpImage.getRawData()

        color = RgbColorModel()
        y, cb, cr = zip(*map(color.getYCbCr, data))

        yBlockSize = 8
        cbBlockSize = 16
        crBlockSize = 16

        y = self.__slice(y, width, height, yBlockSize)
        cb = self.__slice(cb, width, height, cbBlockSize)
        cr = self.__slice(cr, width, height, crBlockSize)

        transform = WalshHadamardTransform(self.__coeff_removal)
        spectralY = transform.transformSequence(y)
        spectralCb = transform.transformSequence(cb)
        spectralCr = transform.transformSequence(cr)

        customizableImage = CustomizableImage()
        customizableImage.setDimensions(width, height)
        customizableImage.setDescriptions(
            (yBlockSize, 4, len(spectralY)),
            (cbBlockSize, 4, len(spectralCb)),
            (crBlockSize, 4, len(spectralCr))
        )
        customizableImage.setData(spectralY, spectralCb, spectralCr)
        customizableImage.save(self.__output)

    def extract(self):
        print 'Action: extract'

        customizableImage = CustomizableImage.load(self.__input)
        width, height = customizableImage.getDimensions()

        transform = WalshHadamardTransform()

        spectralY = customizableImage.getYData()
        if spectralY:
            y = transform.inverseTransformSequence(spectralY)
            y = self.__merge(y, width, height)
            y = np.array(y).reshape(-1).tolist()
            print len(y)
        else:
            y = [0] * (width * height)

        spectralCb = customizableImage.getCbData()
        if spectralCb:
            cb = transform.inverseTransformSequence(spectralCb)
            cb = self.__merge(cb, width, height)
            cb = np.array(cb).reshape(-1).tolist()
            print len(cb)
        else:
            cb = [128] * (width * height)

        spectralCr = customizableImage.getCrData()
        if spectralCr:
            cr = transform.inverseTransformSequence(spectralCr)
            cr = self.__merge(cr, width, height)
            cr = np.array(cr).reshape(-1).tolist()
            print len(cr)
        else:
            cr = [128] * (width * height)

        pixels = map(YCbCrColorModel().getRGB, zip(y, cb, cr))

        bmpImage = BMPImage()
        bmpImage.setDimensions(width, height)
        bmpImage.setRawData(pixels)
        bmpImage.save(self.__output)

    __actions = {'compress': compress, 'extract': extract}

    def withInput(self, input):
        self.__input = input
        return self

    def withCoeffRemoval(self, coeff):
        self.__coeff_removal = coeff
        return self

    def withOutput(self, output):
        self.__output = output
        return self

    def withAction(self, action):
        if action in Task.__actions:
            self.__action = Task.__actions[action]
        return self

    def run(self):
        print 'Input: ' + str(self.__input)
        print 'Output: ' + str(self.__output)
        print 'Action: ' + str(self.__action)
        print 'Coeff removal' + str(self.__coeff_removal)
        if self.__action:
            self.__action(self)
