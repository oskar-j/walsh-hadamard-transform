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

    @staticmethod
    def _get_padding_size(x, a):
        return ((x - 1) // a + 1) * a - x

    def _slice(self, x, width, height, block_size):
        x = np.matrix(x).reshape(height, width)
        print 'Shape: ' + str(x.shape)

        height_padding = self._get_padding_size(height, block_size)
        print 'Height pad: ' + str(height_padding)
        x = np.concatenate((x, np.zeros((height_padding, x.shape[1]))), 0)

        width_padding = self._get_padding_size(width, block_size)
        print 'Width pad: ' + str(width_padding)
        x = np.concatenate((x, np.zeros((x.shape[0], width_padding))), 1)

        blocks = []
        for row in np.vsplit(x, x.shape[0] / block_size):
            blocks.extend(np.hsplit(row, row.shape[1] / block_size))

        print 'Number of blocks: ' + str(len(blocks))
        print blocks[0].shape

        return blocks

    @staticmethod
    def _merge(blocks, width, height):
        block_height, block_width = blocks[0].shape
        blocks_per_row = (width - 1) // block_width + 1
        blocks_per_column = len(blocks) / blocks_per_row
        rows = [np.hstack(blocks[i * blocks_per_row: (i + 1) * blocks_per_row])
               for i in xrange(0, blocks_per_column)]
        return np.vstack(rows)[0:height, 0:width]

    def compress(self):
        print 'Action: compress'

        bmp_image = BMPImage()
        try:
            bmp_image.load(self.__input)
        except IOError as e:
            print 'IO Error: ' + str(e)
            return

        width, height = bmp_image.get_dimensions()
        data = bmp_image.get_raw_data()

        color = RgbColorModel()
        y, cb, cr = zip(*map(color.get_y_cb_cr, data))

        y_block_size = 8
        cb_block_size = 16
        cr_block_size = 16

        y = self._slice(y, width, height, y_block_size)
        cb = self._slice(cb, width, height, cb_block_size)
        cr = self._slice(cr, width, height, cr_block_size)

        transform = WalshHadamardTransform(self.__coeff_removal)
        spectral_y = transform.transform_sequence(y)
        spectral_cb = transform.transform_sequence(cb)
        spectral_cr = transform.transform_sequence(cr)

        customizable_image = CustomizableImage()
        customizable_image.set_dimensions(width, height)
        customizable_image.set_descriptions(
            (y_block_size, 4, len(spectral_y)),
            (cb_block_size, 4, len(spectral_cb)),
            (cr_block_size, 4, len(spectral_cr))
        )
        customizable_image.set_data(spectral_y, spectral_cb, spectral_cr)
        customizable_image.save(self.__output)

    def extract(self):
        print 'Action: extract'

        customizable_image = CustomizableImage.load(self.__input)
        width, height = customizable_image.get_dimensions()

        transform = WalshHadamardTransform()

        spectral_y = customizable_image.get_y_data()
        if spectral_y:
            y = transform.inverse_transform_sequence(spectral_y)
            y = self._merge(y, width, height)
            y = np.array(y).reshape(-1).tolist()
            print len(y)
        else:
            y = [0] * (width * height)

        spectral_cb = customizable_image.get_cb_data()
        if spectral_cb:
            cb = transform.inverse_transform_sequence(spectral_cb)
            cb = self._merge(cb, width, height)
            cb = np.array(cb).reshape(-1).tolist()
            print len(cb)
        else:
            cb = [128] * (width * height)

        spectral_cr = customizable_image.get_cr_data()
        if spectral_cr:
            cr = transform.inverse_transform_sequence(spectral_cr)
            cr = self._merge(cr, width, height)
            cr = np.array(cr).reshape(-1).tolist()
            print len(cr)
        else:
            cr = [128] * (width * height)

        pixels = map(YCbCrColorModel().get_rgb, zip(y, cb, cr))

        bmp_image = BMPImage()
        bmp_image.set_dimensions(width, height)
        bmp_image.set_raw_data(pixels)
        bmp_image.save(self.__output)

    __actions = {'compress': compress, 'extract': extract}

    def with_input(self, input):
        self.__input = input
        return self

    def with_coeff_removal(self, coeff):
        self.__coeff_removal = coeff
        return self

    def with_output(self, output):
        self.__output = output
        return self

    def with_action(self, action):
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
