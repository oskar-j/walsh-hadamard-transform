class AbstractColorModel:

    def get_rgb(self, color):
        pass

    def get_y_cb_cr(self, color):
        pass


class RgbColorModel(AbstractColorModel):

    def get_rgb(self, color):
        return color

    def get_y_cb_cr(self, color):
        r, g, b = color
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return y, cr, cb


class YCbCrColorModel(AbstractColorModel):

    def get_rgb(self, color):
        y, cr, cb = color
        r = int(y + 1.402 * (cr - 128))
        g = int(y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128))
        b = int(y + 1.772 * (cb - 128))
        if r < 0:
            r = 0
        if r > 255:
            r = 255
        if g < 0:
            g = 0
        if g > 255:
            g = 255
        if b < 0:
            b = 0
        if b > 255:
            b = 255
        return r, g, b

    def get_y_cb_cr(self, color):
        return color
