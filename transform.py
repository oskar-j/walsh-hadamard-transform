# coding: utf-8
from fal.task import Task

import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib.pyplot import hist

import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Let'c create compression task

    task = Task()

    # We define in 'data flow' that we wish to transform first
    task.with_action('compress')

    task.with_output('data/transformed.cim')  # this is a raw file after transformation
    # hence atypical filetype, as most of commercial tools won't be able to read this file

    # Let's see the input file ourselves

    input_img = Image.open('data/image.bmp')

    hist(input_img.histogram(), bins=40)

    # This is the input image
    plt.rcParams["figure.figsize"] = (20, 9)
    imshow(np.asarray(input_img))

    # We're telling our framework where to find the image
    task.with_input('data/image.bmp')

    # Let's process it!
    task.run()

    # Now we're recreating image from the result
    task = Task()
    task.with_action('extract')
    task.with_input('data/transformed.cim')
    task.with_output('data/recreated.bmp')

    # Run the process
    # Check inverseTransform(self, src) in fal.transforms for more details

    task.run()

    # And let's see the results

    output_image = Image.open('data/recreated.bmp')
    plt.rcParams["figure.figsize"] = (20, 9)
    imshow(np.asarray(output_image))

    hist(output_image.histogram(), bins=40)

    # As you can see, image after transformation has visible loss of quality
    # and it's color profile (histogram for pixel counts) has changed as well
