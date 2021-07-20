from PIL import Image
import pathlib
import glob
import sys

file_num = len(glob.glob('output/*.jpg'))

images = []
for i in range(file_num):
    img = Image.open('output/%d_img.jpg' % i)
    images.append(img)

images[0].save('output/morphing.gif', save_all = True, append_images = images, duration = 100, loop = 0)
