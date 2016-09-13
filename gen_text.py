from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import matplotlib.pyplot as plt
import numpy as np

import string

#alph = list(string.ascii_lowercase + '$')
alph = list(string.ascii_lowercase + ':/.?' + '$')
alph_dict = dict((v,k) for k,v in enumerate(alph))
tokens = alph

font_list = [
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
]

def get_font(font_type, size):
    return ImageFont.truetype(font_list[font_type], size)

def encode(text):
    return np.array([alph_dict[t] for t in text], dtype=np.uint8)

def render_text(text, img_size, font, offset):
    img = Image.new("L", img_size)
    draw = ImageDraw.Draw(img)
    draw.text(offset,text,255,font=font)

    return np.array(img)

def render_single(text=None, img_size=None):

    if not img_size:
        img_size = (len(text)*10, 20)

    # coin toss for the text offset
    y_offset = np.random.randint(0, 7) 
    x_offset = np.random.randint(0, 30) 
    offset = (x_offset, y_offset)

    # coinn toss for the font type and sizesize
    font_size = np.random.randint(12, 17) 
    font_type = np.random.randint(0, len(font_list))
    font = get_font(font_type, font_size) 

    # render
    img = render_text(text, img_size, font, offset)

    return img


def render_batch(seq_length, batch_size, img_size=None):
    imgs = []
    labels = []

    # choose font size
    min_length, max_length = seq_length

    if not img_size:
        # adjust the size
        img_size = (max_length*10, 20)

        # add some noise to the length
        p = int(0.2*img_size[0])
        shift = np.random.randint(-p,p)
        img_size = (img_size[0] + shift, img_size[1])


    for i in xrange(batch_size):
        # coin toss for the length
        l = np.random.randint(min_length, max_length)
        sample = np.random.choice(alph[:-1], size=l)

        # coin toss for the text offset
        y_offset = np.random.randint(0, 7) 
        x_offset = np.random.randint(0, 30) 
        offset = (x_offset, y_offset)

        # coinn toss for the font type and sizesize
        font_size = np.random.randint(12, 17) 
        font_type = np.random.randint(0, len(font_list))
        font = get_font(font_type, font_size) 

        # render
        img = render_text("".join(sample), img_size, font, offset)
        imgs.append(img)
        labels.append(encode(sample))

    return np.array(imgs), labels

if __name__ == '__main__':
    bx, by = render_batch(seq_length=(5, 20), batch_size=8)

    plt.imshow(np.vstack(bx), cmap=plt.cm.Greys_r)
    plt.show()

