#
# this script Credit goes to https://github.com/dusty-nv/pytorch-segmentation
#   
import os
import re
import copy
import argparse

from PIL import Image
from multiprocessing import Pool as ProcessPool


#
# map of existing class label ID's (range 0-37) to new ID's (range 0-21)
# each entry consists of a tuple (new_ID, name, color)
#
CLASS_MAP = [   (0, 'other'),
                (1, 'wall'),
                (0, 'other'),
                (0, 'other'),
                (2, 'floor'),
                (0, 'other'),
                (14, 'ceiling'),
                (0, 'other'),
                (13, 'bed'),
                (12, 'window'),
                (0, 'other'),
                (3, 'closet'),
                (0, 'other'),
                (11, 'person'),
                (2, 'earth'),
                (16, 'door'),
                (15, 'table'),
                (0, 'other'),
                (0, 'other'),
                (4, 'curtain'),
                (5, 'chair'),
                (0, 'other'),
                (0, 'other'),
                (17, 'painting'),
                (6, 'sofa'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (7, 'carpet'),
                (0, 'other'),
                (5, 'armchair'),
                (5, 'seat'),
                (0, 'other'),
                (8, 'desk'),
                (0, 'other'),
                (3, 'closet'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (6, 'cusion'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (3, 'dresser'),
                (0, 'other'),
                (0, 'other'),
                (18, 'sink'),
                (0, 'other'),
                (0, 'other'),
                (9, 'refrigerator'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (4, 'blind'),
                (15, 'coffee table'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (19, 'stove'),
                (0, 'other'),
                (0, 'other'),
                (20, 'computer'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (10, 'television'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (2, 'ground'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (17, 'poster'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other'),
                (0, 'other')
            ]
           

#
# new class label names, corresponding to remapped class ID's (range 0-21)
#
"""
other
wall
floor
cabinet/closet
curtain
chair
cusion
carpet
desk
refrigerator
television
person
window
bed
ceiling
table
door
painting
sink
stove
computer
"""


# Generate Color Map in PASCAL VOC format
def generate_color_map(N=38):
        """ 
        https://github.com/meetshah1995/pytorch-semseg/blob/801fb200547caa5b0d91b8dde56b837da029f746/ptsemseg/loader/sunrgbd_loader.py#L108
        """
        def bitget(byteval, idx):
                return (byteval & (1 << idx)) != 0

        print('')
        print('color map: ')

        cmap = []

        for i in range(N):
                r = g = b = 0
                c = i

                for j in range(8):
                        r = r | (bitget(c, 0) << 7 - j)
                        g = g | (bitget(c, 1) << 7 - j)
                        b = b | (bitget(c, 2) << 7 - j)
                        c = c >> 3

                color = (r,g,b)
                print(color)
                cmap.append(color)

        return cmap


def remap_labels(args):
    input_path = args[0]
    output_path = args[1]
    colorized = args[2]

    print('{:s} -> {:s}'.format(input_path, output_path))

    if os.path.isfile(output_path):
        print('skipping image {:s}, already exists'.format(output_path))
        return

    img_input = Image.open(input_path)
    img_output = Image.new('RGB' if colorized is True else 'L', (img_input.width, img_input.height))

    for y in range(img_input.height):
        for x in range(img_input.width):
            org_label = img_input.getpixel((x,y))#[0]
            new_label = CLASS_MAP[org_label][2 if colorized else 0]
            img_output.putpixel((x,y), new_label)

    img_output.save(output_path)


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Remap SUNRGB-D Segmentation Images')
    parser.add_argument('input', type=str, metavar='IN', help='path to directory of annotated images to remap')
    parser.add_argument('output', type=str, metavar='OUT', help='path to directory to save remaped annotation images')
    parser.add_argument('--colorized', action='store_true', help='output colorized segmentation maps (RGB)')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    files = sorted_alphanumeric(os.listdir(args.input))
    worker_args = []

    for n in range(len(files)):
        worker_args.append((os.path.join(args.input, files[n]), os.path.join(args.output, 'img-{:06d}.png'.format(n+1)), args.colorized))

    #for n in worker_args:
    #    remap_labels(n)

    with ProcessPool(processes=args.workers) as pool:
        pool.map(remap_labels, worker_args)

