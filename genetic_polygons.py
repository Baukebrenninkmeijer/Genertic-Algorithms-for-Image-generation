#!/usr/bin/python

"""
Below is an example of how to paint a polygon in TK. Your mission, if you choose to accept, is to create a genetic
algorithm that tries to approximate a given target image.
A gene is a random collections of points together with an RGBA color. In other words: The genome is a collection of polygons.
As a fitness function you can sum the pixel distance with the target image.
"""


from tkinter import *
import PIL
from PIL import Image, ImageDraw


def save():
    global image_number
    filename = f'image_{image_number}.png'   # image_number increments by 1 at every save
    image1.save(filename)
    image_number += 1


def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=1)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='black', width=1)
    lastx, lasty = x, y


root = Tk()

lastx, lasty = None, None
image_number = 0

cv = Canvas(root, width=640, height=480, bg='white')
# --- PIL
image1 = PIL.Image.new('RGB', (640, 480), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="save", command=save)
btn_save.pack()

# Phenotype hint...
# points = [0,0,canvas_width,canvas_height/2, 0, canvas_height]
# w.create_polygon(points, outline=python_green,
#                  fill='yellow', width=3)

# Getting raw pixel data might be slow in TK Inter, so SO suggests using images directly:
# from PIL import Image
# im = Image.open('dead_parrot.jpg') # Can be many different formats.
# pix = im.load()
# print im.size  # Get the width and hight of the image for iterating over
# print pix[x,y]  # Get the RGBA Value of the a pixel of an image
# pix[x,y] = value  # Set the RGBA Value of the image (tuple)
# im.save('alive_parrot.png')  # Save the modified pixels as .png

root.mainloop()