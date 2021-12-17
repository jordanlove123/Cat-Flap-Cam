import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import ImageTk, Image

global img

def processImage(im):
    arr = np.array(im)
    r, g, b = np.split(arr, 3, axis=2)
    r = r.reshape(-1)
    g = r.reshape(-1)
    b = r.reshape(-1)
    bitmap = list(map(lambda x: 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2], zip(r, g, b)))
    bitmap = np.array(bitmap).reshape([arr.shape[0], arr.shape[1]])
    bitmap = np.dot((bitmap > 128).astype(float), 255)
    i = Image.fromarray(bitmap.astype(np.uint8))
    i.save('result.jpg')

    rows = len(bitmap)
    cols = len(bitmap[0])

    for i in range(rows):
        for j in range(cols):
            bitmap[i][j] = (255 - bitmap[i][j]) / 255

    bitmap = np.array(bitmap.reshape(-1))

    return bitmap


def addDataThere():
    f = open("data.txt", 'a')
    img = Image.open("cookie.jpg")
    bitmap = processImage(img)
    f.write(str(bitmap) + ":1\n")
    f.close()

def addDataNotThere():
    f = open("data.txt", 'a')
    img = Image.open("cookie.jpg")
    bitmap = processImage(img)
    f.write(str(bitmap) + ":0\n")
    f.close()


if __name__ == "__main__":
    try:
        f = open("data.txt", 'x')
        f.close()
    except FileExistsError:
        f = open("data.txt", "w")
        f.close()
    window = tk.Tk()
    window.title("Image")
    im = Image.open("cookie.jpg")
    width, height = im.size
    window.geometry("{0}x{1}".format(width + 60, height + 60))

    img = ImageTk.PhotoImage(Image.open("cookie.jpg"))
    panel = tk.Label(window, image=img)
    panel.pack(fill="both", expand="yes")
    """hello = tk.Label(text="Hello world!")
    hello.pack()"""
    button = tk.Button(window, text="There", command=addDataThere)
    button.pack()
    button2 = tk.Button(window, text="Not There", command=addDataNotThere)
    button2.pack()

    tk.mainloop()
