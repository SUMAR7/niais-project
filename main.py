import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
from tkinter import PhotoImage
import image_detection as img_d

MIN_WINDOW_DIMENSION = '900x900'

root = tk.Tk()

# entering full screen by default
root.attributes('-zoomed', True)

# Root window min dimensions
root.geometry(MIN_WINDOW_DIMENSION)

# Setting Title & Icon
root.title('Number Plate Recognition System | NIAIS')
root.iconphoto(True, PhotoImage(file="/home/sajjad/PycharmProjects/niais-project/assets/images/sumar-niais.png"))
img = Image.open("/home/sajjad/PycharmProjects/niais-project/assets/images/niaislogo.png")

logo = ImageTk.PhotoImage(img.resize((512, 512), Image.ANTIALIAS))
panel = Label(root, image=logo)
panel.grid(column=0, row=0, pady=200, padx=100)


def open_image_detection():
    new_window = Toplevel(root)
    img_d.start_image_detection(new_window)


def open_live_detection():
    # live_d.start_live_detection()
    print('live detection started')


# Image detection button styles
style = Style()
style.configure('TButton', font=('calibri', 20, 'bold'), borderwidth='5')

# Changes will be reflected
# by the movement of mouse.
style.map('TButton', foreground=[('active', '!disabled', 'green')], background=[('active', 'black')])

# image detection
image_detection = Button(root, text='Image Detection', command=open_image_detection)
image_detection.grid(row=0, column=1, padx=100)

# live detection
live_detection = Button(root, text='Live Detection', command=open_live_detection)
live_detection.grid(row=0, column=2, padx=50)

root.mainloop()
