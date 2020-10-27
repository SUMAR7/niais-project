from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2
import pytesseract as tess
import pandas as pd

owners = pd.read_csv('/home/sajjad/PycharmProjects/number-plates-detection/assets/datasets/owners.csv', sep=",")


def clean2_plate(plate):
    gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
    num_contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if num_contours:
        contour_area = [cv2.contourArea(c) for c in num_contours]
        max_cntr_index = np.argmax(contour_area)

        max_cnt = num_contours[max_cntr_index]
        max_cnt_area = contour_area[max_cntr_index]
        x, y, w, h = cv2.boundingRect(max_cnt)

        if not ratio_check(max_cnt_area, w, h):
            return plate, None

        final_img = thresh[y:y + h, x:x + w]
        return final_img, [x, y, w, h]

    else:
        return plate, None


def ratio_check(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio
    if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
        return False
    return True


def is_max_white(plate):
    avg = np.mean(plate)
    if avg >= 115:
        return True
    else:
        return False


def ratio_and_rotation(rect):
    (x, y), (width, height), rect_angle = rect

    if width > height:
        angle = -rect_angle
    else:
        angle = 90 + rect_angle

    if angle > 15:
        return False

    if height == 0 or width == 0:
        return False

    area = height * width
    if not ratio_check(area, width, height):
        return False
    else:
        return True


def classify(top, file_path):
    Label(top, text="Owner's Detail", font=('Arial Bold', 15, 'bold'), justify=LEFT, anchor="w") \
        .grid(sticky=W, row=0, column=1, padx=50, pady=(50, 0))

    Label(top, text="Detected Number Plate", font=('Arial Bold', 15, 'bold'), justify=LEFT, anchor="w") \
        .grid(sticky=W, row=0, column=2, padx=50, pady=(50, 0))

    res_text = [0]
    res_img = [0]
    img = cv2.imread(file_path)

    img2 = cv2.GaussianBlur(img, (3, 3), 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img2 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=3)
    _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = img2.copy()
    cv2.morphologyEx(src=img2, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    num_contours, hierarchy = cv2.findContours(morph_img_threshold, mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img2, num_contours, -1, (0, 255, 0), 1)

    for i, cnt in enumerate(num_contours):

        min_rect = cv2.minAreaRect(cnt)

        if ratio_and_rotation(min_rect):

            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y:y + h, x:x + w]
            print("Number identified number plate...")

            res_img[0] = plate_img
            cv2.imwrite("/home/sajjad/PycharmProjects/number-plates-detection/assets/result.png", plate_img)

            if is_max_white(plate_img):
                clean_plate, rect = clean2_plate(plate_img)

                if rect:
                    fg = 0
                    x1, y1, w1, h1 = rect
                    x, y, w, h = x + x1, y + y1, w1, h1
                    plate_im = Image.fromarray(clean_plate)
                    text = tess.image_to_string(plate_im, lang='eng')
                    res_text[0] = text

                    if text:
                        Label(top, text=f'Detected Number: {text.strip()}', font=('calibri', 15), justify=LEFT,
                              anchor="w") \
                            .grid(sticky=W, row=1, column=1, padx=50, pady=(0, 0))
                        details = owners[owners["RegistrationNo"] == text.strip()]

                        Label(top, text=f"Owner's Name: {details['Name'].values[0]}", font=('calibri', 15),
                              justify=LEFT,
                              anchor="w") \
                            .grid(sticky=W, row=2, column=1, padx=50, pady=(0, 0))

                        img = Image.open(f"{details['Image'].values[0]}")

                        owner_image = ImageTk.PhotoImage(img.resize((512, 512), Image.ANTIALIAS))
                        o_image_label = Label(top, bd=10, image=owner_image, justify=LEFT, anchor="w")
                        o_image_label.image = owner_image
                        o_image_label.grid(sticky=W, row=3, column=1, padx=(40, 0), pady=(50, 0))

                        Label(top, text=f"Model & Make: {details['Make'].values[0]} - {details['Type'].values[0]}", font=('calibri', 15),
                              justify=LEFT,
                              anchor="w") \
                            .grid(sticky=W, row=4, column=1, padx=50, pady=(0, 0))

                        Label(top, text=f"Original Reg #: {details['RegistrationNo'].values[0]}",
                              font=('calibri', 15),
                              justify=LEFT,
                              anchor="w") \
                            .grid(sticky=W, row=5, column=1, padx=50, pady=(0, 0))

                        print("Number Detected Plate Text : ", text.strip())
                        break

    uploaded = Image.open("/home/sajjad/PycharmProjects/number-plates-detection/assets/result.png")
    result_image = ImageTk.PhotoImage(uploaded)
    r_image_label = Label(top, bd=10, image=result_image, justify=LEFT, anchor="w")
    r_image_label.image = result_image
    r_image_label.grid(sticky=W, row=1, column=2, padx=(40, 0), pady=(0, 0))


def show_classify_button(top, file_path):
    classify_b = Button(top, text="Scan Image", command=lambda: classify(top, file_path), padx=10, pady=5, width=30)
    classify_b.grid(row=5, column=0, sticky=W, padx=(50, 0), pady=(10, 0))


def upload_image(top, image_label):
    try:
        file_path = filedialog.askopenfilename(
            initialdir='/home/sajjad/PycharmProjects/number-plates-detection/assets/test_samples')
        uploaded = Image.open(file_path)
        im = ImageTk.PhotoImage(uploaded.resize((512, 512), Image.ANTIALIAS))
        image_label.configure(image=im)
        image_label.image = im
        image_label.configure(text='')
        image_label.grid(sticky=W, row=3, column=0, padx=(40, 0), pady=(50, 0))
        show_classify_button(top, file_path)
    except:
        pass


def start_image_detection(top):
    top = top

    top.title("Image Detection")

    # sets the geometry of toplevel
    top.geometry("900x900")
    top.attributes('-zoomed', True)
    print('setting up UI')

    Label(top, text='NIAIS', font=('Arial Bold', 15), justify=LEFT, anchor="w").grid(sticky=W, row=0, column=0, padx=40,
                                                                                     pady=(50, 0))
    Label(top, text='Sajjad Umar', font=('Arial Bold', 15), justify=LEFT, anchor="w").grid(sticky=W, row=1, column=0,
                                                                                           padx=40, pady=2)
    Label(top, text='NIAIS-0983', font=('Arial Bold', 15), justify=LEFT, anchor="w").grid(sticky=W, row=2, column=0,
                                                                                          padx=40, pady=2)

    img = Image.open("/home/sajjad/PycharmProjects/niais-project/assets/images/initial.png")
    start_image = ImageTk.PhotoImage(img.resize((512, 512), Image.ANTIALIAS))
    image_label = Label(top, bd=10, image=start_image, justify=LEFT, anchor="w")
    image_label.image = start_image
    image_label.grid(sticky=W, row=3, column=0, padx=(40, 0), pady=(50, 0))

    upload = Button(top, text="Upload an image", command=lambda: upload_image(top, image_label), padx=10, pady=5,
                    width=30)
    upload.grid(row=4, column=0, sticky=W, padx=(50, 0), pady=(50, 0))
