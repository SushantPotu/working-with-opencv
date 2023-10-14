# openCV stuff


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Import library
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import IPython.display as ipd
import tqdm.notebook as tqdm
import subprocess

# Capturing the webcam's feed
vid = cv2.VideoCapture(0)
# Setting frame
frameWidth = 640
frameHeight = 480
vid.set(3, frameWidth)
vid.set(4, frameHeight)
# Checking if webcam video is read
while True:
    # Reading the webcam's feed
    success, img = vid.read()
    # img2 = cv2.imread('images/cats.jpg')
    # rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    blank = np.zeros(img.shape, dtype='uint8')
    cv2.imshow('Blank', blank)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 125, 175)
    contours, hierarchies = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print(f'{len(contours)} found!')
    cv2.drawContours(blank, contours, -1, (0, 0, 255), 1)
    cv2.imshow('contours', blank)

    # plt.imshow(rgb)
    # plt.show()

    # Displaying the webcam's feed
    key = cv2.waitKey(10)
    if key == ord('m'):
         break

vid.release()
cv2.destroyAllWindows()