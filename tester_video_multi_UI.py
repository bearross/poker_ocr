# coding=utf-8
import os
from subprocess import Popen
import threading
import cv2
import sys
import numpy as np
import glob
import argparse
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

#path = 'D:/Upwork/PokerGameOCR/testProject/'


def p_recog(proc_i):
    import os
    os.startfile("tester_video_multi{}.exe".format(proc_i))

    """
    if proc_i == 0:
        p = Popen(["python", r"tester_video_multi0.exe"], cwd=path)
        stdout, stderr = p.communicate()
    elif proc_i == 1:
        p = Popen(["python", r"tester_video_multi0.exe"], cwd=path)
        stdout, stderr = p.communicate()
    elif proc_i == 2:
        p = Popen(["python", r"tester_video_multi0.exe"], cwd=path)
        stdout, stderr = p.communicate()
    elif proc_i == 3:
        p = Popen(["python", r"tester_video_multi0.exe"], cwd=path)
        stdout, stderr = p.communicate()
    """

def detect_lobby(img_color, template):
    w, h = template.shape[::-1]
    # img_color = cv2.imread(file)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img2 = img.copy()
    meth = 'cv2.TM_SQDIFF_NORMED'
    # meth = 'cv2.TM_SQDIFF_NORMED'
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # if top_left[0]>img.shape[1]*0.1 or top_left[1]>img.shape[0]*0.1:
    #    return "None"

    cv2.rectangle(img_color, top_left, bottom_right, (0, 255, 0), 2)
    # cv2.imshow("result", img_color)
    # cv2.waitKey(0)
    rect = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
    return rect


def get_points(top_left_x, top_right_x, lobby_left_y, lobby_right_y):
    rat = (top_right_x - top_left_x) / 825
    bottom_left_y = int(lobby_left_y + rat * 570)
    return [top_left_x, lobby_left_y, top_right_x, bottom_left_y]


def get_ratio_rects(videofile):
    file = "test.png"
    ratios = []
    rects = []
    vid_capture = cv2.VideoCapture(videofile)
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    else:
        frame_count = vid_capture.get(7)
    while (vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = vid_capture.read()
        if ret == True:
            # cv2.imwrite(file, frame)
            # frame = cv2.imread(file)
            template1 = cv2.imread('lobby1.png', 0)
            template2 = cv2.imread('lobby2.png', 0)
            frame_w = frame.shape[1]
            frame_h = frame.shape[0]

            lobbys = []
            settings = []

            for i in range(4):
                res = detect_lobby(frame, template1)
                top_left_x = res[0]
                lobby_left_y = res[1]
                res = detect_lobby(frame, template2)
                bottom_right_x = res[2]
                lobby_right_y = res[3]
                lobbys.append([top_left_x, lobby_left_y])
                settings.append([bottom_right_x, lobby_right_y])

            for i in range(len(lobbys)):
                lobby = lobbys[i]
                x0 = lobby[0]
                y0 = lobby[1]
                x1 = -1
                for setting in settings:
                    if 0 < setting[0] - x0 < 800 and 0 < setting[1] - y0 < 300:
                        x1 = setting[0]
                        break
                if x1 < 0:
                    continue
                ratio = 825 / (x1 - x0)
                new_w = int(ratio * frame_w)
                new_h = int(ratio * frame_h)
                # print(ratio)
                ratios.append(ratio)
                resized_frame = cv2.resize(frame, (new_w, new_h))
                rect = get_points(x0, x1, y0, y0)
                crop = resized_frame[int(ratio * y0):int(ratio * rect[3]),
                       int(ratio * x0):int(ratio * rect[2])]
                x0 = int(ratio * x0)
                x1 = int(ratio * rect[2])
                y0 = int(ratio * y0)
                y1 = int(ratio * rect[3])
                rects.append([x0, y0, x1, y1])
                # cv2.imshow('res', crop)
                # cv2.waitKey(0)
            return ratios, rects
        else:
            break
    return [], []


def writing_videos(videofile):
    rations, rects = get_ratio_rects(videofile)
    # Create a video capture object, in this case we are reading the video from a file
    vid_capture = cv2.VideoCapture(videofile)
    fps = 30
    frame_size = (825, 570)
    output1 = cv2.VideoWriter('output/0.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, frame_size)
    output2 = cv2.VideoWriter('output/1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, frame_size)
    output3 = cv2.VideoWriter('output/2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, frame_size)
    output4 = cv2.VideoWriter('output/3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, frame_size)
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    else:
        # Get frame rate information
        # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        fps = vid_capture.get(5)
        print('Frames per second : ', fps, 'FPS')
        # Get frame count
        # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
        frame_count = vid_capture.get(7)
        print('Frame count : ', frame_count)
    image_id = 0
    while (vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = vid_capture.read()
        image_id += 1
        if image_id % 5 != 0:
            continue
        if ret == True:
            for i in range(len(rations)):
                rat = rations[i]
                rect = rects[i]
                w = int(frame.shape[1] * rat)
                h = int(frame.shape[0] * rat)
                frame_new = cv2.resize(frame, (w, h))
                crop = frame_new[rect[1]:rect[1] + 570, rect[0]:rect[0] + 825]
                crop = cv2.resize(crop, (825, 570))
                # print(crop.shape[1], crop.shape[0])

                if i == 0:
                    cv2.imshow('res1', crop)
                    output1.write(crop)
                elif i == 1:
                    cv2.imshow('res2', crop)
                    output2.write(crop)
                elif i == 2:
                    cv2.imshow('res3', crop)
                    output3.write(crop)
                elif i == 3:
                    cv2.imshow('res4', crop)
                    output4.write(crop)
            #cv2.imshow('Frame', frame)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(1)
        else:
            break

        # Release the video capture object
    vid_capture.release()
    cv2.destroyAllWindows()
    output1.release()
    output2.release()
    output3.release()
    output4.release()
    return len(rations)


def main():
    global filename
    get_value()
    videofile = filename
    st = time.time()
    for file_old in glob.glob('output/*.avi'):
        os.remove(file_old)
    print("Splitting Videos!!!")
    num_videos = writing_videos(videofile)
    print("Processing Videos!!!")
    list_thread = []
    # num_videos = 4
    #application_path = os.path.dirname(sys.executable)
    #print(application_path)
    for proc_i in range(num_videos):
        thread = threading.Thread(target=p_recog, args=(proc_i,))
        list_thread.append(thread)
    for thread in list_thread:
        thread.start()
    for th in list_thread:
        th.join()

    # p_recog(2)
    print('Total processing time is {} secs.'.format(time.time() - st))

filename = ''
seat1 = 0
seat2 = 0
seat3 = 0
seat4 = 0

# create the root window
root = tk.Tk()
root.title('Tkinter Open File Dialog')
root.resizable(False, False)
root.geometry('500x150')

def get_value():
   seat1 = int(variable1.get())
   seat2 = int(variable1.get())
   seat3 = int(variable1.get())
   seat4 = int(variable1.get())
   filename = os.path.dirname(sys.executable) + '/output/seats.txt'
   f = open(filename, 'w')
   f.write('{},{},{},{}'.format(seat1, seat2, seat3, seat4))
   f.close()

def select_file():
    global filename
    filetypes = (
        ('text files', '*.mp4'),
        ('text files', '*.mkv'),
        ('text files', '*.avi'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

    #showinfo(
    #    title='Selected File',
    #    message=filename
    #)
    label.config(text='Path: {}'.format(filename.split('/')[-1]))

# open button
open_button = ttk.Button(
    root,
    text='Open a File',
    command=select_file
)

# start button
start_button = ttk.Button(
    root,
    text='Start',
    command=main
)
#open_button.pack(expand=True, side='left')
#start_button.pack(expand=True, side='right')

label = ttk.Label(root, text="Path:")
#label.place(x=5, y=5)
vl1 = ttk.Label(root, text="Seat number of Video1")
vl2 = ttk.Label(root, text="Seat number of Video2")
vl3 = ttk.Label(root, text="Seat number of Video3")
vl4 = ttk.Label(root, text="Seat number of Video4")

variable1 = tk.StringVar(root)
variable1.set("9")  # default value

txt1 = tk.OptionMenu(root, variable1, '2', '6', '8', '9')
txt1.config(width=10)

variable2 = tk.StringVar(root)
variable2.set("9")  # default value
txt2 = tk.OptionMenu(root, variable2, '2', '6', '8', '9')
txt2.config(width=10)

variable3 = tk.StringVar(root)
variable3.set("9")  # default value
txt3 = tk.OptionMenu(root, variable3, '2', '6', '8', '9')
txt3.config(width=10)

variable4 = tk.StringVar(root)
variable4.set("9")  # default value
txt4 = tk.OptionMenu(root, variable4, '2', '6', '8', '9')
txt4.config(width=10)


label.grid(row=0, column=0)
vl1.grid(row=1, column=0)
vl2.grid(row=1, column=1)
vl3.grid(row=3, column=0)
vl4.grid(row=3, column=1)
txt1.grid(row=2, column=0)
txt2.grid(row=2, column=1)
txt3.grid(row=4, column=0)
txt4.grid(row=4, column=1)

open_button.grid(row=5, column=0)
start_button.grid(row=5, column=1)

# run the application
root.mainloop()