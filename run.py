from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
#import tkinter as tk
#from tkinter import ttk
import threading

import cv2 as cv
import numpy as np
import os, sys
from time import time
from windowcapture import WindowCapture

import pygetwindow
import time
import tensorflow as tf
import random

#
TRAINING_STEPS = 16000
BATCH_SIZE = 16
#
LEARNING_RATE_BASE = 0.001
MOMENTUM = 0.9
REG_LAMBDA = 0.0001
GRAD_CLIP = 5.0
#
BATCH_SIZE_VALID = 1
VALID_FREQ = 100
LOSS_FREQ = 1
#
#
model_recog_dir = 'model'
model_recog_name = 'model_recog'
model_recog_pb_file = model_recog_name + '.pb'
height_norm = 36
alphabet = '''0123456789#'''
alphabet_blank = '`'

class ModelRecog():
    #
    HEIGHT_NORM = height_norm
    #
    def __init__(self):
        #
        # default pb path
        self.pb_file = os.path.join(model_recog_dir, model_recog_pb_file)
        #
        self.sess_config = tf.ConfigProto()
        # self.sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95
        #
        self.is_train = False
        #
        self.graph = None
        self.sess = None
        #
        self.train_steps = TRAINING_STEPS
        self.batch_size = BATCH_SIZE
        #
        self.learning_rate_base = LEARNING_RATE_BASE
        self.momentum = MOMENTUM
        self.reg_lambda = REG_LAMBDA
        self.grad_clip = GRAD_CLIP
        #
        self.batch_size_valid = BATCH_SIZE_VALID
        self.valid_freq = VALID_FREQ
        self.loss_freq = LOSS_FREQ
        #

    def prepare_for_prediction(self, pb_file_path=None):
        #
        if pb_file_path == None: pb_file_path = self.pb_file
        #
        if not os.path.exists(pb_file_path):
            print('ERROR: %s NOT exists, when load_pb_for_predict()' % pb_file_path)
            return -1
        #
        self.graph = tf.Graph()
        #
        with self.graph.as_default():
            #
            with open(pb_file_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                #
                tf.import_graph_def(graph_def, name="")
                #
            #
            # input/output variables
            #
            self.x = self.graph.get_tensor_by_name('x-input:0')
            self.w = self.graph.get_tensor_by_name('w-input:0')
            #
            self.seq_len = self.graph.get_tensor_by_name('seq_len:0')
            self.result_logits = self.graph.get_tensor_by_name('rnn_logits/BiasAdd:0')
            #
            self.result_i = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:0')
            self.result_v = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:1')
            self.result_s = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:2')
            #
        #
        print('graph loaded for prediction')
        #
        self.sess = tf.Session(graph=self.graph, config=self.sess_config)
        #

    def predict(self, image_in):
        #
        # input data
        if isinstance(image_in, str):
            img = Image.open(image_in)
            img = img.convert('RGB')
            img_size = img.size
            if img_size[1] != height_norm:
                w = int(img_size[0] * height_norm * 1.0 / img_size[1])
                img = img.resize((w, height_norm))
            img_data = np.array(img, dtype=np.float32) / 255  # (height, width, channel)
            img_data = [img_data[:, :, 0:3]]
        else:
            # np array
            img_data = image_in
        #
        w_arr = [img_data[0].shape[1]]  # batch, height, width, channel
        #
        with self.graph.as_default():
            #
            feed_dict = {self.x: img_data, self.w: w_arr}
            #
            results, seq_length, d_i, d_v, d_s = \
                self.sess.run([self.result_logits, self.seq_len,
                               self.result_i, self.result_v, self.result_s], feed_dict)
            #
            decoded = tf.SparseTensorValue(indices=d_i, values=d_v, dense_shape=d_s)
            trans = convert2ListLabels(decoded)
            # print(trans)
            #
            for item in trans:
                seq = list(map(mapOrder2Char, item))
                str_result = ''.join(seq)
                #
        #
        return str_result

def convert2ListLabels(sparse_tensor_value):

    shape = sparse_tensor_value.dense_shape
    indices = sparse_tensor_value.indices
    values = sparse_tensor_value.values

    list_labels = []
    #
    item = [0] * shape[1]
    for i in range(shape[0]): list_labels.append(item)
    #

    for idx, value in enumerate(values):
        #
        posi = indices[idx]
        #
        list_labels[posi[0]][posi[1]] = value
        #

    return list_labels

def define_alphabet():
    pass

def mapChar2Order(char): return alphabet.index(char)

def mapOrder2Char(order):
    if order == len(alphabet):
        return alphabet_blank
    else:
        return alphabet[order]

class NewWindow():

    def __init__(self):
        super().__init__()
        pass

    def show_window(self, results=None, clicked_btn=None):
        self.newWindow = Tk()
        # sets the title of the
        # Toplevel widget
        self.newWindow.title("Result")

        # sets the geometry of toplevel
        self.newWindow.geometry("300x350")
        self.tree_player = Treeview(self.newWindow, selectmode='browse')
        self.tree_player.place(x=50, y=50)
        self.tree_player.update_idletasks()
        self.tree_player.yview_moveto('1.0')
        vsb = Scrollbar(self.newWindow, orient="vertical", command=self.tree_player.yview)
        vsb.place(x=50 + 200, y=50, height=200 + 20)

        self.tree_player.configure(yscrollcommand=vsb.set)

        self.tree_player["columns"] = ("1", "2")
        self.tree_player['show'] = 'headings'
        self.tree_player.column("1", width=100, anchor='c')
        self.tree_player.column("2", width=100, anchor='c')
        self.tree_player.heading("1", text="Action")
        self.tree_player.heading("2", text="Time")
        self.label_player_selected = Label(self.newWindow, text="Player {}".format(clicked_btn))
        self.label_player_selected.place(x=20, y=25)

        for it in self.tree_player.get_children():
            self.tree_player.delete(it)

        for it in results:
            if clicked_btn == int(it[1].split('Player')[1]) and (it[2] != 'SB' and it[2] != 'BB' and it[2] != 'DONTSHOW'
                                                                      and it[2] != 'SHOWCARDS' and it[2] != 'MUCK' and it[2] != 'RESERVED'):
                print(it[2])
                t = float(it[3])
                # try:
                if it[2] == 'BET':
                    self.tree_player.insert("", 'end', text="L1", values=('{}'.format(it[2]), "%.1f" % round(t, 2)), tags=("BET"))
                    self.tree_player.tag_configure('BET', foreground='red', background='white')
                elif it[2] == 'CALL':
                    self.tree_player.insert("", 'end', text="L1", values=('{}'.format(it[2]), "%.1f" % round(t, 2)),
                                            tags=("CALL"))
                    self.tree_player.tag_configure('CALL', foreground='blue', background='white')
                elif it[2] == 'RAISE':
                    self.tree_player.insert("", 'end', text="L1", values=('{}'.format(it[2]), "%.1f" % round(t, 2)),
                                            tags=("RAISE"))
                    self.tree_player.tag_configure('RAISE', foreground='purple', background='white')
                elif it[2] == 'CHECK':
                    self.tree_player.insert("", 'end', text="L1", values=('{}'.format(it[2]), "%.1f" % round(t, 2)),
                                            tags=("CHECK"))
                    self.tree_player.tag_configure('CHECK', foreground='green', background='white')
                elif it[2] == 'FOLD':
                    self.tree_player.insert("", 'end', text="L1", values=('{}'.format(it[2]), "%.1f" % round(t, 2)),
                                            tags=("FOLD"))
                    self.tree_player.tag_configure('FOLD', foreground='black', background='white')
                self.tree_player.update_idletasks()
                self.tree_player.yview_moveto('1.0')
                self.image_id_old = it[-1]
                # except:
                #    continue

        self.newWindow.mainloop()

    def insert_tree(self, it):
        t = float(it[3])
        # try:
        if it[2] == 'BET':
            self.tree_player.insert("", 'end', text="L1", values=('{}'.format(it[2]), "%.1f" % round(t, 2)),
                                    tags=("BET"))
            self.tree_player.tag_configure('BET', foreground='red', background='white')
        elif it[2] == 'CALL':
            self.tree_player.insert("", 'end', text="L1", values=('{}'.format(it[2]), "%.1f" % round(t, 2)),
                                    tags=("CALL"))
            self.tree_player.tag_configure('CALL', foreground='blue', background='white')
        elif it[2] == 'RAISE':
            self.tree_player.insert("", 'end', text="L1", values=('{}'.format(it[2]), "%.1f" % round(t, 2)),
                                    tags=("RAISE"))
            self.tree_player.tag_configure('RAISE', foreground='purple', background='white')
        elif it[2] == 'CHECK':
            self.tree_player.insert("", 'end', text="L1", values=('{}'.format(it[2]), "%.1f" % round(t, 2)),
                                    tags=("CHECK"))
            self.tree_player.tag_configure('CHECK', foreground='green', background='white')
        elif it[2] == 'FOLD':
            self.tree_player.insert("", 'end', text="L1", values=('{}'.format(it[2]), "%.1f" % round(t, 2)),
                                    tags=("FOLD"))
            self.tree_player.tag_configure('FOLD', foreground='black', background='white')
        self.tree_player.update_idletasks()
        self.tree_player.yview_moveto('1.0')

    def update_newwindow(self, it):
        for i in range(10):
            try:
                self.insert_tree(it)
                break
            except:
                continue



class App:
    def __init__(self, window_title):
        self.win = Tk()
        self.win.title(window_title)
        self.win.minsize(width=330, height=400)
        self.win.resizable(width=0, height=0)
        self.label = Label(self.win, text="Number of Seats")
        self.label.place(x=30, y=5)

        #self.label_player = Label(self.win, text="Open Player Window")
        #self.label_player.place(x=170, y=50)

        #self.entry = tk.Entry(self.win)
        #self.entry.insert(0, '9')
        #self.entry.place(x=30, y=50)

        # Combobox creation
        self.variable = StringVar(self.win)
        self.variable.set("9")  # default value

        self.w = OptionMenu(self.win, self.variable, '2', '6', '8', '9')
        self.w.config(width=10)
        self.w.pack()

        self.num_players = 0
        self.global_res = []
        self.tree = Treeview(self.win, selectmode='browse')
        self.tree.place(x=30, y=80)

        vsb = Scrollbar(self.win, orient="vertical", command=self.tree.yview)
        vsb.place(x=132, y=80, height=200 + 20)

        self.tree.configure(yscrollcommand=vsb.set)

        self.tree["columns"] = ("1")
        self.tree['show'] = 'headings'
        self.tree.column("1", width=102, anchor='c')
        self.tree.heading("1", text="Hand Number")

        #self.tree_player = Treeview(self.win, selectmode='browse')
        #self.tree_player.place(x=330, y=80)

        #vsb1 = Scrollbar(self.win, orient="vertical", command=self.tree_player.yview)
        #vsb1.place(x=530, y=80, height=200 + 20)

        #self.tree_player.configure(yscrollcommand=vsb1.set)

        #self.tree_player["columns"] = ("1", "2")
        #self.tree_player['show'] = 'headings'
        #self.tree_player.column("1", width=100, anchor='c')
        #self.tree_player.column("2", width=100, anchor='c')
        #self.tree_player.heading("1", text="Action")
        #self.tree_player.heading("2", text="Time")
        #self.tree_player.update_idletasks()
        #self.tree_player.yview_moveto('1.0')

        self.btn_start = Button(self.win, text="Start", width=15, command=self.main_start)
        self.btn_start.pack(anchor=CENTER, expand=True)
        self.btn_start.place(x=30, y=320)

        self.btn_end = Button(self.win, text="End", width=15, command=self.main_end)
        self.btn_end.pack(anchor=CENTER, expand=True)
        self.btn_end.place(x=30, y=350)

        self.label_player_selected = Label(self.win)
        self.label_player_selected.place(x=330, y=50)

        self.btn_1 = Button(self.win, text="1", width=5, command=self.click_player1)
        self.btn_1.pack(anchor=CENTER, expand=True)
        self.btn_1.place(x=155, y=80)

        self.btn_2 = Button(self.win, text="2", width=5, command=self.click_player2)
        self.btn_2.pack(anchor=CENTER, expand=True)
        self.btn_2.place(x=205, y=80)

        self.btn_3 = Button(self.win, text="3", width=5, command=self.click_player3)
        self.btn_3.pack(anchor=CENTER, expand=True)
        self.btn_3.place(x=255, y=80)

        self.btn_4 = Button(self.win, text="4", width=5, command=self.click_player4)
        self.btn_4.pack(anchor=CENTER, expand=True)
        self.btn_4.place(x=155, y=130)

        self.btn_5 = Button(self.win, text="5", width=5, command=self.click_player5)
        self.btn_5.pack(anchor=CENTER, expand=True)
        self.btn_5.place(x=205, y=130)

        self.btn_6 = Button(self.win, text="6", width=5, command=self.click_player6)
        self.btn_6.pack(anchor=CENTER, expand=True)
        self.btn_6.place(x=255, y=130)

        self.btn_7 = Button(self.win, text="7", width=5, command=self.click_player7)
        self.btn_7.pack(anchor=CENTER, expand=True)
        self.btn_7.place(x=155, y=180)

        self.btn_8 = Button(self.win, text="8", width=5, command=self.click_player8)
        self.btn_8.pack(anchor=CENTER, expand=True)
        self.btn_8.place(x=205, y=180)

        self.btn_9 = Button(self.win, text="9", width=5, command=self.click_player9)
        self.btn_9.pack(anchor=CENTER, expand=True)
        self.btn_9.place(x=255, y=180)


        self.th = threading.Thread(target=self.update)

        self.positions = {2: [[655, 250, 750, 270], [70, 250, 165, 270]],
                          8: [[540, 90, 640, 110], [645, 195, 745, 215], [645, 300, 745, 320], [515, 392, 620, 413], [205, 392, 310, 410], [75, 300, 180, 320], [80, 195, 180, 215], [185, 90, 280, 110]],
                          9: [[540, 90, 640, 110], [645, 195, 745, 215], [645, 300, 745, 320], [515, 392, 620, 413], [390, 422, 495, 443], [205, 392, 310, 410], [75, 300, 180, 320], [80, 195, 180, 215], [185, 90, 280, 110]],
                          6: [[538, 90, 640, 110], [650, 250, 755, 273], [515, 393, 615, 415], [205, 392, 310, 412], [70, 250, 170, 270], [185, 90, 285, 110]]}

        self.turns_p = {2: [[755, 238], [0, 238]],
                        6: [[645, 77], [755, 237], [621, 378], [137, 379], [10, 240], [114, 77]],
                        8: [[644, 76], [749, 183], [750, 286], [622, 380], [135, 378], [8, 285], [8, 183], [113, 78]],
                        9: [[644, 76], [749, 183], [750, 286], [622, 380], [321, 408], [135, 378], [8, 285], [8, 183], [113, 78]]}

        # load the class labels actions from disk
        self.rows = open('model/labels.txt').read().strip().split("\n")
        self.classes = [r[r.find(" ") + 1:].split(",")[0] for r in self.rows]
        self.flag_stop = True
        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv.dnn.readNetFromCaffe('model/deploy.prototxt', 'model/model.caffemodel')
        self.model = ModelRecog()
        self.model.prepare_for_prediction()

        # load the class labels cardnum from disk
        self.rows = open('model/cardnum_label.txt').read().strip().split("\n")
        self.classes_cardnum = [r[r.find(" ") + 1:].split(",")[0] for r in self.rows]
        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net_cardnum = cv.dnn.readNetFromCaffe('model/cardnum.prototxt', 'model/cardnum.model')


        #=======================================================================================================
        #Playing
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        self.title_game = self.get_gamename()
        # initialize the WindowCapture class
        self.wincap = WindowCapture(self.title_game)

        self.loop_time = time.time()
        self.image_id = 0
        self.image_id_old = 0
        self.clicked_btn = -1
        self.st = time.time()
        self.player_id_old = {}
        self.turn = 1
        self.turn_old = 1
        self.cardnum = 0
        self.cardnum_old = 0
        self.queue_actions = []
        self.queue_turns = []
        # ==============================

        self.res_player1_win = None
        self.res_player2_win = None
        self.res_player3_win = None
        self.res_player4_win = None
        self.res_player5_win = None
        self.res_player6_win = None
        self.res_player7_win = None
        self.res_player8_win = None
        self.res_player9_win = None

        # ==============================
        self.win.mainloop()

    def main_start(self):
        try:
            #res = messagebox.askquestion("askquestion", "Please check the seat number and opened the room, is it ok?")
            #if res == 'yes':
            #for it in self.tree_player.get_children():
            #    self.tree_player.delete(it)
            for it in self.tree.get_children():
                self.tree.delete(it)
            self.num_players = int(self.variable.get())
            self.flag_stop = True

            self.th = threading.Thread(target=self.update)
            self.th.start()
            # else:
            #    return

        except:
            self.flag_stop = False
            return

    def main_end(self):

        self.flag_stop = False
        #self.res_player1_win.quit_newwin()

        self.win.quit()
        self.win.destroy()
        print('ending')

    def click_player1(self):
        self.clicked_btn = 1
        self.label_player_selected.config(text='Player1')
        #self.show_player()
        self.res_player1_win = NewWindow()
        self.res_player1_win.show_window(self.global_res, self.clicked_btn)
    def click_player2(self):
        self.clicked_btn = 2
        self.label_player_selected.config(text='Player2')
        #self.show_player()
        self.res_player2_win = NewWindow()
        self.res_player2_win.show_window(self.global_res, self.clicked_btn)
    def click_player3(self):
        self.clicked_btn = 3
        self.label_player_selected.config(text='Player3')
        #self.show_player()
        self.res_player3_win = NewWindow()
        self.res_player3_win.show_window(self.global_res, self.clicked_btn)
    def click_player4(self):
        self.clicked_btn = 4
        self.label_player_selected.config(text='Player4')
        #self.show_player()
        self.res_player4_win = NewWindow()
        self.res_player4_win.show_window(self.global_res, self.clicked_btn)
    def click_player5(self):
        self.clicked_btn = 5
        self.label_player_selected.config(text='Player5')
        #self.show_player()
        self.res_player5_win = NewWindow()
        self.res_player5_win.show_window(self.global_res, self.clicked_btn)

    def click_player6(self):
        self.clicked_btn = 6
        self.label_player_selected.config(text='Player6')
        #self.show_player()
        self.res_player6_win = NewWindow()
        self.res_player6_win.show_window(self.global_res, self.clicked_btn)
    def click_player7(self):
        self.clicked_btn = 7
        self.label_player_selected.config(text='Player7')
        #self.show_player()
        self.res_player7_win = NewWindow()
        self.res_player7_win.show_window(self.global_res, self.clicked_btn)
    def click_player8(self):
        self.clicked_btn = 8
        self.label_player_selected.config(text='Player8')
        #self.show_player()
        self.res_player8_win = NewWindow()
        self.res_player8_win.show_window(self.global_res, self.clicked_btn)
    def click_player9(self):
        self.clicked_btn = 9
        self.label_player_selected.config(text='Player9')
        #self.show_player()
        self.res_player9_win = NewWindow()
        self.res_player9_win.show_window(self.global_res, self.clicked_btn)

    def show_player(self):
        self.res_player1_win = NewWindow()

    def get_turn(self, im, w, h, mask_circle):
        for i in range(len(self.turns_p[self.num_players])):
            x0, y0 = self.turns_p[self.num_players][i]
            img = im[y0:y0 + h, x0:x0 + w]
            # cv.imshow('turn', img)
            # cv.waitKey(0)
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, (36, 25, 25), (70, 255, 255))
            mask = cv.bitwise_and(mask, mask, mask=mask_circle)
            sum_pixels_all = 0
            for j in range(len(mask)):
                sum_pixels_all += sum(mask[j])

            if sum_pixels_all > 140000:
                self.turn = i
                break
        #print('turn: ', self.turn)


    def recog_cardnum(self, im):
        image = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        blob = cv.dnn.blobFromImage(image, 1, (100, 28), (154.21214))
        self.net_cardnum.setInput(blob)
        preds = self.net_cardnum.forward()
        idxs = np.argsort(preds[0])[::-1][:5]
        #la = ''
        #for (i, idx) in enumerate(idxs):
        la = self.classes_cardnum[idxs[0]]
        #print(la)
        return int(la)


    def update(self):
        image = cv.imread('circle.jpg')
        mask_circle = np.ones(image.shape[:2], dtype="uint8") * 255
        cv.circle(mask_circle, (35, 35), 28, 0, -1)
        w = h = 70

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        self.title_game = self.get_gamename()
        # initialize the WindowCapture class
        self.wincap = WindowCapture(self.title_game)
        st = time.time()
        changed_id = -1
        self.num_players = int(self.variable.get())
        handnum_res_old = 0
        hand_num_old = 0
        for id in range(self.num_players):
            self. player_id_old[id] = 'PLAYER'

        #cap = cv.VideoCapture('2min.mp4')

        while (True):

            if not self.flag_stop:
                break
            # time.sleep(1)
            # get an updated image of the game
            try:
                #ret, screenshot = cap.read()
                #screenshot = screenshot[37:,:]
                #cv.imshow('res', screenshot)
                #cv.waitKey(1)
                screenshot = self.wincap.get_screenshot()
            except:
                self.saving_csv()
                self.label_player_selected.config(text='')
                self.clicked_btn = 0
                #for it in self.tree_player.get_children():
                #    self.tree_player.delete(it)
                break

            screenshot = cv.resize(screenshot, (825, 570))

            self.image_id += 1
            #print(self.image_id)
            # debug the loop rate
            #try:
            #    print('FPS {}'.format(1 / (time.time() - self.loop_time)))
            #except:
            #    print()
            self.loop_time = time.time()

            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                break

            self.get_turn(screenshot, w, h, mask_circle)
            self.cardnum = self.recog_cardnum(screenshot[210:285, 280:545])


            cr_handnum = screenshot[10:23, 262:347]
            cr_handnum = cv.resize(cr_handnum, (int(cr_handnum.shape[1] * 36 / cr_handnum.shape[0]), 36))
            if self.image_id < 5:
                handnum_res = self.model.predict([cr_handnum / 255])
                handnum_res_old = handnum_res
            if self.image_id % 5 == 0:
                handnum_res = self.model.predict([cr_handnum / 255])
                handnum_res_old = handnum_res
            else:
                handnum_res = handnum_res_old
            player_id = {}
            special_action_flag = False
            #print(self.cardnum, self.cardnum_old)
            if self.queue_actions:
                if self.cardnum == 3 and self.cardnum_old == 0:
                    print('--------------------------')
                    print(self.queue_actions)
                    print('--------------------------')
                    print(self.cardnum, self.turn)
                    print(self.cardnum_old, self.turn_old)
                    print('--------------------------')
                    if self.queue_actions[-1] == 'CHECK':
                        player_id[self.turn_old] = 'CHECK'
                        self.queue_actions.append('CHECK')
                        special_action_flag = True
                    elif self.queue_actions[-1] == 'CALL':
                        player_id[self.turn_old] = 'CALL'
                        self.queue_actions.append('CALL')
                        special_action_flag = True
                    elif self.queue_actions[-1] == 'RAISE' or self.queue_actions[-1] == 'BET':
                        player_id[self.turn_old] = 'CALL'
                        self.queue_actions.append('CALL')
                        special_action_flag = True
                elif self.cardnum == 4 and self.cardnum_old == 3:
                    if self.queue_actions[-1] == 'CHECK':
                        player_id[self.turn_old] = 'CHECK'
                        self.queue_actions.append('CHECK')
                        special_action_flag = True
                    elif self.queue_actions[-1] == 'CALL':
                        player_id[self.turn_old] = 'CALL'
                        self.queue_actions.append('CALL')
                        special_action_flag = True
                    elif self.queue_actions[-1] == 'RAISE' or self.queue_actions[-1] == 'BET':
                        player_id[self.turn_old] = 'CALL'
                        self.queue_actions.append('CALL')
                        special_action_flag = True
                elif self.cardnum == 5 and self.cardnum_old == 4:
                    if self.queue_actions[-1] == 'CHECK':
                        player_id[self.turn_old] = 'CHECK'
                        self.queue_actions.append('CHECK')
                        special_action_flag = True
                    elif self.queue_actions[-1] == 'CALL':
                        player_id[self.turn_old] = 'CALL'
                        self.queue_actions.append('CALL')
                        special_action_flag = True
                    elif self.queue_actions[-1] == 'RAISE' or self.queue_actions[-1] == 'BET':
                        player_id[self.turn_old] = 'CALL'
                        self.queue_actions.append('CALL')
                        special_action_flag = True

            for id, p in enumerate(self.positions[self.num_players]):
                if id == self.turn_old and special_action_flag:
                    continue
                cr = screenshot[p[1]:p[3], p[0]:p[2]]
                recognized_label = self.recog(cr)
                player_id[id] = recognized_label
            if special_action_flag:
                print(self.player_id_old)
                print(player_id)

            for id in range(self.num_players):
                if player_id[id] == 'NONE':
                    continue

                if (self.player_id_old[id] == 'PLAYER' and player_id[id] != 'PLAYER') or \
                    (self.player_id_old[id] != 'PLAYER' and player_id[id] != 'PLAYER' and self.player_id_old[id] != player_id[id]):
                    if special_action_flag:
                        print(player_id[id], self.turn_old)
                    self.queue_actions.append(player_id[id])
                    if handnum_res == 0:
                        continue
                    #print('----------------------')
                    #print(id)
                    #print(self.player_id_old[id])
                    print(player_id[id], 'recognized')
                    t_elapsed = time.time() - st
                    t_elapsed = "{:.2f}".format(round(t_elapsed, 2))
                    if handnum_res != str(hand_num_old):
                        self.tree.insert("", 'end', text="L1", values=('{}'.format(handnum_res)))
                        hand_num_old = handnum_res
                    it = ['{}'.format(handnum_res), 'Player{}'.format(id + 1), '{}'.format(player_id[id]), '{}'.format(t_elapsed), self.image_id]
                    self.global_res.append(it)
                    if it[2] != 'SB' and it[2] != 'BB' and it[2] != 'DONTSHOW' and it[2] != 'SHOWCARDS' and it[2] != 'MUCK' and it[2] != 'RESERVED':
                        print(it)
                        t = float(it[3])
                        if self.image_id > self.image_id_old:
                            if it[1] == 'Player1' and self.res_player1_win:
                                self.res_player1_win.update_newwindow(it)
                            elif it[1] == 'Player2' and self.res_player2_win:
                                self.res_player2_win.update_newwindow(it)
                            elif it[1] == 'Player3' and self.res_player3_win:
                                self.res_player3_win.update_newwindow(it)
                            elif it[1] == 'Player4' and self.res_player4_win:
                                self.res_player4_win.update_newwindow(it)
                            elif it[1] == 'Player5' and self.res_player5_win:
                                self.res_player5_win.update_newwindow(it)
                            elif it[1] == 'Player6' and self.res_player6_win:
                                self.res_player6_win.update_newwindow(it)
                            elif it[1] == 'Player7' and self.res_player7_win:
                                self.res_player7_win.update_newwindow(it)
                            elif it[1] == 'Player8' and self.res_player8_win:
                                self.res_player8_win.update_newwindow(it)
                            elif it[1] == 'Player9' and self.res_player9_win:
                                self.res_player9_win.update_newwindow(it)
                            #self.tree_player.insert("", 'end', text="L1", values=('{}'.format(it[2]), "%.1f" % round(t, 2)))
                            #self.tree_player.update_idletasks()
                            #self.tree_player.yview_moveto('1.0')
                            self.image_id_old = it[-1]
                    self.tree.update_idletasks()
                    self.tree.yview_moveto('1.0')
                    """
                    p = self.positions[self.num_players][id]
                    for i in range(5):
                        dx = random.randint(-2,2)
                        dy = random.randint(-2,2)
                        cr = screenshot[p[1]+dy:p[3]+dy, p[0]+dx:p[2]+dx]
                        cv.imwrite('data/{}_{}_{}.jpg'.format(self.image_id, i, player_id[id]), cr)
                    """
                    st = time.time()
                    changed_id = id
                    break

            self.player_id_old = player_id
            self.turn_old = self.turn
            self.cardnum_old = self.cardnum
            len_actions = len(self.queue_actions)
            if len_actions > 9:
                self.queue_actions = self.queue_actions[len_actions - 9:]

        self.flag_stop = False
        return

    def saving_csv(self):
        currentTime = time.strftime('%Y-%m-%d %H-%M-%S')
        application_path = os.path.dirname(sys.executable)
        print(application_path)
        path = application_path + '/Res/{}.csv'.format(str(currentTime))
        f = open(path, 'w')
        f.write('Hand NUmber,Player,Action,Time\n')
        id_v = -1
        for value in self.global_res:
            t = float(value[3])
            id_v += 1
            if self.num_players == 2:
                try:
                    if value[2] == 'FOLD' and self.global_res[id_v][0] == self.global_res[id_v + 1][0]:
                        continue
                except:
                    continue
            if value[2] == 'SB' or value[2] == 'BB' or value[2] == 'SHOWCARDS' or value[2] == 'DONTSHOW' or value[2] == 'MUCK' or value[2] == 'RESERVED':
                txt = ',,,\n'
            else:
                txt = '{},{},{},{}\n'.format(value[0], value[1], value[2], "%.1f" % round(t, 2))
            f.write(txt)
        for it in self.tree.get_children():
            self.tree.delete(it)
        f.close()
        self.global_res = []
        """
        currentTime = time.strftime('%Y-%m-%d %H-%M-%S')
        application_path = os.path.dirname(sys.executable)
        print(application_path)
        path = application_path + '/Res/{}.csv'.format(str(currentTime))
        f = open(path, 'w')
        res_json = {}
        for i in range(1, self.num_players + 1):
            res_json['Player{}'.format(i)] = []
        res_list = []
        oldid = 0
        oldhn = ''
        oldhn_flag = False
        for it in self.tree.get_children():
            hn = self.tree.item(it)['values'][0]
            if not oldhn_flag:
                oldhn = hn
                oldhn_flag = True
            sec = self.tree.item(it)['values'][3]
            playerid = self.tree.item(it)['values'][1]
            state = self.tree.item(it)['values'][2]
            pi = int(playerid.split('Player')[1])
            #self.tree.delete(it)
            if pi > oldid and oldhn == hn:
                for i in range(oldid + 1, pi): res_list.append(['Player{}'.format(i), "", "", hn])
                res_list.append([playerid, state, sec, hn])
                oldid = pi
                continue
            elif pi > oldid and oldhn != hn:
                for i in range(oldid + 1, self.num_players + 1): res_list.append(['Player{}'.format(i), "", "", hn])
                for i in range(1, pi):res_list.append(['Player{}'.format(i), "", "", hn])
                res_list.append([playerid, state, sec, hn])
                oldid = pi
                oldhn = hn
            elif pi <= oldid and oldhn == hn:
                for i in range(oldid + 1, self.num_players + 1): res_list.append(['Player{}'.format(i), "", "", hn])
                for i in range(1, pi):res_list.append(['Player{}'.format(i), "", "", hn])
                res_list.append([playerid, state, sec, hn])
                oldid = pi
                continue
            else:
                for i in range(oldid + 1, self.num_players + 1): res_list.append(['Player{}'.format(i), "", "", hn])
                for i in range(1, pi):res_list.append(['Player{}'.format(i), "", "", hn])
                res_list.append([playerid, state, sec, hn])
                oldid = pi
                oldhn = hn
                continue
        txt = ','
        for p in range(1, self.num_players + 1):
            txt += 'Player{},,'.format(p)
        f.write(txt + '\n')
        #print(res_list)
        old_handnum = res_list[0][-1]
        txt = '{},'.format(res_list[0][-1])
        id = 1
        for l in res_list:
            txt += "{},{},".format(l[1],l[2])
            if id == self.num_players:
                f.write(txt + '\n')
                txt = '{},'.format(l[-1])
                id = 1
                continue
            id += 1
        f.write(txt)
        f.close()
        """

    def get_gamename(self):
        z1 = pygetwindow.getAllTitles()
        for t in z1:
            if 'BTC' in t or 'BCH' in t:
                return t
        return ''

    def turn_recognition(self, img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        ## mask of green (36,25,25) ~ (86, 255,255)
        # mask = cv.inRange(hsv, (36, 25, 25), (86, 255,255))
        mask = cv.inRange(hsv, (36, 25, 25), (70, 255, 255))

        ## slice the green
        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]

    def recog(self, image):
        # load the input image from disk
        # im = cv.imread(file)
        # #image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        blob = cv.dnn.blobFromImage(image, 1, (100, 28), (52.013927, 48.681072, 45.881073))

        # set the blob as input to the network and perform a forward-pass to
        # obtain our output classification
        self.net.setInput(blob)
        start = time.time()
        preds = self.net.forward()
        end = time.time()
        # print("[INFO] classification took {:.5} seconds".format(end - start))

        # sort the indexes of the probabilities in descending order (higher
        # probabilitiy first) and grab the top-5 predictions
        idxs = np.argsort(preds[0])[::-1][:5]
        # loop over the top-5 predictions and display them
        la = ''
        for (i, idx) in enumerate(idxs):
            # draw the top prediction on the input image
            if i == 0:
                text = "Label: {}, {:.2f}%".format(self.classes[idx],
                                                   preds[0][idx] * 100)
                # cv.putText(image, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX,
                #            0.7, (0, 0, 255), 2)
            # display the predicted label + associated probability to the
            # console
            # print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1, classes[idx], preds[0][idx]))
            la = self.classes[idxs[0]]
        # display the output image
        # cv.imshow("Image", image)
        # cv.waitKey(0)
        return la


App("Demo")



