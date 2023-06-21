# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import os, sys
from time import time
from windowcapture import WindowCapture

import pygetwindow
import time
import tensorflow as tf
import random
import argparse

positions = {2: [[655, 250, 750, 270], [70, 250, 165, 270]],
             8: [[540, 90, 640, 110], [645, 195, 745, 215], [645, 300, 745, 320], [515, 392, 620, 413],
                 [205, 392, 310, 410], [75, 300, 180, 320], [80, 195, 180, 215], [185, 90, 280, 110]],
             9: [[540, 90, 640, 110], [645, 195, 745, 215], [645, 300, 745, 320], [515, 392, 620, 413],
                 [390, 422, 495, 443], [205, 392, 310, 410], [75, 300, 180, 320], [80, 195, 180, 215],
                 [185, 90, 280, 110]],
             6: [[538, 90, 640, 110], [650, 250, 755, 273], [515, 393, 615, 415], [205, 392, 310, 412],
                 [70, 250, 170, 270], [185, 90, 285, 110]]}
turns_p = {2: [[755, 238], [0, 238]],
           6: [[645, 77], [755, 237], [621, 378], [137, 379], [10, 240], [114, 77]],
           8: [[644, 76], [749, 183], [750, 286], [622, 380], [135, 378], [8, 285], [8, 183], [113, 78]],
           9: [[644, 76], [749, 183], [750, 286], [622, 380], [321, 408], [135, 378], [8, 285], [8, 183], [113, 78]]}
# load the class labels from disk
rows = open('model/labels.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
# load our serialized model from disk
print("[INFO] loading model...")
net = cv.dnn.readNetFromCaffe('model/deploy.prototxt', 'model/model.caffemodel')

rows = open('model/cardnum_label.txt').read().strip().split("\n")
classes_cardnum = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
# load our serialized model from disk
print("[INFO] loading model...")
net_cardnum = cv.dnn.readNetFromCaffe('model/cardnum.prototxt', 'model/cardnum.model')

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

model_recog_dir = 'model'
model_recog_name = 'model_recog'
model_recog_pb_file = model_recog_name + '.pb'
height_norm = 36
alphabet = '''0123456789#'''
alphabet_blank = '`'

queue_actions = []
turn = 1
turn_old = 1
cardnum = 0
cardnum_old = 0


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


def get_gamename():
    z1 = pygetwindow.getAllTitles()
    for t in z1:
        if 'BTC' in t:
            return t
    return ''


def recog(im):
    # load the input image from disk
    # im = cv.imread(file)
    # image = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

    blob = cv.dnn.blobFromImage(im, 1, (100, 28), (52.013927, 48.681072, 45.881073))

    # set the blob as input to the network and perform a forward-pass to
    # obtain our output classification
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    idxs = np.argsort(preds[0])[::-1][:5]
    la = ''
    for (i, idx) in enumerate(idxs):
        # draw the top prediction on the input image
        if i == 0:
            text = "Label: {}, {:.2f}%".format(classes[idx],
                                               preds[0][idx] * 100)

        la = classes[idxs[0]]
    return la


def get_turn(im, w, h, mask_circle, num_players):
    for i in range(len(turns_p[num_players])):
        x0, y0 = turns_p[num_players][i]
        if x0 >= 755:
            x0 = 754
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
            turn = i
            return turn

    return 1


def recog_cardnum(im):
    image = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    blob = cv.dnn.blobFromImage(image, 1, (100, 28), (154.21214))
    net_cardnum.setInput(blob)
    preds = net_cardnum.forward()
    idxs = np.argsort(preds[0])[::-1][:5]
    # la = ''
    # for (i, idx) in enumerate(idxs):
    la = classes_cardnum[idxs[0]]
    # print(la)
    return int(la)


def detect_lobby(img_color, template):
    w, h = template.shape[::-1]
    # img_color = cv.imread(file)
    img = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    img2 = img.copy()
    meth = 'cv.TM_SQDIFF_NORMED'
    # meth = 'cv.TM_SQDIFF_NORMED'
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # if top_left[0]>img.shape[1]*0.1 or top_left[1]>img.shape[0]*0.1:
    #    return "None"

    cv.rectangle(img_color, top_left, bottom_right, (0, 255, 0), 2)
    cv.imshow("result", img_color)
    cv.waitKey(0)
    # rect = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
    return [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]


def get_points(top_left_x, top_right_x, lobby_left_y, lobby_right_y):
    rat = (top_right_x - top_left_x) / 825
    bottom_left_y = int(lobby_left_y + rat * 570)
    return [top_left_x, lobby_left_y, top_right_x, bottom_left_y]


def get_ratio_rects():
    ratios = []
    rects = []
    file = "test.png"
    frame = cv.imread(file)
    template1 = cv.imread('lobby1.png', 0)
    template2 = cv.imread('lobby2.png', 0)
    frame_w = frame.shape[1]
    frame_h = frame.shape[0]
    for i in range(4):
        res = detect_lobby(frame, template1)
        top_left_x = res[0]
        lobby_left_y = res[1]
        res = detect_lobby(frame, template2)
        bottom_right_x = res[2]
        lobby_right_y = res[3]
        ratio = 825 / (bottom_right_x - top_left_x)
        new_w = int(ratio * frame_w)
        new_h = int(ratio * frame_h)
        print(ratio)
        ratios.append(ratio)
        resized_frame = cv.resize(frame, (new_w, new_h))
        rect = get_points(top_left_x, bottom_right_x, lobby_left_y, lobby_right_y)
        crop = resized_frame[int(ratio * lobby_left_y):int(ratio * rect[3]),
               int(ratio * top_left_x):int(ratio * rect[2])]
        x0 = int(ratio * top_left_x)
        x1 = int(ratio * rect[2])
        y0 = int(ratio * lobby_left_y)
        y1 = int(ratio * rect[3])
        rects.append([x0, y0, x1, y1])
        cv.imshow('res', crop)
        cv.waitKey(0)
    return ratios, rects


model = ModelRecog()
model.prepare_for_prediction()


def main_proc(num_players, videofileid):
    videofile = 'output/{}.avi'.format(videofileid)
    global model, queue_actions, turn, turn_old, cardnum, cardnum_old
    num_players = int(num_players)
    w = h = 70
    image = cv.imread('circle.jpg')
    mask_circle = np.ones(image.shape[:2], dtype="uint8") * 255
    cv.circle(mask_circle, (35, 35), 28, 0, -1)
    global_res = []
    cap = cv.VideoCapture(videofile)
    if (cap.isOpened() == False):
        print("Unable to read camera feed")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps == 0:
        t_per_frame = 0
    else:
        t_per_frame = 1 / fps
    loop_time = time.time()
    image_id = 0
    st = 0
    changed_id = -1
    st_frame = 0
    player_id_old = {}
    handnum_res_old = 0
    for id in range(num_players):
        player_id_old[id] = 'PLAYER'
    handnumimageid = 0
    while (cap.isOpened()):
        ret, screenshot = cap.read()
        if not ret:
            break
        # screenshot = cv.resize(screenshot, (825, 570))
        cv.imshow('Computer Vision {}'.format(videofileid), screenshot)
        # cv.imwrite('images_origin/14/{}.jpg'.format(image_id), screenshot)
        image_id += 1
        #if image_id % 5 != 0:
        #    continue
        t_frames = image_id * t_per_frame
        # debug the loop rate
        # print('FPS {}'.format(1 / (time.time() - loop_time)))
        loop_time = time.time()

        # press 'q' with the output window focused to exit.
        # waits 1 ms every loop to process key presses
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            saving_csv(global_res, videofileid)
            break
        turn = get_turn(screenshot, w, h, mask_circle, num_players)
        crop_number = screenshot[210:285, 280:545]
        # cv.imshow('number_card', crop_number)
        cardnum = recog_cardnum(crop_number)
        cr_handnum = screenshot[9:24, 260:353]
        # cv.imshow('number_hand', cr_handnum)
        # cv.imwrite('num_hands/handnum_{}_{}.jpg'.format(videofileid, handnumimageid), cr_handnum)
        handnumimageid += 1
        cr_handnum = cv.resize(cr_handnum, (int(cr_handnum.shape[1] * 36 / cr_handnum.shape[0]), 36))
        if image_id < 5:
            handnum_res = model.predict([cr_handnum / 255])
            handnum_res_old = handnum_res
        if image_id % 5 == 0:
            handnum_res = model.predict([cr_handnum / 255])
            handnum_res_old = handnum_res
        else:
            handnum_res = handnum_res_old
        player_id = {}
        special_action_flag = False
        # print(self.cardnum, self.cardnum_old)
        if queue_actions:
            if cardnum == 3 and cardnum_old == 0:

                if queue_actions[-1] == 'CHECK':
                    player_id[turn_old] = 'CHECK'
                    queue_actions.append('CHECK')
                    special_action_flag = True
                elif queue_actions[-1] == 'CALL':
                    player_id[turn_old] = 'CALL'
                    queue_actions.append('CALL')
                    special_action_flag = True
                elif queue_actions[-1] == 'RAISE' or queue_actions[-1] == 'BET':
                    player_id[turn_old] = 'CALL'
                    queue_actions.append('CALL')
                    special_action_flag = True
            elif cardnum == 4 and cardnum_old == 3:
                if queue_actions[-1] == 'CHECK':
                    player_id[turn_old] = 'CHECK'
                    queue_actions.append('CHECK')
                    special_action_flag = True
                elif queue_actions[-1] == 'CALL':
                    player_id[turn_old] = 'CALL'
                    queue_actions.append('CALL')
                    special_action_flag = True
                elif queue_actions[-1] == 'RAISE' or queue_actions[-1] == 'BET':
                    player_id[turn_old] = 'CALL'
                    queue_actions.append('CALL')
                    special_action_flag = True
            elif cardnum == 5 and cardnum_old == 4:
                if queue_actions[-1] == 'CHECK':
                    player_id[turn_old] = 'CHECK'
                    queue_actions.append('CHECK')
                    special_action_flag = True
                elif queue_actions[-1] == 'CALL':
                    player_id[turn_old] = 'CALL'
                    queue_actions.append('CALL')
                    special_action_flag = True
                elif queue_actions[-1] == 'RAISE' or queue_actions[-1] == 'BET':
                    player_id[turn_old] = 'CALL'
                    queue_actions.append('CALL')
                    special_action_flag = True

        for id, p in enumerate(positions[num_players]):
            cr = screenshot[p[1]:p[3], p[0]:p[2]]
            recognized_label = recog(cr)
            player_id[id] = recognized_label

        for id in range(num_players):
            if player_id[id] == 'NONE':
                continue

            if (player_id_old[id] == 'PLAYER' and player_id[id] != 'PLAYER') or \
                    (player_id_old[id] != 'PLAYER' and player_id[id] != 'PLAYER' and player_id_old[id] !=
                     player_id[id]):
                if special_action_flag:
                    print(player_id[id], turn_old)
                queue_actions.append(player_id[id])
                if handnum_res == 0:
                    continue
                print(player_id)
                t_elapsed = t_frames - st
                if t_elapsed <= t_per_frame * 3:
                    continue
                t_elapsed = "{:.2f}".format(round(t_elapsed, 2))
                it = ['{}'.format(handnum_res), 'Player{}'.format(id + 1), '{}'.format(player_id[id]),
                      '{}'.format(t_elapsed), image_id]
                global_res.append(it)
                st = t_frames
                changed_id = id
                break

        player_id_old = player_id
        turn_old = turn
        cardnum_old = cardnum
        len_actions = len(queue_actions)
        if len_actions > 9:
            queue_actions = queue_actions[len_actions - 9:]

    cap.release()
    saving_csv(global_res, videofileid)


def saving_csv(global_res, videoid):
    currentTime = time.strftime('%Y-%m-%d %H-%M-%S')
    path = 'Res/{}_{}.csv'.format(videoid, str(currentTime))
    if not os.path.isdir('Res'):
        os.mkdir('Res')
    # path = 'Res/{}.csv'.format(str(currentTime))
    f = open(path, 'w')
    f.write('Hand NUmber,Player,Action,Time\n')

    for value in global_res:
        t = float(value[3])
        if value[2] == 'SB' or value[2] == 'BB' or value[2] == 'SHOWCARDS' or value[2] == 'DONTSHOW' or value[
            2] == 'MUCK' or value[2] == 'RESERVED':
            txt = ',,,\n'
        else:
            txt = '{},{},{},{}\n'.format(value[0], value[1], value[2], "%.1f" % round(t, 2))
        f.write(txt)

    f.close()
    global_res = []


ap = argparse.ArgumentParser()
ap.add_argument("-videoid", required=False, help="videoid")
#ap.add_argument("-f", required=False, help="videoid")


args = vars(ap.parse_args())

if __name__ == "__main__":
    # main(6, '0')
    #if args['videoid'] == '0' or args['videoid'] == '3':
    #    main_proc(9, args['videoid'])
        #main(6, 'output/1.avi')
    #else:
    #    main_proc(6, args['videoid'])
    f = open('output/seats.txt', 'r')
    lines = f.readlines()
    f.close()
    seat_num = int(lines[0].split(',')[2])
    main_proc(seat_num, 2)
