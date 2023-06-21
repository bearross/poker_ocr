import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture

import pygetwindow
import time
import os
import pyautogui
import PIL
import argparse

positions = {2:[[70,250,165,270],[655,250,750,270]],
             8:[[185,90,280,110],[80,195,180,215],[75,300,180,320],[205,392,310,410],[515,392,620,413],[645,300,745,320],[645,195,745,215],[540,90,640,110]],
             9:[[185,90,280,110],[80,195,180,215],[75,300,180,320],[205,392,310,410],[390,422,495,443],[515,392,620,413],[645,300,745,320],[645,195,745,215],[540,90,640,110]],
             6:[[185,90,285,110],[70,250,170,270],[205,392,310,412],[515,393,615,415],[650,250,755,273],[538,90,640,110]]}

# load the class labels from disk
rows = open('model/labels.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
# load our serialized model from disk
print("[INFO] loading model...")
net = cv.dnn.readNetFromCaffe('model/deploy.prototxt', 'model/model.caffemodel')


def get_gamename():
    z1 = pygetwindow.getAllTitles()
    for t in z1:
        if 'BTC' in t:
            return t
    return ''

def recog(im):
    # load the input image from disk
    #im = cv2.imread(file)
    image = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

    blob = cv.dnn.blobFromImage(image, 1, (100, 28), (48.625359))

    # set the blob as input to the network and perform a forward-pass to
    # obtain our output classification
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    #print("[INFO] classification took {:.5} seconds".format(end - start))

    # sort the indexes of the probabilities in descending order (higher
    # probabilitiy first) and grab the top-5 predictions
    idxs = np.argsort(preds[0])[::-1][:5]
    # loop over the top-5 predictions and display them
    la = ''
    for (i, idx) in enumerate(idxs):
        # draw the top prediction on the input image
        if i == 0:
            text = "Label: {}, {:.2f}%".format(classes[idx],
                                               preds[0][idx] * 100)
            #cv.putText(image, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX,
            #            0.7, (0, 0, 255), 2)
        # display the predicted label + associated probability to the
        # console
        #print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1, classes[idx], preds[0][idx]))
        la = classes[idxs[0]]
    # display the output image
    #cv.imshow("Image", image)
    #cv.waitKey(0)
    return la

def main(num_palyers, imagename):
    im = cv.imread(imagename)
    im = cv.resize(im, (825, 570))
    screenshot = im

    cv.imshow('Computer Vision', screenshot)

    player_id = {}
    for id, p in enumerate(positions[num_palyers]):
        cr = screenshot[p[1]:p[3],p[0]:p[2]]
        recognized_label = recog(cr)
        player_id[id] = recognized_label
    print(player_id)
    cv.waitKey(0)

ap = argparse.ArgumentParser()
ap.add_argument("-n", required=False, help="number of person")
ap.add_argument("-f", required=False, help="file name")

args = vars(ap.parse_args())

if __name__ == "__main__":

    #imagename = 'test/2/1.jpg'
    #num_player = 2
    #main(num_player, im)
    main(int(args['n']), args['f'])




print('Done.')