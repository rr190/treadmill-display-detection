import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import numpy as np

def return_list(model, img):
    res = model.predict(img)[0]

    xyxy = {}

    for box in res.boxes:
        num = str(res.names[int(box.cls[0])])
        if num not in xyxy.keys():
            xyxy[num] = [(box.xyxy, box.conf)]
        else:
            xyxy[num].append((box.xyxy, box.conf))
        
    blank_img = np.array(img.copy())

    numbers = []
    label_list = []
    for key in xyxy.keys():

        label = key
        
        for item in xyxy[key]:
            x, y, x1, y1 = item[0][0]
            x = int(x)
            y = int(y)
            x1 = int(x1)
            y1 = int(y1)
            
            if not label.isdigit() and label != 'board' and label != '.':
                label_list.append((item[0][0], item[1], label))
            elif label != 'board':
                numbers.append((item[0][0], item[1], label))
            blank_img = cv2.rectangle(blank_img, (int(x), int(y)), (int(x1), int(y1)), (0, 255, 0), 2)
            blank_img = cv2.putText(blank_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)

    return label_list, numbers, xyxy, blank_img
def return_lists(TEST_PATH, model):
    '''''
    Input:
        TEST_PATH: path to the images to be tested
        model: YOLOv8 .pt files

    Output:
        boards: (len(images), # of boards detected, 2)
            (tensor(xyxy), tensor(conf))
        numbers_list: (len(images), # of numbers detected, 3)
            (tensor(xyxy), tensor(conf), label)
        blank_imgs: (len(images), 640, 640, 3)
    '''''
    for root, _, filenames in os.walk(TEST_PATH):

        blank_imgs = []
        boards = []
        labels = []
        numbers_list = []
        for f in filenames:
            img = Image.open(os.path.join(root, f)).resize((640, 640))
            label_list, numbers, xyxy, blank_img = return_list(model, img)
            boards.append(xyxy['board'])
            labels.append(label_list)
            numbers_list.append(numbers)
            blank_imgs.append(blank_img)

    return boards, numbers_list, blank_imgs, labels

def plot_image(nrows, ncols, images, filename):
    '''''
    Input:
        nrows: # of rows for the plot
        ncols: # of cols for the plot
        images: array of images
        filename: filename for the plot

    Saves the plot
    '''''
    fig, _ = plt.subplots(nrows, ncols, figsize=(10, 7))

    for i in range(1, nrows*ncols+1):
        fig.add_subplot(nrows, ncols, i)
        plt.imshow(images[i-1])
        plt.axis("off")

    plt.savefig(filename)
    
def combine_numbers(numbers_list, boards):
    '''''
    Input:
       boards: (len(images), # of boards detected, 2)
            (tensor(xyxy), tensor(conf))
        numbers_list: (len(images), # of numbers detected, 3)
            (tensor(xyxy), tensor(conf), label)
    Output:
        included: list of combined strings
    '''''
    all_included = []
    for j in range(len(numbers_list)):

        b = boards[j]
        include = ''
        xb, yb, x1b, y1b = b[0][0][0]
        xb = int(xb) - 20
        yb = int(yb) - 20
        x1b = int(x1b) + 20
        y1b = int(y1b) + 20
        n = numbers_list[j]
        n = sorted(n, key=lambda x: x[0][0])
        sorted_list = sorted(n, key=lambda x: x[0][0])

        #Remove overlapped detections
        unique_list = []

        prevx =sorted_list[0][0][0]
        prevy =sorted_list[0][0][1]
        prevx1 =sorted_list[0][0][2]
        prevy1 =sorted_list[0][0][3]
        unique_list.append(sorted_list[0])
        for i in range(1, len(sorted_list)):
            x, y, x1, y1 = sorted_list[i][0]
            if not (sorted_list[i][2] == '.' and abs(prevx1 - x) > 4) and (abs(prevx - x) < 1 and abs(prevy - y) < 1 and abs(prevx1 - x1) < 1 and abs(prevy1 - y1) < 1):
                continue
            prevx = x
            prevy = y
            prevx1 = x1
            prevy1 = y1
            unique_list.append(sorted_list[i])

        # Put spacings respectively
        prevx = 0
        prevy = 0
        prevx1 = 0
        prevy1 = 0 
        included = []

        for each in unique_list:
            x, y, x1, y1 = each[0]
            if ((x - prevx1) > 20 and prevx != 0) :
                included.append(include)
                include = ""
            if (x >= xb and x1 <= x1b and y >= yb and y1<= y1b):
                include += each[2]
            prevx = x
            prevy = y
            prevx1 = x1
            prevy1 = y1

        if include != "":
            included.append(include)
        all_included.append(included)

    return all_included

def combine_labels(labels):

    combined_labels = []
    combined_extended_labels = []
    for each in labels:
        l = sorted(each, key=lambda x: x[0][0])
        combined_extended_labels.append(l)
        lbls = []
        # for e in l:
        #     if e[1] > .5:
        #         lbls.append(e[-1])
        lbls.append(l)
        combined_labels.append(lbls)
    return combined_extended_labels, combined_labels

# if __name__ == "__main__":
#     main()
