# IMPORTS
from PIL import Image

def get_img(TEST_PATH):
    """""
    Reads the image and resize it to 640x640
    Input:
        TEST_PATH: String
    Output:
        img: Image of shape ((640, 640, ,3))
    """""
    img = Image.open(TEST_PATH)
    img = img.resize((640, 640))
    return img

def return_pred(model ,img):
    res = model.predict(img)[0]

    boards = []
    numbers = []
    labels = []
    for box in res.boxes:
        num = str(res.names[int(box.cls[0])])
        if num == 'board':
            boards.append((box.xyxy, box.conf))
        elif num.isdigit() or num == ".":
            numbers.append((box.xyxy, box.conf, num))
        else: 
            labels.append((box.xyxy, box.conf, num))
    return boards, numbers, labels

def iou(xyxy1, xyxy2):

    x, y, x1, y1 = xyxy1
    x2, y2, x21, y21 = xyxy2

    overlap = 0
    if (x1 > x2) and (y1 > y2):
        numerator = (x1 - x2) * (y1 - y2)
        x_ = (x1 - x2) * (y1 - y)
        y_ = (x1 - x2) * (y21 * y2)
        denom = x_ * y_
        overlap = numerator / denom
    if overlap > 0:
        return True
    return False

def return_board(boards):

    # Return the one with the highest confidence
    boards = sorted(boards, key= lambda x: x[-1])
    return boards[0]

def return_nums(numbers, board):

    include = ''
    xb, yb, x1b, y1b = board[0][0]
    xb = int(xb) - 20
    yb = int(yb) - 20
    x1b = int(x1b) + 20
    y1b = int(y1b) + 20

    #Sort by conf
    n = sorted(numbers, key=lambda x: x[-1])

    #Sort by x1
    sorted_list = sorted(n, key=lambda x: x[0][0][0])

    # min_dist = 10
    #Remove overlapped detections
    unique_list = []
    xyxy = sorted_list[0][0][0]
    prevx, prevy, prevx1, prevy1 =xyxy
    unique_list.append(sorted_list[0])
    total_i = 0
    if sorted_list[0][-1] != ".":
        mean_width = prevx1 - prevx
    for i in range(1, len(sorted_list)):
        x, y, x1, y1 = sorted_list[i][0][0]
        # if not (sorted_list[i][-1] == '.' and abs(prevx1 - x) > 4) and (abs(prevx - x) < 5 or abs(prevy - y) < 5 or abs(prevx1 - x1) < 5 or abs(prevy1 - y1) < 5):
        # if not (sorted_list[i][-1] == '.' and abs(prevx1 - x) > 4) and (abs(prevx - x) < min_dist):
        #     continue
        if sorted_list[i][-1] != ".":
            mean_width += x1 - x
            total_i += 1

        if sorted_list[i][-1] != "." and iou([prevx, prevy, prevx1, prevy1],sorted_list[i][0][0]):
            continue
        prevx = x
        prevy = y
        prevx1 = x1
        prevy1 = y1
        unique_list.append(sorted_list[i])
    mean_width /= total_i
    # Put spacings respectively
    prevx = 0
    prevy = 0
    prevx1 = 0
    prevy1 = 0 
    included = []
    for each in unique_list:
        x, y, x1, y1 = each[0][0]
        if ((x - prevx1) > mean_width and prevx != 0) :
            if include != "":
                included.append(include)
                include = ""
        if (iou(board[0][0], each[0][0]) > 0.5) :
            include += each[2]
        prevx = x
        prevy = y
        prevx1 = x1
        prevy1 = y1

    if include != "":
        included.append(include)

    return included


def combine_labels(labels):

    labels = sorted(labels, key=lambda x: x[-1])
    labels = sorted(labels, key=lambda x: x[0][0][0])
    
    out = []
    prev = labels[0]
    out.append([prev[-1]])
    i = 0
    for label in labels[1:]:
        
        if (label[0][0][0] - prev[0][0][0]) < 5:
            out[-1].append(label[-1])
        else:
            out.append([label[-1]])
        prev = label

    return out

def reshape(num, maxm, minm):
    num = float(num)
    if num > maxm:
        while num > maxm:
            num /= 10
    elif num < minm:
        while num < minm:
            num *= 10

    return str(num)
# if __name__ == "__main__":
#     main()
