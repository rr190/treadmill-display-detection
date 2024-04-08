import streamlit as st
# import pymongo
# from pymongo.server_api import ServerApi
from ultralytics import YOLO
from util import get_img, return_pred, return_board, return_nums, combine_labels, reshape

st.title('Gym Tracker')
st.write("Record Your Miles!")

#Get Image Data
buf = st.camera_input("Take a picture!")
# buf = st.file_uploader("Choose a file")

#Predict
if buf is not None:
    model = YOLO("best3.pt")
    img = get_img(buf)

    #Retrieve Predicted Data
    boards, numbers, labels= return_pred( model, img)
    board = return_board(boards)
    combined_nums = return_nums(numbers, board)
    combined_labels = combine_labels(labels)
    print(combined_labels)
    print(combined_nums)

    col1, col2 = st.columns(2)
    labels = ["Time", "Calories", "Distance", "Pace","Speed", "Incline", "Heart-Rate"]
    num_restrain = [(100, 0), (1000, 0), (10, 0), (100, 0), (30, 0), (100, 0), (1000, 10)]
    # max_len = max(len(combined_labels),len(combined_nums)) 
    #Turn into a table and graph
    with col1:
        for i, each in enumerate(combined_labels):
            if each[0] == "Time Elapsed" or each[0] == "Time-Remaining":
                each[0] = "Time"
            idx = labels.index(each[0])
            if i < len(combined_nums):
                combined_nums[i] = reshape(combined_nums[i], num_restrain[idx][0], num_restrain[idx][1])
            option = st.selectbox(
            "Label", options=labels, index= idx, key=i, label_visibility="hidden")


    with col2:
        for each in combined_nums:
            num = st.number_input(
                "Number", float(each) ,label_visibility="hidden"
            )
