import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to your Personal Gym Tracker!👋")

st.markdown(
    """
    Gym Tracker uses a trained Yolov8 model to detect 
    and automatically log data from a treadmill display
    board. This auto retrieval can easen up your fitness
    tracking.

    *Note that it only works if there are following labels 
    on the display board.

    - The possible logged labels include: 
        - Calories
        - Incline
        - Distance
        - Pace
        - HeartRate
        - Time
        - Speed
    """
)

st.write("## How to use (Choose a File):")
st.write("Picture used:")
st.image("demo/demo.jpeg", width=200)
st.video("demo/demo.webm")