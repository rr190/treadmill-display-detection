Here is the link: 

This Gym Tracker uses a trained Yolov8 model to detect and automatically log data from a treadmill display board. This auto retrieval can easen up your fitness tracking.

### Collecting Data
- I used the dataset with 32 images that I took at my gym and used roboflow to augment the data. 
    - https://universe.roboflow.com/jr-s-nxihj/treadmill-display-detection 
- I used many different labels for the numbers (1 - 10) and display labels (Calories, Incline, etc.) and the 11-segment display board (which is where the numbers are at).

### Training 
- I trained the image many times with Yolov8 using Kaggle and saved the best weights.

### Deploy
- I used streamlit so that people can take a picture and upload picture to automatically log the numbers and labels. 