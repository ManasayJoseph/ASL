from keras.models import load_model
import cv2
import numpy as np
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./model/keras_Model.h5", compile=False)

# Load the labels
class_names = open("./model/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture('http://192.168.182.118:4747/video')

while True:
    # Grab the web camera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height, 224-width) pixels
    image_resized = cv2.resize(image, (224, 224), 1, interpolation=cv2.INTER_AREA)

    # Convert the resized image to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Apply additional processing steps as needed
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image_processed = cv2.threshold(th3, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Convert the processed grayscale image to a 3-channel image (add two dummy channels)
    image_rgb = cv2.cvtColor(image_processed, cv2.COLOR_GRAY2RGB)

    # Show the processed image in a window before prediction
    cv2.imshow("Processed Image", image_rgb)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image_rgb, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    
    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break
    

camera.release()
cv2.destroyAllWindows()
