from keras.models import load_model
from StarPlus.HandDetector import HandDetector
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./model/keras_Model.h5", compile=False)

# Load the labels
class_names = open("./model/labels.txt", "r").readlines()

# Initialize HandDetector
handD = HandDetector()

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)
value = 70
while True:
    # Grab the web camera's image.
    ret, image = camera.read()

    # Detect hands in the image
    hands = handD.findHands(img=image, draw=False)

    if hands:
        bbox = hands[0]["bbox"]
        x, y, w, h = bbox

        # Calculate the center of the hand bounding box
        center_x, center_y = x + w // 2, y + h // 2

        # Calculate the side length for a square region around the hand
        side_length = max(w, h)

        # Calculate the new bounding box for the square region
        x_new = max(0, center_x - side_length // 2 -20 )
        y_new = max(0, center_y - side_length // 2 -20)
        w_new = min(image.shape[1] - x_new, side_length+ 20)
        h_new = min(image.shape[0] - y_new, side_length +30)

        # Extract the square region of interest (ROI) from the original image
        hand_roi = image[y_new:y_new + h_new, x_new:x_new + w_new]

        # Apply processing steps to the hand ROI (e.g., make it grayscale, use filters)
        # gray_hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        # blurred_hand_roi = cv2.GaussianBlur(gray_hand_roi, (5, 5), 2)
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)

        # Apply additional processing steps as needed
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        image_processed = cv2.threshold(th3,value, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
        # Resize the processed hand ROI to match the expected input shape of the model
        resized_hand_roi = cv2.resize(image_processed, (224, 224), interpolation=cv2.INTER_AREA)

        # Ensure the resized image has 3 channels (BGR or RGB)


        # Display the processed hand ROI
        
        cv2.imshow("Processed Hand ROI", resized_hand_roi)
        resized_hand_roi = cv2.cvtColor(resized_hand_roi, cv2.COLOR_GRAY2BGR)
        # Make the resized hand ROI a numpy array and reshape it to the model's input shape.
        processed_hand_roi = np.asarray(resized_hand_roi, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the processed hand ROI array if needed
        processed_hand_roi = (processed_hand_roi / 127.5) - 1

        # Predict the model with the processed hand ROI
        prediction = model.predict(processed_hand_roi)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:])
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2])

        # Display the original image with the hand bounding box
        cv2.rectangle(image, (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 255, 0), 2)
        cv2.imshow("Original Image with Hand Bounding Box", image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
