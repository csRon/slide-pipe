import argparse
import time
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import cv2
import numpy as np
import os
import pyperclip
import subprocess

# Function to be called when a new file is created
def on_new_file(event):
    '''
    Function to be called when a new file is created in the folder being monitored
    :param event: event object
    :return: None
    '''
    # print new file path
    print(event.src_path)

    # rectify the slide
    rectified_slide = filter_and_rectify_slide(event.src_path)

    # copy the rectified slide to the clipboard
    copy_to_clipboard(rectified_slide)

def copy_to_clipboard(image: np.ndarray):
    '''
    Function to copy the image to the clipboard
    :param image: image to be copied
    :return: None
    '''
    # Save the image to a temporary file (optional)
    cv2.imwrite('temp_image.png', image)

    # Now, copy the image to the clipboard using Pyperclip
    with open('temp_image.png', 'rb') as img_file:
        image_data = img_file.read()
        try:
            pyperclip.copy(image_data)
        except:
            subprocess.run(['xclip', '-selection', 'clipboard', '-t', 'image/png', '-i', 'temp_image.png'])

    # Optionally, remove the temporary file
    # os.remove('temp_image.png')

def filter_and_rectify_slide(image_path:str) -> np.ndarray:
    '''
    Function to filter and rectify the slide
    :param image_path: path to the image of the slide
    :return: rectified slide image 1024x576
    '''
    # Load the image
    image = cv2.imread(image_path)

    # Scale down the image by 50 percent
    # scaled_image = cv2.resize(image, None, fx=0.2, fy=0.2)
    # cv2.imshow("Scaled Image", scaled_image)
    # cv2.waitKey(0)

    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((3,3),np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations= 3)
    # cv2.imshow("Closed", closed_image)
    # cv2.waitKey(0)

    # Convert the scaled image to grayscale
    gray = cv2.cvtColor(closed_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    # cv2.imshow("Blurred image", blurred)
    # cv2.waitKey(0)

    # Apply thresholding to isolate white areas
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Thresholded", thresh)
    # cv2.waitKey(0)

    canny = cv2.Canny(thresh, 30, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # cv2.imshow("Edges", canny)  
    # cv2.waitKey(0)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(scaled_image, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("Contours", scaled_image)
    # cv2.waitKey(0)

    # Assuming the largest contour corresponds to the slide
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the corners of the slide
    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Assuming the slide has four corners, get the coordinates
    if len(approx) == 4:
        # transform approx to (4,2) shape
        approx = approx.reshape((4,2))
        
        rect = np.zeros((4, 2), dtype="float32")

        # Calculate the sum of x and y coordinates for each point
        sums = np.sum(approx, axis=1)

        # Find the index of the point with the smallest sum of x and y coordinates
        bottom_left_index = np.argmin(sums)

        # assign corners to rect based on index starting from bottom left and going clockwise
        for i in range(4):
            index = (bottom_left_index + i) % 4
            rect[i] = approx[index]
    else:
        # If the slide is not detected, return the scaled base image
        # scale down image to 40% of original
        scaled_image = cv2.resize(image, None, fx=0.4, fy=0.4)
        return scaled_image

    # Define the four corners of the rectified slide
    # (assuming a standard size of the slide, adjust accordingly)
    rectified_corners = np.array([[0, 0], [0, 576],[1024, 576], [1024, 0]], dtype="float32")

    # Compute the perspective transform matrix
    perspective_transform = cv2.getPerspectiveTransform(rect, rectified_corners)

    # Apply the perspective transform to rectify the slide
    rectified_slide = cv2.warpPerspective(image, perspective_transform, (1024, 576))

    # Display the original and rectified images
    # cv2.imshow("Rectified Slide", rectified_slide)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return rectified_slide
       
def configure_observer(folder_path:str) -> PollingObserver:
    '''
    Function to configure the observer
    :param folder_path: path to the folder to be monitored
    :return: observer object
    '''
    # watch recursively for new files
    event_handler = FileSystemEventHandler()
    event_handler.on_created = on_new_file
    observer = PollingObserver()
    observer.schedule(event_handler, path=folder_path, recursive=True)
    return observer
    
def start_observer(observer:PollingObserver):
    '''
    Function to start the observer
    :param observer: observer object
    :return: None
    '''
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a folder for new file creations")
    parser.add_argument("folder", help="Path to the folder to monitor")
    args = parser.parse_args()

    folder_path_to_watch = args.folder
    print("Watching folder: {}".format(folder_path_to_watch))

    observer = configure_observer(folder_path_to_watch)
    start_observer(observer)