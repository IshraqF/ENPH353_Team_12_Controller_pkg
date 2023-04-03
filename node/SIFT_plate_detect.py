#!/usr/bin/env python3

import cv2
import numpy as np

# Minimum number of matches required to perform homography
MIN_MATCH_COUNT = 10

# @class My_App
#  @brief Main class for the object tracking application using homography
class My_App():
    # SLOT_query_camer - slot function for performing SIFT-based object detection using OpenCV.
    # @param self: reference to the object
    def SLOT_query_camera(self):
        """
        Slot function that reads the current frame from the camera device, converts it to grayscale,
        performs object tracking using SIFT and FLANN, and displays the result.
        """
        # Read the current frame from the camera device
        frame = cv2.imread(self.target)

        # Load the template image in grayscale
        image = cv2.imread(self.template, cv2.IMREAD_GRAYSCALE)

        # Create a SIFT object
        sift = cv2.SIFT_create()

        # Convert the current frame to grayscale and compute SIFT keypoints and descriptors for grayscale image and template image
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_grayscale, desc_grayscale = sift.detectAndCompute(grayscale, None)
        kp_image, desc_image = sift.detectAndCompute(image, None)

        # Create a FLANN matcher
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Match the descriptors using FLANN
        matches = flann.knnMatch(desc_image, desc_grayscale, k=2)

        # Keep only good matches based on Lowe's ratio test
        good_points = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_points.append(m)

        # If enough good matches are found, perform homography
        MIN_MATCH_COUNT = 10
        if len(good_points) > MIN_MATCH_COUNT:
            query_pts = np.float32(
                [kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32(
                [kp_grayscale[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(
                query_pts, train_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # Draw the object bounding box on the current frame
            h, w = image.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]
                             ).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            frame = cv2.polylines(
                frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            
            # Get the x, y coordinates of the top-left corner and bottom-right corner of the bounding box
            x1, y1 = np.int32(dst[0][0])
            x2, y2 = np.int32(dst[2][0])

            # Crop the frame to the bounding box
            cropped_frame = frame[y1:y2, x1:x2]

            # # Draw the matches on the current frame
            # draw_matches = dict(matchColor=(0, 255, 0),  # draw matches in green color
            #                     singlePointColor=None,
            #                     matchesMask=matchesMask,  # draw only inliers
            #                     flags=2)
            # frame = cv2.drawMatches(
            #     image, kp_image, frame, kp_grayscale, good_points, None, **draw_matches)

        # display in the Qt window
        cv2.imshow("Object Detect", frame)
        cv2.imshow("Cropped Image", cropped_frame)
        cv2.waitKey(1)

    def __init__(self):
        self.template = "sendout.png"
        self.target = "sendout_original.png"

# Main program execution.
if __name__ == "__main__":
    myApp = My_App()
    while True:
        myApp.SLOT_query_camera()