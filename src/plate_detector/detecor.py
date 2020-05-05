import os.path

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


class PlateDetector2:

    def __init__(self) -> None:
        super().__init__()
        self.plate_images = []
        self.confThreshold = 0.5  # Confidence threshold
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold

        self.inpWidth = 416  # 608     #Width of network's input image
        self.inpHeight = 416  # 608     #Height of network's input image

        # Load names of classes
        self.classesFile = "src/plate_detector/classes.names"

        self.classes = None
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.

        self.modelConfiguration = "src/plate_detector/darknet-yolov3.cfg"
        self.modelWeights = "src/plate_detector/lapi.weights"

        self.net = cv.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def get_found_plates(self):
        return self.plate_images

    # Get the names of the output layers
    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert (classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                     (0, 0, 255), cv.FILLED)
        # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                # if detection[4]>0.001:
                scores = detection[5:]
                classId = np.argmax(scores)
                # if scores[classId]>confThreshold:
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        result_boxes = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            result_boxes.append(box)
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

        return result_boxes

    def detect(self, path, is_video=False):
        # Process inputs
        winName = 'Deep learning object detection in OpenCV'
        cv.namedWindow(winName, cv.WINDOW_NORMAL)

        # Open the image file
        if not os.path.isfile(path):
            print("Input file ", path, " doesn't exist")
            return
        cap = cv.VideoCapture(path)

        # Get the video writer initialized to save the output video

        while cv.waitKey(1) < 0:
            # get frame from the video
            has_frame, frame = cap.read()

            # Stop the program if reached end of video
            if not has_frame:
                if is_video:
                    cv.waitKey(3000)
                break

            # cv.imshow('Video', frame)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Input frame')

            gray_car_image = rgb2gray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            binary_car_image = gray_car_image > threshold_otsu(gray_car_image)
            ax1.set_title('Grayscale frame')
            ax1.imshow(gray_car_image, cmap="gray")
            ax2.set_title('Binary frame')
            ax2.imshow(binary_car_image, cmap="gray")
            plt.show()
            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            self.net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = self.net.forward(self.getOutputsNames())

            # Remove the bounding boxes with low confidence
            counter = 0
            boxes = self.postprocess(frame, outs)
            if len(boxes) != 0:
                for box in boxes:
                    fig, (ax, ax_plate) = plt.subplots(1, 2)
                    fig.suptitle("Detected box #%d" % counter)
                    counter += 1
                    left = box[0]
                    top = box[1]
                    width = box[2]
                    height = box[3]
                    rect_border = patches.Rectangle((left, top), width, height,
                                                    edgecolor="red",
                                                    linewidth=2, fill=False)
                    ax.add_patch(rect_border)
                    plate_image = binary_car_image[top:top + height,
                    left:left + width]
                    ax_plate.imshow(plate_image, cmap="gray")
                    self.plate_images.append(plate_image)
                    ax.imshow(binary_car_image, cmap="gray")
                    plt.show()
            else:
                print("WARNING: Plate Not Found")

