import cv2
import numpy as np
import os
from properties.common_property import *

def getInput():
    return os.listdir(VIDEO_INSTALLER_OUTPUT_PATH)

def createOutputFolder(outputPath):
    print(outputPath)
    try:
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
    except OSError:
        print ('Error: Creating directory of data')


def capture(inputPath, outputPath):
    createOutputFolder(outputPath)

    capture = cv2.VideoCapture(inputPath)

    currentFrame = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = capture.read()
        if not ret: 
            break

        # Saves image of the current frame in jpg file
        name = VIDEO_TO_IMAGE_SPLITTER_IMAGE_NAME.format(outputPath, str(currentFrame))
        print ('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    #When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()

def formatFileName(fileName):
    if fileName is not None:
        return fileName.split(VIDEO_TO_IMAGE_SPLITTER_VIDEO_SUFFIX)[0]

def main():
    inputList = getInput()
    if inputList is not None:
        for videoFileName in inputList:
            folderName = formatFileName(videoFileName) + "/"
            inputPath = VIDEO_INSTALLER_OUTPUT_PATH + videoFileName
            outputPath = VIDEO_TO_IMAGE_SPLITTER_OUTPUT_PATH + folderName 
           
            capture(inputPath, outputPath)

if __name__ == "__main__":
    main()
