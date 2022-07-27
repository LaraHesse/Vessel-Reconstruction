'''
This program reconstructs the 3D geometry of a simulated blood vessel from a tracked 2D Ultrasound scan

Note: Need to change pathToData in main in order to run file
'''

import os
import re
import cv2
import glob
import errno
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix


'''
Implementation of a simple thresholding algorithm that will take a greyscale image/volume, along with an upper and 
lower intensity threshold and return a binary segmentation. Pixels/voxels that fall within the threshold range are 
assigned the value 1, and all others pizels/voxels are assigned the value 0.

Parameters:
    image (int array): greyscale image/volume
    lowerIntensity (int): lower intensity threshold
    upperIntensity (int): upper intensity threshold
Returns:
    binarySegmentation (int array): binary segmentation of greyscale image/volume
'''
def generateThresholdingSegmentation(image, lowerIntensity, upperIntensity):
    if len(image.shape) == 3:
        binarySegmentation = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

        for z in range(image.shape[0]):
            for y in range(image.shape[1]):
                for x in range(image.shape[2]):
                    if image[z][y][x] <= upperIntensity and image[z][y][x] >= lowerIntensity:
                        binarySegmentation[z][y][x] = 1
        return binarySegmentation


    binarySegmentation = np.zeros((image.shape[0], image.shape[1]))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x] <= upperIntensity and image[y][x] >= lowerIntensity:
                binarySegmentation[y][x] = 1
    
    return binarySegmentation

'''
Determines the minimum and maximum intensity pixel/voxel in a given image/volume. 
Helper function for the function regionGrowing(...) to determine potential initial seeds.

Parameters:
    image (int array): greyscale image/volume
Returns:
    maxIntensity (int): maximum intensity pixel/voxel in the image/volume
    minIntensity (int): minimum intensity pixel/voxel in the image/volume
'''  
def getMinMaxIntensityPixel(image):
    height, width = image.shape
    maxIntensity = float('-inf')
    minIntensity = float('inf')
    for y in range(height//2): # vessel appears in top half of image
        for x in range(width):
            if image[y][x] > maxIntensity:
                maxIntensity = image[y][x]
                maxPixel = (x,y)
            if image[y][x] < minIntensity:
                minIntensity = image[y][x]
                minPixel = (x,y)
    return maxPixel, minPixel


'''
Implementation of a simplified version of the region growing algorithm that will take a greyscale image, along 
with a list of the current seeds, and the maximum difference threshold. The function splits the image into 2 
regions, 1 region that is the object of interest, 1 region for the background. The returned segmentation is binary 
and the same dimensions as the original image. Pixels within the object are assigned the value 1 and pixels belonging 
to the background (as well as unassigned pixels) are assigned the value 0.

Parameters:
    image (int array): greyscale image
    seedList (pixel/voxel array): a list of current seeds (1 seed for object, 1 seed for background),
    maxDiff (int): maximum difference threshold for a pixel/voxel to join into a region
Returns:
    segmentedImage (int array): binary segmentation of greyscale image
'''
def generateRegionGrowingSegmentation(image, seedList, maxDiff):
    height, width = image.shape
    segmentedImage = np.zeros((height, width))

    neighbours = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    unallocatedPixels = []
    visited = np.zeros((height, width))

    backgroundRegion = np.zeros((height, width))
    backgroundRegion[seedList[0][1]][seedList[0][0]] = 1
    backgroundSum = image[seedList[0][1]][seedList[0][0]]
    backgroundLength = 1
    
    foregroundRegion = np.zeros((height, width))
    foregroundRegion[seedList[1][1]][seedList[1][0]] = 1
    foregroundSum = image[seedList[1][1]][seedList[1][0]]
    foregroundLength = 1


    for seed in seedList:
        for neighbour in neighbours:
            neighbourX = seed[0] + neighbour[0]
            neighbourY = seed[1] + neighbour[1]
            if (neighbourX >= 0) and (neighbourY >= 0) and (neighbourX < width) and (neighbourY < height):
                if visited[neighbourY][neighbourX] == 0:
                    unallocatedPixels.append((neighbourX,neighbourY))
                    visited[neighbourY][neighbourX] = 1

    while len(unallocatedPixels) > 0:
        pixel = unallocatedPixels.pop(0)
        currentX = pixel[0]
        currentY = pixel[1]

        backgroundDiff = abs(int(image[currentY][currentX]) - int(backgroundSum/backgroundLength))
        foregroundDiff = abs(int(image[currentY][currentX]) - int(foregroundSum/foregroundLength))

        if backgroundDiff <= maxDiff or foregroundDiff <= maxDiff:
            # Find which regions the current pixel borders
            adjacentRegions = {"background":[], "foreground":[]}
            closestAdjacentRegions = {"background":[], "foreground":[]}
            currentNeighbours = []
            for neighbour in neighbours:
                neighbourX = pixel[0] + neighbour[0]
                neighbourY= pixel[1] + neighbour[1]
                if (neighbourX >= 0) and (neighbourY >= 0) and (neighbourX < width) and (neighbourY < height):
                    if backgroundRegion[neighbourY][neighbourX] == 1:
                        if neighbour[0] == 0 or neighbour[1] == 0:
                            closestAdjacentRegions["background"].append((neighbourX,neighbourY))
                        adjacentRegions["background"].append((neighbourX,neighbourY))
                    elif foregroundRegion[neighbourY][neighbourX] == 1:
                        if neighbour[0] == 0 or neighbour[1] == 0:
                            closestAdjacentRegions["foreground"].append((neighbourX,neighbourY))
                        adjacentRegions["foreground"].append((neighbourX,neighbourY))
                    currentNeighbours.append((neighbourX,neighbourY))
            
            if len(closestAdjacentRegions["background"]) > 0 or len(closestAdjacentRegions["foreground"]) > 0:
                adjacentRegions = closestAdjacentRegions
            
            # If it borders both, we need to assign according to closest neighbours' border
            if len(adjacentRegions["background"]) > 0 and len(adjacentRegions["foreground"]) > 0:
                if len(adjacentRegions["background"]) > len(adjacentRegions["foreground"]) and backgroundDiff <= maxDiff:
                    backgroundRegion[currentY][currentX] = 1
                    backgroundSum += int(image[currentY][currentX])
                    backgroundLength += 1
                    segmentedImage[currentY][currentX] = 0
                elif len(adjacentRegions["foreground"]) and foregroundDiff <= maxDiff:
                    foregroundRegion[currentY][currentX] = 1
                    foregroundSum += int(image[currentY][currentX])
                    foregroundLength += 1
                    segmentedImage[currentY][currentX] = 1

            # If it only borders the background region, assign to background
            elif len(adjacentRegions["background"]) > 0 and backgroundDiff <= maxDiff:
                backgroundRegion[currentY][currentX] = 1
                backgroundSum += int(image[currentY][currentX])
                backgroundLength += 1
                segmentedImage[currentY][currentX] = 0
            
            # If it only borders the foreground region, assign to foreground
            elif len(adjacentRegions["foreground"]) > 0 and foregroundDiff <= maxDiff:
                foregroundRegion[currentY][currentX] = 1
                foregroundSum += int(image[currentY][currentX])
                foregroundLength += 1
                segmentedImage[currentY][currentX] = 1 


            # Add neighbours to unallocated if not already done
            for neighbour in currentNeighbours:
                neighbourX = neighbour[0]
                neighbourY = neighbour[1]
                if visited[neighbourY][neighbourX] == 0:
                    unallocatedPixels.append((neighbourX,neighbourY))
                    visited[neighbourY][neighbourX] = 1

    return segmentedImage


'''
Combines the slices of a 4D array into one volume/3D array

Parameters:
  arr (int array): 4D array to be combined into a volume
Returns:
  vol (int array): 3D array/volume
'''
def recombineVolume(arr):
    vol = np.zeros((arr.shape[0],arr.shape[1],arr.shape[2]))
    for z in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            for x in range(arr.shape[2]):
                val = [(1 if x > 0.5 else 0) for x in arr[z][y][x]]
                if val[0]==0 and val[1]==1:
                    vol[z][y][x] = 1
                elif val[0]==1 and val[1]==0:
                    vol[z][y][x] = 0
    return vol


'''
Applies a square linear filter to a given image. 
    Parameters:
        image (int array): an array of intensity values, which represents an image
        filter (int array): a square filter of any size
    Returns:
        enhancedImage (int array): an array of intensity values after a linear filter is applied
'''
def applyFilter(image, filter, numberOfTimes=1):
    enhancedImage = np.shape(image.shape)
    for n in range(numberOfTimes):
        enhancedImage = cv2.filter2D(image,-1,filter)  
    return enhancedImage


'''
Resizes and normalizes image. Uses loaded model to predict a segmentation of the resized and normalized image

Parameters:
    model: loaded model for prediction
    image (int array): greyscale image
Returns: 
    segmentedImage (int array): binary segmentation of greyscale image
'''
def generateUNetSegmentation(model, image):
    image = cv2.resize(image, (128,128))
    image = np.reshape(image, image.shape + (1,))
    image = image / image.max()
    segmentedImage = model.predict(image[None,:,:,:])
    segmentedImage = recombineVolume(segmentedImage)
    segmentedImage = segmentedImage.reshape((128,128))
    return cv2.resize(segmentedImage, (512, 512))
    

'''
Loads keras model from json config file and trained weights

Parameters:
    pathToData (string): path to root directory of data
Returns:
    model: an trained keras model for segmentation
'''
def loadModel(pathToData):
    with open(pathToData + "/Assignment4Data/UNet/unet.json",'r') as f:
        json = f.read()
    model = model_from_json(json)
    model.load_weights(pathToData + "/Assignment4Data/UNet/unet.h5")
    return model


'''
Loads ultrasound images. Calls generateThresholdingSegmentation, generateRegionGrowing, and generateUNetSegmentation 
to generate binary segmentations. Saves each segmentation

Parameters:
  pathToData (string): path to root directory of data
'''
def generateSegmentations(pathToData):
    model = loadModel(pathToData)
    try:
        os.makedirs(pathToData + "/segmentations/thresholding")
        os.makedirs(pathToData + "/segmentations/regionGrowing")
        os.makedirs(pathToData + "/segmentations/unet")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    for filename in sorted(glob.glob(pathToData + "/Assignment4Data/Test_Images/Ultrasound/*.png")):
        if "_segmentation" not in filename:
            image = np.array(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY))

            img = generateThresholdingSegmentation(image, 150, 255)
            plt.imsave(pathToData + "/segmentations/thresholding/" + os.path.splitext(os.path.basename(filename))[0] + "_segmentation.png", img, cmap="gray")

            maxPixel, minPixel = getMinMaxIntensityPixel(image)
            img = generateRegionGrowingSegmentation(image, [minPixel, maxPixel], 50)
            plt.imsave(pathToData + "/segmentations/regionGrowing/" + os.path.splitext(os.path.basename(filename))[0] + "_segmentation.png", img, cmap="gray")

            img = generateUNetSegmentation(model, image)
            plt.imsave(pathToData + "/segmentations/unet/" + os.path.splitext(os.path.basename(filename))[0] + "_segmentation.png", img, cmap="gray")
            

'''
Finds the largest contiguous segment in a segmented image. Returns a binary image with a black background, and
all pixels in the largest segment a value of 1. All smaller segments are marked as background

Parameters:
    image (int array): segmented image
    improvedSegmentation (bool): used for question 8 of the assignment "Refine your predictions". When true, applies various techniques
    on the segmentation to improve the accuracy of the
Returns:
    segmentedImage (int array): binary image containing largest segmentation
'''
def findLargestSegmentation(image, improvedSegmentation=False):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image.astype('uint8'), connectivity=4)
    sizes = stats[:, -1]
    maxLabel = 1
    if len(sizes) > 1:
        maxSize = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > maxSize:
                maxLabel = i
                maxSize = sizes[i]
    segmentedImage = np.zeros(output.shape)
    segmentedImage[output == maxLabel] = 1

    if improvedSegmentation and np.count_nonzero(segmentedImage == 1) < 100: # Set a minimum size for segmentation
        segmentedImage = np.zeros(output.shape)

    return segmentedImage.astype('int')  


'''
Takes in a binary image of a segmented object and finds the contours of the object within the image. Returns 
the contours as a numpy array of points in the image coordinate system. The function down-samples the 
number of points for more efficient computation

Parameters:
    image (int array): binary image of containing segmentation of an object
    improvedSegmentation (bool): used for question 8 of the assignment "Refine your predictions". When true, downsample the
    number of contour points to more closely match the number of contour points in the ground truth
Returns:
    points (tuple array): array of points as tuples that define the contour
'''
def findBorders(image, improvedSegmentation=False):
    contours, hierarchy = cv2.findContours(image.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    if len(contours) > 0:
        contour = contours[0]
    else:
        contour = contours
    
    if len(contour) > 1 and not improvedSegmentation:
        contour = contour[::2] # Half the number of contour points for faster computation

    if len(contour) > 10 and improvedSegmentation:
        skipBy = len(contour)//10
        contour = contour[::skipBy]

    points = []
    for point in contour:
        points.append((point[0][0], point[0][1]))
    return points


'''
Transforms the contour points of a single image and returns the coordinates for the same points in the RAS 
coordinate system that is used in 3D Slicer

Parameters:
    contour (tuple array): array of points as tuples that define the contour
    npyFilePath (string): name of relevent probeToReference transform
    imageToProbe (int array): matrix for the image to probe transform
    referenceToRAS (int array): matrix for the reference to RAS transform
Returns:
    transformedPoints (int array): numpy array of the transformed points in the RAS 
        coordinate system
'''
def transformPoints(contour, npyFilePath, pathToData, imageToProbe, referenceToRAS):
    probeToReference = np.load(pathToData + "/Assignment4Data/transforms/ProbeToReference/" + npyFilePath)
    if len(contour) == 0:
        return contour
    for i in range(len(contour)):
        point = [int(contour[i][0]), int(contour[i][1]), 0, 1]            
        transformedPoint = np.matmul(referenceToRAS, np.matmul(probeToReference, np.matmul(imageToProbe, point)))[:3]
        if i == 0:
            transformedPoints = transformedPoint[:3]
        else:
            transformedPoints = np.vstack((transformedPoints, transformedPoint))
    return transformedPoints


'''
Compares a predicted segmentation to the ground truth segmentation to determine the accuracyo of the 
predicted segmentation. Calculates accuracy by determining the number of true positives (ie. predicted segmentation
correctly classified the object as object), true negatives (ie. predicted segmentation correctly classified the background 
as background), false positives (ie. predicted segmentation classified object as background), and false negatives (ie.
predicted segmentation classified background as object)

Parameters:
    y_true (int array): ground truth segmentation as a 2D numpy array
    y_pred (int array): predicted segmentation as a 2D numpy array
Returns:
    accuracy (float): accuracy of the predicted segmentation
'''
def accuracy(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for x in range(y_true.shape[0]):
        for y in range(y_true.shape[1]):
            if y_true[x][y] == 1 and y_pred[x][y] == 1:
                tp += 1
            elif y_true[x][y] == 0 and y_pred[x][y] == 0: 
                tn += 1
            elif y_true[x][y] == 1 and y_pred[x][y] == 0:
                fp += 1
            else:
                fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


'''
Calculates the intersection-over-union of the ground truth segmentation to the predicted segmentation

Parameters:
    y_true (int array): ground truth segmentation as a 2D numpy array
    y_pred (int array): predicted segmentation as a 2D numpy array
Returns:
    iou (float): IoU of the predicted segmentation
'''
def IoU(y_true, y_pred):
    smooth = 1e-12
    intersection = K.get_value(K.sum(y_true * y_pred))
    sum_ = K.get_value(K.sum(y_true + y_pred))
    iou = K.constant((intersection + smooth) / (sum_ - intersection + smooth))
    return K.mean(iou)


'''
Calls relevent functions to generate contours for each segmented slice and combine all 
the transformed contour points into a single numpy array for the ground truth segmentation

Parameters:
    pathToData (string):  path to root directory of data
'''
def groundTruthReconstruction(pathToData):
    imageToProbeTransform = np.load(pathToData + "/Assignment4Data/transforms/ImageToProbe.npy")
    referenceToRASTransform = np.load(pathToData + "/Assignment4Data/transforms/ReferenceToRAS.npy")
    Test_Images_Ultrasound_Labels = pd.read_csv(pathToData + "/Assignment4Data/Test_Images/Ultrasound/Test_Images_Ultrasound_Labels.csv")
    ProbeToReferenceTimeStamps = pd.read_csv(pathToData + "/Assignment4Data/transforms/ProbeToReferenceTimeStamps.csv")

    contours = np.array([])
    for filename in sorted(glob.glob(pathToData + "/Assignment4Data/Test_Images/Ultrasound/*_segmentation.png")):
        segmentedImage = np.array(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY))
        imageNum = int(re.findall(r'\d+', os.path.basename(filename))[0])

        # Get border of segmentation
        contourImage = findBorders(segmentedImage)

        # Transform border
        timeStamp = Test_Images_Ultrasound_Labels.loc[Test_Images_Ultrasound_Labels['VesselModel-segmentation'] == (os.path.basename(filename))]['Time Recorded'].values[0]
        npyFilePath = ProbeToReferenceTimeStamps.loc[ProbeToReferenceTimeStamps['Time'] == (timeStamp)]['Filepath'].values[0]
          
        transformedContour = transformPoints(contourImage, npyFilePath, pathToData, imageToProbeTransform, referenceToRASTransform)
        if len(transformedContour) > 1:
            if len(contours) == 0:
                contours = transformedContour
            else:
                contours = np.vstack((contours, transformedContour))
    
    np.save(pathToData + "/groundTruthContours.npy", contours)


'''
Calls relevent functions to find the largest segmentation, generate contours for each segmented slice, and 
combines all  the transformed contour points into a single numpy array for the specified segmentation method

Parameters:
    pathToData (string):  path to root directory of data
'''
def predictedReconstruction(pathToData, segmentationMethod = "unet"):
    imageToProbeTransform = np.load(pathToData + "/Assignment4Data/transforms/ImageToProbe.npy")
    referenceToRASTransform = np.load(pathToData + "/Assignment4Data/transforms/ReferenceToRAS.npy")
    Test_Images_Ultrasound_Labels = pd.read_csv(pathToData + "/Assignment4Data/Test_Images/Ultrasound/Test_Images_Ultrasound_Labels.csv")
    ProbeToReferenceTimeStamps = pd.read_csv(pathToData + "/Assignment4Data/transforms/ProbeToReferenceTimeStamps.csv")

    contours = np.array([])
    avgAccuracy = 0
    avgIoU = 0
    for filename in sorted(glob.glob(pathToData + "/segmentations/" + segmentationMethod + "/*_segmentation.png")):
        image = np.array(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY))
        imageNum = int(re.findall(r'\d+', os.path.basename(filename))[0])

        # Get largest segmentation
        segmentedImage = findLargestSegmentation(image)

        # Get border of segmentation
        contourImage = findBorders(segmentedImage)

        # Transform border
        timeStamp = Test_Images_Ultrasound_Labels.loc[Test_Images_Ultrasound_Labels['VesselModel-segmentation'] == (os.path.basename(filename))]['Time Recorded'].values[0]
        npyFilePath = ProbeToReferenceTimeStamps.loc[ProbeToReferenceTimeStamps['Time'] == (timeStamp)]['Filepath'].values[0]
          
        transformedContour = transformPoints(contourImage, npyFilePath, pathToData, imageToProbeTransform, referenceToRASTransform)
        if len(transformedContour) > 1:
            if len(contours) == 0:
                contours = transformedContour
            else:
                contours = np.vstack((contours, transformedContour))

        # Get Accuracy and IoU
        basename = os.path.basename(filename)
        y_true = np.array(cv2.cvtColor(cv2.imread(pathToData + "/Assignment4Data/Test_Images/Ultrasound/" + basename), cv2.COLOR_BGR2GRAY))//255
        avgAccuracy += accuracy(y_true, segmentedImage)
        avgIoU += IoU(y_true, segmentedImage)
    
    np.save(pathToData + "/" + segmentationMethod +"Contours.npy", contours)

    numImages = len(glob.glob(pathToData + "/segmentations/" + segmentationMethod + "/*_segmentation.png"))
    avgAccuracy = avgAccuracy / numImages
    avgIoU = avgIoU / numImages
    print("Average accuracy of {0} segmentation: {1:.4f}\nAverage IoU of {0} segmentation: {2:.4f}".format(segmentationMethod, avgAccuracy, avgIoU))

    
'''
Loads ultrasound images. Applies preprocessing techniques on the images to improve prediction accuracy. Calls generateUNetSegmentation 
to generate binary segmentations. Saves each segmentation

Parameters:
  pathToData (string): path to root directory of data
'''
def generatedImprovedUNetSegmentations(pathToData):
    model = loadModel(pathToData)
    try:
        os.makedirs(pathToData + "/segmentations/improvedUNetSegmentations")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    for filename in sorted(glob.glob(pathToData + "/Assignment4Data/Test_Images/Ultrasound/*.png")):
        if "_segmentation" not in filename:
            image = np.array(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY))
            enhancedImage = applyFilter(image, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]), 10)
            enhancedImage = cv2.equalizeHist(enhancedImage)
            enhancedImage = cv2.GaussianBlur(enhancedImage,(7,7),cv2.BORDER_DEFAULT)
            img = generateUNetSegmentation(model, enhancedImage)
            plt.imsave(pathToData + "/segmentations/improvedUNetSegmentations/" + os.path.splitext(os.path.basename(filename))[0] + "_segmentation.png", img, cmap="gray")
            

'''
Calls relevent functions to find the largest segmentation, generate contours for each segmented slice, and 
combines all  the transformed contour points into a single numpy array for the improved U-Net segmentation method

Parameters:
    pathToData (string):  path to root directory of data
'''
def improvedUNetReconstruction(pathToData):
    imageToProbeTransform = np.load(pathToData + "/Assignment4Data/transforms/ImageToProbe.npy")
    referenceToRASTransform = np.load(pathToData + "/Assignment4Data/transforms/ReferenceToRAS.npy")
    Test_Images_Ultrasound_Labels = pd.read_csv(pathToData + "/Assignment4Data/Test_Images/Ultrasound/Test_Images_Ultrasound_Labels.csv")
    ProbeToReferenceTimeStamps = pd.read_csv(pathToData + "/Assignment4Data/transforms/ProbeToReferenceTimeStamps.csv")

    contours = np.array([])
    avgAccuracy = 0
    avgIoU = 0
    files = random.sample(sorted(glob.glob(pathToData + "/segmentations/improvedUNetSegmentations/*_segmentation.png")),250)
    for filename in files:
        image = np.array(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY))
        imageNum = int(re.findall(r'\d+', os.path.basename(filename))[0])

        # Get largest segmentation
        segmentedImage = findLargestSegmentation(image, improvedSegmentation=True)

        # Get border of segmentation
        contourImage = findBorders(segmentedImage, improvedSegmentation=True)

        # Transform border
        timeStamp = Test_Images_Ultrasound_Labels.loc[Test_Images_Ultrasound_Labels['VesselModel-segmentation'] == (os.path.basename(filename))]['Time Recorded'].values[0]
        npyFilePath = ProbeToReferenceTimeStamps.loc[ProbeToReferenceTimeStamps['Time'] == (timeStamp)]['Filepath'].values[0]
          
        transformedContour = transformPoints(contourImage, npyFilePath, pathToData, imageToProbeTransform, referenceToRASTransform)
        if len(transformedContour) > 1:
            if len(contours) == 0:
                contours = transformedContour
            else:
                contours = np.vstack((contours, transformedContour))

        # Get Accuracy and IoU
        basename = os.path.basename(filename)
        y_true = np.array(cv2.cvtColor(cv2.imread(pathToData + "/Assignment4Data/Test_Images/Ultrasound/" + basename), cv2.COLOR_BGR2GRAY))//255
        avgAccuracy += accuracy(y_true, segmentedImage)
        avgIoU += IoU(y_true, segmentedImage)
    
    np.save(pathToData + "/unetContours.npy", contours) # save as unetContours.npy to be able to view in Slicer module

    avgAccuracy = avgAccuracy / len(files)
    avgIoU = avgIoU / len(files)
    print("Average accuracy of improved U-Net segmentation: {:.4f}\nAverage IoU of improved U-Net segmentation: {:.4f}".format(avgAccuracy, avgIoU))





def main():
    # Paths to modify
    pathToData = ""

    generateSegmentations(pathToData)
    groundTruthReconstruction(pathToData)
    predictedReconstruction(pathToData, "thresholding")
    predictedReconstruction(pathToData, "regionGrowing")
    predictedReconstruction(pathToData, "unet")

    generatedImprovedUNetSegmentations(pathToData)
    improvedUNetReconstruction(pathToData)



main()
