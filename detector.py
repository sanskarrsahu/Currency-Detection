import cv2
import math
import numpy as np
from config import *

# QUERY IMAGE = THE IMAGE FORM OUR DATASET(THAT WE'LL QUERY AGAINST)
# TRAIN IMAGE = THE CURRENT IMAGE

featureExtractor = cv2.ORB_create()

matcherBruteForce = cv2.BFMatcher_create(cv2.NORM_HAMMING)

def getFeatures(image):
    
    keypoints = featureExtractor.detect(image, None)

    kp, des = featureExtractor.compute(image, keypoints)
    return kp, des


def drawKeypointOnIamge(image, keyPoints):
    cv2.drawKeypoints(image, keyPoints, image, color=KEYPOINT_DRAW_COLOR, flags=0)


def filterFalsePositives(foundMatchings):
    if not foundMatchings:
        return []

    good = []
    try:
        for m, n in foundMatchings:
            if m.distance < 0.7 * n.distance:
                good.append(m)

    except ValueError:
        pass
    return good


def drawMatches(img1, kp1, img2, kp2, matches):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def getMatchingPoints(queryImageDes, trainImageDes):
    
    if queryImageDes is None or len(queryImageDes) == 0:
        return []

    if trainImageDes is None or len(trainImageDes) == 0:
        return []


    return filterFalsePositives(matcherBruteForce.knnMatch(queryImageDes, trainImageDes, k=2))


def buildHomographyInputData(inputImageKp, maxMatchingsData):
    if not (maxMatchingsData is None):
        queryImageData = maxMatchingsData[1]
        queryImageKeypoints = queryImageData[1][0]
        matches = maxMatchingsData[0]
        queryImageShape = queryImageData[0].shape
        return (queryImageShape, queryImageKeypoints, inputImageKp, matches)


def getPossibleCurrency(sampleData, inputImageGray):

    matchesSummary = {}

    maxMatchings = 0
    maxMatchingsData = None
    maxMatchingsCurrency = None


    kp, des = getFeatures(inputImageGray)

    for currencyValue, images in sampleData.items():
        totalMatches = 0
        for imageData in images:
            matches = getMatchingPoints(imageData[1][1], des)
            countMatches = len(matches)

            if countMatches > maxMatchings:
                maxMatchings = countMatches
                maxMatchingsCurrency = currencyValue
                maxMatchingsData = (matches, imageData)

            totalMatches += countMatches
            matchesSummary[currencyValue] = totalMatches

    return buildHomographyInputData(kp, maxMatchingsData), \
           matchesSummary, maxMatchings, maxMatchingsCurrency


def drawMatchingSummary(image, matchesSummary):
    origin = [50, 50]
    for k, v in matchesSummary.items():
        displayText = "{}: {}".format(k, v)
        image = cv2.putText(image, displayText, tuple(origin), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        origin[1] += 50

    return image


def highlightBill(frame, detectedCurrency, queryImageShape, kp1, kp2, matches):
    try:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if not (matrix is None) and matrix.size > 0:
            h, w = queryImageShape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)


            origin = dst[0][0]
            font_x = math.floor(origin[0] - CURRENCY_DISPLAY_OFFSET_X)
            font_y = math.floor(origin[1] - CURRENCY_DISPLAY_OFFSET_Y)
            font_position = (font_x, font_y)
            frame = cv2.putText(frame, detectedCurrency, font_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                CURRENCY_DISPLAY_COLOR, 0.5)


            return cv2.polylines(frame, [np.int32(dst)], True, CURRENCY_CONT_COLOR, CURRENCY_CONT_THINKNESS,
                                 cv2.LINE_AA)
        else:
            if DEBUG:
                print("Couldn't create matrix")
            return frame
    except Exception as e:
        if DEBUG:
            print("Exception in highlight function: {}".format(e))
        return frame


def processFrame(frame, cache):
    colorFrame = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    homographyData, matchingSummary, maxMatching, detectedCurrency = getPossibleCurrency(cache, frame)

    if not (detectedCurrency is None) and maxMatching > MIN_MATCH_COUNT:
        print("Detected currency is {} with {} matches".format(detectedCurrency, maxMatching))

        if not (homographyData is None):
            colorFrame = highlightBill(colorFrame, detectedCurrency, *homographyData)
        if DEBUG:
            colorFrame = drawMatchingSummary(colorFrame, matchingSummary)

    else:
        print("No currency detected")
    return colorFrame


def realTime(cache):
    videoCapture = cv2.VideoCapture(0)
    while 1:
        ret, frame = videoCapture.read()

        colorFrame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        homographyData, matchingSummary, maxMatching, detectedCurrency = getPossibleCurrency(cache, frame, colorFrame)

        if not (detectedCurrency is None) and maxMatching > MIN_MATCH_COUNT:
            print("Detected currency is {} with {} matches".format(detectedCurrency, maxMatching))

            if DEBUG:
                if not (homographyData is None):
                    colorFrame = highlightBill(colorFrame, detectedCurrency, *homographyData)
                colorFrame = drawMatchingSummary(colorFrame, matchingSummary)
        else:
            print("No currency detected")

        cv2.imshow("frame", colorFrame)
        key = cv2.waitKey(1)

        if key == CLOSE_KEY:
            break


def main():
    realTime(None)


if __name__ == "__main__":
    main()
