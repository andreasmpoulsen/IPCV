"""
Created on Fri June 11 08:59:38 2021

@author: Michael Helbo Nygaard
Credits to:
http://people.csail.mit.edu/mrub/papers/phasevid-siggraph13.pdf
https://hbfs.wordpress.com/2018/05/08/yuv-and-yiq-colorspaces-v/
https://github.com/Pantsworth/temporal-median-video
https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
https://docs.opencv.org/4.5.1/d4/d1f/tutorial_pyramids.html
https://theailearner.com/tag/laplacian-pyramid-opencv/
https://patents.google.com/patent/EP3242036B1/en?oq=EP3242036
https://arxiv.org/abs/1811.02060
"""

import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack
from matplotlib.pylab import *
from icecream import ic
from lucas_kanade import optical_flow
from collections import defaultdict
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import ccovf

def sparseLucasKanadeMethod(video, timePerFrame, fps, framesToSkip=30):
    # Read the video
    cap = cv2.VideoCapture(video)

    # Create random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first valid frame and find corners in it
    cap.set(cv2.CAP_PROP_POS_FRAMES, framesToSkip)
    succes, oldFrame = cap.read()
    oldFrameGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("snapshot.png", oldFrameGray)

    # This variable we use to store the pixel location
    referencePoint = []
    referencePoints = []

    # click event function
    def clickEvent(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            referencePoint.append([[x, y]])
            font = cv2.FONT_HERSHEY_SIMPLEX
            strXY = str(x) + ", " + str(y)
            cv2.putText(img, strXY, (x, y), font, 0.5, (255, 255, 0), 2)
            cv2.imshow("Sensor location", img)
            # Saving the image
            cv2.imwrite("visualSensorPoints.png", img)

        elif event == cv2.EVENT_RBUTTONDOWN:
            referencePoints.append([x, y])
            font = cv2.FONT_HERSHEY_SIMPLEX
            strXY = str(x) + ", " + str(y)
            cv2.putText(img, strXY, (x, y), font, 0.5, (255, 255, 0), 2)

        elif event == cv2.EVENT_RBUTTONUP:
            referencePoints.append([x, y])
            ic(referencePoints)
            font = cv2.FONT_HERSHEY_SIMPLEX
            strXY = str(x) + ", " + str(y)
            cv2.putText(img, strXY, (x, y), font, 0.5, (255, 255, 0), 2)
            cv2.rectangle(img, (referencePoints[0][0], referencePoints[0][1]),
                          (referencePoints[1][0], referencePoints[1][1]), (255, 0, 0), 2)
            cv2.imshow("Sensor location", img)
            # Saving the image
            cv2.imwrite("visualSensorArea.png", img)

    # Select measure point
    while True:
        # Here, you need to change the image name and it's path according to your directory
        img = oldFrameGray
        cv2.imshow("Sensor location", img)

        # calling the mouse click event
        cv2.setMouseCallback("Sensor location", clickEvent)

        k = cv2.waitKey(timePerFrame) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break

    if referencePoint:
        p0 = np.array(referencePoint).astype(np.float32)
    elif referencePoints:
        minimumSensorDistance1 = int(np.sqrt(referencePoints[1][0] - referencePoints[0][0])) * 3
        minimumSensorDistance2 = int(np.sqrt(referencePoints[1][1] - referencePoints[0][1])) * 3

        referencePoints = [[[x, y]] for x in
                           list(range(referencePoints[0][0], referencePoints[1][0], minimumSensorDistance1)) for y in
                           list(range(referencePoints[0][1], referencePoints[1][1], minimumSensorDistance2))]
        p0 = np.array(referencePoints).astype(np.float32)
    # Show used sensor coordinates
    sensorCoordinates = p0
    ic(sensorCoordinates)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(oldFrame)
    sensorDictionary = defaultdict(list)

    # Run trough video frames
    while True:
        # Read new frame
        ret, frame = cap.read()
        if not ret:
            break
        newFrameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow with LUCAS-KANADE algorithm
        p1 = optical_flow(oldFrameGray, newFrameGray, p0, 10)

        # Select good points
        newPoints = sensorCoordinates + p1
        oldPoints = p0

        # Draw the tracks
        for i, (new, old) in enumerate(zip(newPoints, oldPoints)):
            a, b = new.ravel()
            sensorDictionary[i].append(np.sqrt(np.power(a, 2) + np.power(b, 2)))

            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 3)
            frame = cv2.circle(frame, (int(a), int(b)), 7, color[i].tolist(), -1)

        # Display optical flow on image
        img = cv2.add(frame, mask)
        cv2.imshow("Sensor location", img)
        k = cv2.waitKey(timePerFrame) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break

        # Update the previous frame and previous points
        oldFrameGray = newFrameGray.copy()
        p0 = newPoints.reshape(-1, 1, 2)

    # Post analysis of optical flow
    butterRollIndex = 0
    averagedSensorData = np.array((np.average(np.array(list(sensorDictionary.values())), axis=0)))
    filteredData = averagedSensorData  # butterworthBandpassFilter(averagedSensorData, temporalLowCut, temporalHighCut, fs=fps, order=3)
    sensorDictionary[i + 1] = filteredData[butterRollIndex:]
    sensorDictionary[i + 1] = list(sensorDictionary[i + 1] - sensorDictionary[i + 1].mean(axis=0))

    if referencePoint:
        dummyReferencePoint = ["Average"]
        referencePoint.append(dummyReferencePoint)
    elif referencePoints:
        referencePoint = referencePoints
        dummyReferencePoints = ["Average"]
        referencePoint.append(dummyReferencePoints)

    frequencyDictionary = defaultdict(list)

    for key in sensorDictionary.keys():
        # Display vector change as sensor time series
        N = len(sensorDictionary[key])
        T = 1.0 / fps
        window = signal.blackmanharris(len(sensorDictionary[key]))
        yf = scipy.fftpack.fft(sensorDictionary[key] * window)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

        # chi squared alignment at native resolution
        if key < len(sensorDictionary) - 1:
            s = phaseAlign(sensorDictionary[key], sensorDictionary[key + 1], [5, 20])
            ic('phase alignment', s)

        dcStart = 5
        xf = xf[dcStart:]

        fig, axs = plt.subplots(2)
        fig.set_size_inches(12, 8)
        fig.suptitle('Measure point and FFT at ' + str(referencePoint[key]))
        axs[0].plot(range(len(sensorDictionary[key])), sensorDictionary[key])
        axs[0].set_xlabel('Samples (-)')

        halfSidedAbsoluteSpectrum = 2.0 / N * np.abs(yf[dcStart:N // 2])
        indexes = signal.argrelextrema(np.array(halfSidedAbsoluteSpectrum), comparator=np.greater, order=2)

        axs[1].plot(xf, halfSidedAbsoluteSpectrum)
        axs[1].plot(xf[indexes[0]], halfSidedAbsoluteSpectrum[indexes[0]], "x")
        for indexX, indexY in zip(xf[indexes[0]], halfSidedAbsoluteSpectrum[indexes[0]]):
            if indexY > np.mean(halfSidedAbsoluteSpectrum[indexes[0]] * 2):
                axs[1].text(indexX, indexY, '({}, {})'.format(indexX, indexY))
        frequencyDictionary[key].append([[xf[dcStart:]], [halfSidedAbsoluteSpectrum[dcStart:]]])
        axs[1].set_xlabel('Frequency (Hz)')
        fig.savefig('analysisOutput_' + str(key) + '.png', dpi=100)
        plt.show()

    cv2.destroyAllWindows()
    return sensorDictionary, frequencyDictionary, sensorCoordinates

def phaseAlign(reference, target, roi, res=100):
    '''
    Cross-correlate data within region of interest at a precision of 1./res
    if data is cross-correlated at native resolution (i.e. res=1) this function
    can only achieve integer precision 

    Args:
        reference (1d array/list): signal that won't be shifted
        target (1d array/list): signal to be shifted to reference
        roi (tuple): region of interest to compute chi-squared
        res (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    # Convert to int to avoid indexing issues
    ROI = slice(int(roi[0]), int(roi[1]), 1)

    # Interpolate data to a higher resolution grid 
    x,r1 = highres(reference[ROI],kind='linear',res=res)
    x,r2 = highres(target[ROI],kind='linear',res=res)

    # Subtract mean
    r1 -= r1.mean()
    r2 -= r2.mean()

    # Compute cross covariance 
    cc = ccovf(r1,r2,demean=False,adjusted=False)

    # Determine if shift is  positive or negative 
    if np.argmax(cc) == 0:
        cc = ccovf(r2,r1,demean=False,adjusted=False)
        mod = -1
    else:
        mod = 1

    # Often found this method to be more accurate then the way below
    return np.argmax(cc)*mod*(1./res)

def highres(y,kind='cubic',res=100):
    '''
    Interpolate data onto a higher resolution grid by a factor of *res*

    Args:
        y (1d array/list): signal to be interpolated
        kind (str): order of interpolation (see docs for scipy.interpolate.interp1d)
        res (int): factor to increase resolution of data via linear interpolation

    Returns:
        shift (float): offset between target and reference signal 
    '''
    y = np.array(y)
    x = np.arange(0, y.shape[0])
    f = interp1d(x, y,kind='cubic')
    xnew = np.linspace(0, x.shape[0]-1, x.shape[0]*res)
    ynew = f(xnew)
    return xnew,ynew