from icecream import ic
import numpy as np
import cv2
import SupportFunctions
import scipy.signal as signal

if __name__ == '__main__':
    desiredFPS = 4435
    desiredResample = 1
    amplification = 50

    fps = int(desiredFPS/desiredResample)  # 1069
    ic(fps)
    timePerFrame = 10  # ms, determines play speed
    videoMagnifiedName = 'steerable_lowFreq_550_highFreq_600_pyramidLevels_4_amplification_50_from_580Hz1_5G.mp4_amplifiedMotionOut.avi'  # "inputBaseline.mp4"#


    ### LUCAS-KANADE
    sensorDictionary, frequencyDictionary, sensorCoordinates = SupportFunctions.sparseLucasKanadeMethod(
            videoMagnifiedName, timePerFrame, fps)

    # Decision assistance (Generic - not fully adapted)
    rmsList = []
    ts = 1/fps
    rmsCalibration = 1
    velocityCalibration = 1/amplification
    for sensorIndex in range(len(sensorCoordinates)):
        ic(sensorCoordinates[sensorIndex])
        filteredAverageDisplacementSeries = np.array(
            sensorDictionary[sensorIndex])-np.mean(sensorDictionary[sensorIndex])
        filteredAverageVelocitySeries = np.array(
            [sum(filteredAverageDisplacementSeries[:i])/ts * velocityCalibration for i in range(len(filteredAverageDisplacementSeries))])
        ic(max(filteredAverageVelocitySeries) * velocityCalibration)

        # Displacement RMS
        RMS = np.sqrt(
            np.mean(filteredAverageDisplacementSeries**2))*rmsCalibration
        rmsList.append(RMS)
        ic(RMS)

        # Spectrum quantification
        spectrumAverage = frequencyDictionary[sensorIndex][0]
        peakFrequency = spectrumAverage[0][0][np.argmax(spectrumAverage[1])]
        ic(peakFrequency)

        # Phase detection
        nSamples = len(sensorDictionary[0])
        crossCorrelation = signal.correlate(
            sensorDictionary[sensorIndex], sensorDictionary[sensorIndex+1])
        dt = np.arange(1 - nSamples, nSamples)
        recoveredTimeShift = dt[crossCorrelation.argmax()]
        ic("Estimated time (phase) shift: %d" % (recoveredTimeShift))

    # Average result
    sensorIndex = sensorIndex + 1
    filteredAverageDisplacementSeries = np.array(
        sensorDictionary[sensorIndex]) - np.mean(sensorDictionary[sensorIndex])
    filteredAverageVelocitySeries = np.array(
        [sum(filteredAverageDisplacementSeries[:i])/ts * velocityCalibration for i in range(len(filteredAverageDisplacementSeries))])
    ic(max(filteredAverageVelocitySeries) * velocityCalibration)

    # Displacement RMS and statistical spread
    RMS = np.sqrt(np.mean(filteredAverageDisplacementSeries ** 2))
    RSD = np.std(rmsList)/np.mean(rmsList)*100
    # The standard deviation is the square root of the average of the squared deviations from the mean
    ic(RMS, str(RSD)+' %')

    # Spectrum quantification
    spectrumAverage = frequencyDictionary[sensorIndex][0]
    peakFrequency = spectrumAverage[0][0][np.argmax(spectrumAverage[1])]
    ic(peakFrequency)

    # Save for external post processing
    color = np.random.randint(0, 255, (100, 3))
    img = cv2.imread("snapshot.png")
    mask = np.zeros_like(img)
    for i, (coordinate, rms) in enumerate(zip(sensorCoordinates, rmsList)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = coordinate.ravel()
        rmsString = str(rms.ravel())
        cv2.putText(img, rmsString, (int(x), int(y)),
                    font, 0.5, (255, 255, 0), 2)
    cv2.imwrite("visualSensorValues.png", img)