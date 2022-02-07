# Exercise - The Basics - Pixels

## Exercise 1

1. What is the bit sequence of the pixel value 150?
2. In a 100x100 grayscale image, each pixel is re represented by 256 gray levels. How much memory (bytes) is required to store the image?
3. In a 100 x 100 gray-scale image each pixel is represented by 4 gray levels. How much memory (bytes) is required to store the image?
4. We want to photograph an object which is 1m tall and 10m away from the camera. The height of the object in the image should be 1mm. What should the focal length (f) be? (we assume that the object is in focus at the focal point, hence f=b)

## Exercise 2

Mick is 2m tall and standing 5m from a camera. The camera’s focal length is 5mm. The camera image has a size of 640x480 pixels.
The camera has a 1/2” CCD chip:

![CCD chip image](extra\billede.png)

1. A focused image of Mick is formed inside the camera. At which distance from the lens?
2. How tall (in mm) will Mick be on the CCD-chip?
3. How tall (in pixels) will Mick be on the CCD-chip?
4. What is the field-of-view of the camera?

## Exercise 3

1. Install OpenCV
2. Make a program which loads and displays an image using OpenCV
3. Download a picture. Make a program which loops over each pixel and prints its value, row by row.

## Exercise 4

1. Design an algorithm (on paper), which can do histogram stretching.
2. Use [ImageJ](https://imagej.nih.gov/ij/download.html) to play around with gray level mapping, histograms, and thresholding.
3. Use ImageJ to improve the quality of the image [enhance_me.jpg](exercise_materials\enhance-me.jpg)
4. Use ImageJ to improve the image [dots.jpg](exercise_materials\dots.jpg) so that only 28 non-connected dots remain.
5. Use ImageJ to improve the image paper.jpg so that the text is black and the rest is white. Consider subtracting the mean of the image before thresholding.
6. How will the Threshold-algorithm output look if two threshold values are used instead of just one?
7. Is it wise to have a binary image consisting of 0s and 1s?

## Exercise 5

1. Use OpenCV to load and display a color image
2. Convert it to grayscale using nested loops
3. Convert it to grayscale using matrix multiplication
4. Convert it to grayscale using the built-in OpenCV function