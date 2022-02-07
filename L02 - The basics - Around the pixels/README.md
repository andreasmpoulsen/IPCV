# Exercise

## Exercise 1

1. With [lion.jpg](exercise_materials\lion.jpg), play around with the different filters in ImageJ. Discuss/show the effects of different kernel sizes.
2. Correlate the image and kernel below. Remember to normalize and round.

![image and kernel](extra\image.png)
 
3. When doing neighborhood processing the output image is smaller than the input image​
    * Why?​
    * Does it matter?​
    * What can we do about it?

## Exercise 2

1. Use Python and OpenCV to load and display [lion.jpg​](exercise_materials\lion.jpg)
    * Implement a mean filter with configurable kernel size​
    * Implement a Gaussian blur filter with configurable kernel size​
2. Use Python and OpenCV to load and display [neon-text.png](exercise_materials\neon-text.png)​. Use template matching to make an image which shows the positions of the three hearts, similar to the one below (the white dots showing the positions are very small, but there are three of them). Show the correlation image as an intermediate step.

![white dots image](extra\white-dots.png)

## Exercise 3

1. Play around with the different morphologic operations in ImageJ : Process -> Binary
2. How can Morphology be used to find the outline (edge) of an object in a binary image? 
3. Use morphology to improve the results from the previous lecture on the image [dots.jpg](..\L01%20-%20The%20Basics%20-%20Pixels\exercise_materials\dots.jpg)
4. Find g(x,y), where:

![equation](extra\equation.png)

given that

![f(x)](extra\f(x).png)

and 

![SE](extra\SE.png)

## Exercise 4
Use depth-first Connected Component Analysis on this image:

![example image](extra\example1.png)

In what order will the pixels be marked if a ​

1. 4-connected kernel is used?​
2. 8-connected kernel is used?​

## Exercise 5
Calculate the perimeter of the object below using the following distance measures (you can look them up online):

![example image](extra\example2.png)

1. Euclidean distance
2. City-block (Manhattan) distance
3. Chessboard (Chebyshev) distance​

## Exercise 6

Imagine you are using the size of a BLOB as a feature in your project. In feature matching, you need to compare the model of the BLOB with the BLOB extracted from the image​.
Find a way to normalize the feature matching, so the result always is in the interval: [0,1] (where 0 means a very poor match and 1 means a perfect match)​

## Exercise 7

1. Make a Python program which performs a connected component analysis on  [shapes.png](exercise_materials\shapes.png).​ Implement the connected component analysis with your own functions.
2. Compute and print the compactness for each shape.
3. Try to use OpenCV’s connectedComponents-function​