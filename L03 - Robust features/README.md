# Exercise

## Exercise 1

1. Load two partly overlapping images into OpenCV ([aau-city-1](exercise_material\aau-city-1) and [aau-city-2](exercise_material\aau-city-2) from Moodle, or take your own pictures)
2. Extract Harris Corners from both images. You may use the OpenCV function cornerHarris()
3. Design and implement your own simple corner matching procedure to find the same points in both images
4. Calculate the transformation that aligns the images

## Exercise 2

1. Change your solution from last exercise to extract ORB keypoints and descriptors instead of Harris Corners and your own descriptors
2. Match feature descriptors using the DescriptorMatcher class
3. Visualize the matches using the drawMatches function