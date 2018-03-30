
## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_examples/car_not_car.png "Car ot Not Car"
[image2]: ./writeup_examples/sliding_window.png "Sliding Window"
[image3]: ./writeup_examples/cars1.png "Detected Cars"
[image4]: ./writeup_examples/cars2.png "Detected Cars"
[image5]: ./writeup_examples/cars3.png "Detected Cars"
[image6]: ./writeup_examples/before_add_heat.png "Before Heatmap"
[image7]: ./writeup_examples/heatmap.png "Heatmap"
[image8]: ./writeup_examples/threshold_heatmap.png "Threshold Heatmap"
[image9]: ./writeup_examples/labels.png "Labels"
[image10]: ./writeup_examples/final_image.png "Final Image"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `get_hog_features` in the fourth code cell of the IPython notebook `vehicle_detection_and_tracking.ipynb`.  
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I tried multiple color spaces- like, YUV, LUV , YCrCb and fnally settled with LUV.


#### 2. Explain how you settled on your final choice of HOG parameters.

I settled on my final choice of HOG parameters based upon the performance of the SVM classifier produced using them. I found the following combination gave a better test accuracy for my classifier:

`orient: 11` <br>
`pixels_per_cell: 16, testing shows 16 better than 8` <br>
`cells_per_block: 2` <br>


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

 I trained a linear SVM with the default classifier parameters. I used HOG features, histogram of colors and spatial binning features to train my model. The classifier is implemented in 7th code cell in the IPython notebook. I have achieved 98.34% accuracy on test set.
 
`Using: 11 orientations 16 pixels per cell and 2 cells per block` <br>
`Feature vector length: 2052` <br>
`5.34 Seconds to train SVC...` <br>
`Test Accuracy of SVC =  0.9834` <br>
 
 For positive prediction I have used SVM `decision_function` and if the value returned is more than 2 then only I took that as positive. This is to reduce chance of False positive detection.
 
 ### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started with the `slide_window` function in code cell five. This function generates a list of windows to search. I extracted features for each windows and had the model to make the prediction. Object in different positions will have different size because the further the distance, the smaller the object. The main benefit is that this function fits nicely with previous code. The main drawback was the slow speed because it has to calculate the HOG feature for each window.

Then I adapted the sub-sampling techniques. Windows are generated based on cells and blocks in HOG features. For each image, the technique calculates HOG features only once.

Since windows are generated based on cells and blocks, windows will be moved to next position using metrics based on cells or blocks. The sub-sampling technique doesn't enlarge the window size, instead it scales the image before feature extraction and keeps window size the same, which is equivalent to zooming the window.

I explored several configurations of window sizes and positions, with various overlaps in the X and Y directions. Finally I have used the following two sizes:

`   ystart = 400` <br>
`    ystop = 520` <br>
`    scale = 1 (64 pixels)` <br>

`    ystart = 400` <br>
`    ystop = 600` <br>
`    scale = 2 (128 pixels that get scaled to 64 pixels)` <br>


![alt text][image2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
![alt text][image4]
![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/nandi-abhishek/Udacity-CarND-Vehicle-Detection-and-Tracking/blob/master/project_video_out.mp4)



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

A true positive is typically accompanied by several positive detections, while false positives are typically accompanied by only one or two detections. I used a combined heatmap and threshold to differentiate the two. The `add_heat` function increments the pixel value (referred to as "heat") inside a detected rectangle. Areas encompassed by more overlapping rectangles are assigned higher levels of heat. The following images shows a heatmap generated from the detected windows:

![alt text][image6]

![alt text][image7]

Here is the heatmap after applying a threshold value(in this case the threshold is 1). All pixels that don't exceed the threshold are set to zero. The result is below:

![alt text][image8]

The `scipy.ndimage.measurements.label()` function collects spatially contiguous areas of the heatmap and assigns each a label:

![alt text][image9]

And the final detection area is set to the extremities of each identified label:

![alt text][image10]

I have used a class to store the detected windows for the last few frames. the detections for the past 15 frames are combined and added to the heatmap and the threshold for the heatmap is set to `1 + len(det.prev_rects) * 3 // 4` - - this value was found to perform best empirically.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I faced is with the SVM classifier. It was producing too many false positive. Although integrating detections from previous frames mitigates the effect of the misclassifications, but it also introduces another problem: sometime a real vehicle may miss detection for few frames due to the threshold applied.

The pipeline is probably most likely to fail in cases where vehicles (or the HOG features thereof) don't resemble those in the training dataset, but lighting and environmental conditions might also play a role (e.g. a white car against a white background). 

I noticed that current reconstruction of boxes will combine two cars nearby. Current algorithm will definitely fail in case of heavy traffic.

I think a better classifier needs to be used to reduce so mant false positive and too much dependency on the threshold. I plan to revisit the project with a Fast R-CNN model.
