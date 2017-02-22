# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_nocar]: ./files/cat_nocar.png
[car_nocar_hog]: ./files/car_nocar_hog.png
[nocar_hog]: ./files/nocar_hog.png
[slide_win1]: ./files/slide_win1.png
[slide_win2]: ./files/slide_win2.png
[heatmap]: ./files/heatmap.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it(README)!  
I used [this jupyter notebook](https://github.com/HidetoKimura/carnd_vehicle_detection/blob/master/project_main.ipynb) in this project. Because in this project to do data analysis and visualization is very important.
Here is the file structures.

### File Structures

project_main.ipynb - The jupyter notebook of the project.    
README.md - This file containing details of the project.  
/output_images/processed_project_video.mp4 - The result video.  
/files/ - Folder for README.md.  
/vehicles/ - Folder for the vehicle training data. NOTE: There is not on Github. This project uses the udacity data. You can download [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip).  
/non-vehicles/ - - Folder for the non-vehicle training data. NOTE: There is not on Github. This project uses the udacity data. You can download [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).  

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2nd/4th code cell of the IPython notebook located in "./project_main.ipynb "  

I used the training data given by Udacity, which come from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself. All images are 64x64 pixels. A third data set released by Udacity was not used here. There are 8792 images of vehicles and 8968 images of non-vehicles.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_nocar]

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][car_nocar_hog]
![alt text][nocar_hog]

####2. Explain how you settled on your final choice of HOG parameters.

First, I used the RGB color space. But it could not detect vehicles at all.   
Next, I used HLS color space and L(Lighting) channel. But it was not good, too.
The YUV model defines a color space in terms of one luma (Y) and two chrominance (UV) components.  
The Y channel corresponds to the gray scale and it seems like the best for detecting shape.  
Finally, I got the good result using YUV and Y(luma) channel. 

Below was why I chose the HOG parameters - `YUV, Y channel, orient=9, pix_per_cell=8, cells_per_block=2`.
Using smaller values of than `orient=9` caused more false positives. Using values larger than `orient=9, pix_per_cell=8, cells_per_block=2` increased the feature vector and did not improve results. And using `hog_cannel=ALL` caused out of memory. 

|  colorspace | orient | pix_per_cell | cell_per_block | hog_channel  | result|
|:--------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| RGB | 9 | 8 | 2 | 0/1/2 | Bad |
| HLS | 9 | 8 | 2 | 1 | Bad |
| YUV | 9 | 8 | 2 | 0 | Pretty good |
| YUV | 8 | 8 | 2 | 0 | Good(but caused false positives) |
| YUV | 7 | 8 | 2 | 0 | Good(but caused false positives) |
| YUV | 9 | 8 | 2 | ALL | Out of Memory |

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the 6th code cell of the IPython notebook located in "./project_main.ipynb "     

1. I get the feature vector of `car `and `not_car` using `extract_features()`.
2. Then, I concatenate the feature vector of `car `and `not car`.
3. Using sklearn.preprocessing.StandardScaler(), I normalize the feature vector.
4. Next, I concatenate the label 1 of of the car training set and the label 0 of the notcars trainig set.
5. I split up the data into randomized 80% training and 20% test sets using `sklearn.model_selection.train_test_split()`. This automatically shuffles the dataset.
6. Using `sklearn.svm.LinearSVC()`, I fit the training features and labels to the model.
7. Print out the accuracy.

~~~~
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
svc.fit(X_train, y_train)
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
~~~~

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

