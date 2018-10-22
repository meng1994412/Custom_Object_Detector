# Custom Object Detector
## Project Objectives
* Built a custom object detector based on caltech101 dataset.
* Define a JSON file to store all framework configurations.
* Trained a HOG with SVM model to detect the specific object.
* Implemented non-maxima suppression to find the best location of object.
* Applied hard-negative mining techniques to increase the accuracy of object detector.

## Software/Packages Used
* Python 3.5
* [OpenCV](https://docs.opencv.org/3.4.1/) 3.4
* [Scikit-Learn](http://scikit-learn.org/stable/)
* [Scikit-Image](http://scikit-image.org/docs/0.13.x/)
* [JSON-minify](https://github.com/getify/JSON.minify/tree/python)
* [Imutils](https://github.com/jrosebr1/imutils)
* [HDF5](https://www.h5py.org/)

## Algorithms & Methods Used
* Experiment preparation
  * Define a `.json` file to store framework configurations.
  * Compute the average object dimensions.
* Feature extraction
  * Use the average object dimensions to choose appropriate Histogram of Oriented Gradient descriptor dimensions along with sliding window size.
    * Histogram of Oriented Gradient (HOG) descriptor
  * Implement data augmentation to increase dataset size.
  * Extract HOG features from positive and negative image example training images from datasets.
* Detector training
  * Utilize extracted HOG feature vectors and associated class labels to train a Linear SVM for object detection.
    * Support vector machine
* Non-maxima suppression
  * Implement non-maxima suppression to reduce overlapping bounding boxes.
* Hard negative mining
  * Implement hard-negative mining in object detection framework to reduce false positive cases.
* Detector retrained
  * Use hard-negative mining examples mined from previous step to re-train our object detector.


## Approaches
* The dataset is obatined from [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) and [13 Natural Scene Categories](http://vision.stanford.edu/resources_links.html)
* In this project, car side category inside caltech101 dataset is used for building the object detector. Generally speaking, any categories can be used to build the specific object detector. All we need to do is define a `.json` file to store framework configurations, which looks similar to `car.json`[file](https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/configuration/cars.json).

## Results
### Experiment preparation
Generally speaking, the object detector can be used for detecting any objects (at least, any object categories inside caltech101 dataset), though the car side category is used for demo in this project. For different objects, all we need to do is defining a new (or modifying) the `.json` file, e.g., in this project, `cars.json`[file](https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/configuration/cars.json).

After running `explore_dims.py`[file](https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/explore_dims.py), average object dimensions are computed. The results are shown in Figure 1.

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/explore_dimension.png" width="200">

Figure 1: The average dimensions for car side category.

__The average object dimensions is helpful for choosing appropriate HOG descriptor dimensions along with sliding window size.__

### Feature extraction
After running `extract_features.py`[file](https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/extract_features.py), extracted HOG feature vectors are all saved into `.hdf5` file which looks like Figure 2.

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/car_features.png" width="400">

Figure 2: car_features.hdf5 for storing extracted features from car side category.

### Detector training
After running `train_model.py`[file](https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/train_model.py) and `test_model_no_nms.py`[file](https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/test_model_no_nms.py), which implement HOG + SVM to build the detector, initial detector model is finished.

Figure 3 & Figure 4 shows two sample test results (without non max suppression).

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/test_results_no_nms_1.png" width="400">

Figure 3: Test result for sample # 1 (without non-maxima suppression).

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/test_results_no_nms_2.png" width="400">

Figure 4: Test results for sample # 2 (without non-maxima suppression).

As the results shown, there are multiple bounding boxes on the object. The best bounding box will be chosen in the next part.

Though there are some false positive cases, such as Figure 5 and Figure 6 shown, this problem will be solved when applying hard negative mining.

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/test_results_no_nms_false_positive_1.png" width="400">

Figure 5: False positive case of test results for sample # 3 (without non-maxima suppression).

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/test_results_no_nms_false_positive_2.png" width="400">

Figure 6: False positive case of test results for sample # 4 (without non-maxima suppression).

### Non-maxima suppression
After applying `test_model.py`[file](https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/test_model.py), non-maxima suppression is implemented to find the best bounding box for the object detected.

Figure 7 and Figure 8 demonstrate the test results before and after non-maxima suppression applied.

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/test_results_1.png" width="800">

Figure 7: Test result before and after non-maxima suppression for sample # 1.

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/test_results_1.png" width="800">

Figure 8: Test result before and after non-maxima suppression for sample # 2.

False positive cases are still existing, as Figure 9 & Figure 10 shown, they will be eliminated in next part.

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/test_results_false_positive_1.png" width="800">

Figure 9: False positive case of test result for sample # 3.

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/test_results_false_positive_2.png" width="800">

Figure 10: False positive case of test result for sample # 4.

### Hard negative mining
After applying `hard_negative_mine.py`, hard-negative samples are added to `.hdf5` file to improve the accuracy of object detector, as Figure 11 shown. And `.hdf5` file is a little larger than before, as Figure 12 shown.

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/hard_negative_mining.png" width="800">

Figure 11: Training data with hard-negative samples.

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/car_features_hard_negative.png" width="400">

Figure 12: car_features.hdf5 for storing extracted features with hard-negative samples from car side category.

### Detector retrained
After retrain the model by using `train_model.py`, the false positive cases are eliminated, as Figure 13 and Figure 14 shown, comparing to the results in Figure 9 and Figure 10.

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/test_results_hard_1.png" width="800">

Figure 13: Test result with hard-negative mining for sample # 3.

<img src="https://github.com/meng1994412/Custom_Object_Detector/blob/master/custom_object_detector/output/milestone_demo/test_results_hard_2.png" width="800">

Figure 14: Test result with hard-negative mining for sample # 4.
