# **Behavioral Cloning** 


---

[//]: # (Image References)

[centerdriving]: ./images/centerdriving.jpg "Center driving"
[fake_image1]: ./images/fake_image1.jpg "Fake data"
[left_recovery]: ./images/leftrecovery.jpg "Left recover"
[alongcurve]: ./images/alongcurve.jpg "Along curve"
[reverse]: ./images/reverse.jpg "Reverse"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py`, `helper.py` and `image_helpers.py` together containing the script to create and train the model.
For the sake of clarity of code, I split my processing methods into a `helper.py` and `image_helpers.py`. 
So `model.py` contains just model constructions and training which makes it easy to review.
* `drive.py` for driving the car in autonomous mode. I had to change the speed of driving (to 20mph) to match something closer to my driving pattern ( i could not drive really slow since my computer could not handle it well). 
However model does perform fairly well for range of speeds. 
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` which contains write up
* `video.mp4` which contains generated video from autonomous run.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run
```

#### 3. Submission code is usable and readable

I split my processing methods into a `helper.py` and `image_helpers.py`. 
So `model.py` contains just model constructions and training which makes it easy to review.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I first tried with a simple 2 layer convnet followed by three full connected layers (this was inspired by my solution for traffic classifier project). I tried tuning dropoffs, learning rate etc.
But I could not get great results. Basically my model was not learning recovery modes well.  
 
For model, I tried tweaking a bit to by taking clues from vgg net, googlenet architecture. But in all of them either my training or validation loss was not good. 

I then used the approach suggested in project description. [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

Model can be found in `model.py` from line #68-#112. I will describe the architecture and modifications in further sections below.

#### 2. Attempts to reduce overfitting in the model

Even with this model, I noticed lot of times my model was overfitting to center driving and not doing great with recovery. To counter this,
I added some dropout which improved validation and test loss. See line #99 in `model.py`.


#### 3. Model parameter tuning

I used adam optimizer. I tuned both batch size, learning rate and epoochs.

Batch size 128 worked best for my model.

I tried 0.001 and 0.0001 for Learning rates. Finally 0.0001 got best result.

For epochs, I noticed that after 3 epochs my validation loss was fluctuating while training loss was still decreasing. This indicated overfitting. So I chose `3 epochs`.


You can see parameters in line #114-#115

#### 4. Appropriate training data

I first generated the training data like it was suggested in project videos. 3 laps of center driving, 1 laps along curve, 1 laps for recover and 2 laps for reverse direction (clockwise).

This was not sufficient for recovery modes because of the way `drive.py` implements driving vs how I was driving manually in training mode. More details are explained in training process section below.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As explained previously, I used [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) for final approach.
Reading the paper, it looked most appropriate for my use case. This model detects curves, expecially road or lane marking detection.

Model But I could not use it out of box. Before feed the data into this model, I added couple of layers to my keras model. First one is,
`Cropping2D` to crop out other parts of the image (other than lanes), `resize` to convert the image to `66,200,3` as expected by Nvidia model.

Also I applied RELU activations for non linearity and dropout for avoiding overfitting (I found dropout to be to super important for recovery mode detection).
Without dropout, my model overfit to straight line driving because of the nature of the track.

I was able to test this by running `driving.py` with a slightly different speed than what I was driving in training model. As kept adding RELU and dropout, my model was getting better at recovery. Also there was some training generation techniques I need to do for this (which are explained in further sections).
 
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road or cutting the yellow line.

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-112) is as follows:

* `Cropping2D` to crop out top 50 and bottom 20 pixels.
* `resize` layer to convert image to `66, 200, 3`
* `Normalization` layer for faster training
* `Convolution2D` with 5X5 kernel, stride = 2, output channels = 24. Applied RELU activation
* `Convolution2D` with 5X5 kernel, stride = 2, output channels = 36. Applied RELU activation
* `Convolution2D` with 5X5 kernel, stride = 2, output channels = 48. Applied RELU activation
* `Convolution2D` with 3X3 kernel, no strides, output channels = 64. Applied RELU activation
* `Convolution2D` with 3X3 kernel, no strides, output channels = 64. Applied RELU activation
* `Flatten`
* `Dropout` with a keep_prob of 60%.
* `Fully Connected layer` with 100 output units
* `Fully Connected layer` with 50 output units
* `Fully Connected layer` with 10 output units
* `Fully Connected layer` with 1 output unit corresponding to steering angle.

I compiled with model with `mse` loss and `adam` optimizer. Also used `fit_generator` as suggested to avoid memory issues.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Driving][centerdriving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover incase it detects vehicle is escaping off the road. Here is a image where vehicle is about to detect escaping to water on left side :

![Left recover][left_recovery]


Then I collect some images for drving along the curves on steep turns. 
I need this because, `drive.py` seems to apply a constant speed. But when I train on my computer, my natural game playing instincts made me slow the car. I tried replicating drive.py in my driving but found that to be laborious. 
Instead I felt its easier to make car learn to driving along curve.

![alongcurve][alongcurve]

I also drove in reverse direction to simulate flipping of images. 

![reverse][reverse]

Lot of this process was watching car behave in autonomous mode and making car learn situations that arise from it and making sure to generalize this.

After collection, I still did some preprocessig of data in my program. Some ot the pre processing I did are.

1. Observe the distribution of data and generate fake data for angles which are under represented in distribution. This was needed becuse, straight driving is more than curves in track. To make sure model learns enough from curves and recovery, I had to do that.
2. For fake data, perform some image transformation techniques. Like adding blur, translating image in horizontal direction. I did no translate in vertical direction because later my model would chop image and this causes problems. 

Here is a sample fake image.

![Fake Image][fake_image1]

I then randomly shuffled data and split it into training, validation and test set.

Training size: 17483
Validation set: 3294 (~20% of training)
Test set: 6119 (~35% of training)

As explained earlier, these were my hyper parameters:

* Batch size `128` worked best for my model.
* For learning reate, I tried `0.001` and `0.0001`. Finally `0.0001` got best result.
* For epochs, I noticed that after 3 epochs my validation loss was fluctuating while training loss was still decreasing. This indicated overfitting. So I chose `3 epochs`.


#### 4. Results

I ran the car twice around the track in autonomous mode and captured video. 

As you can see in `video.mp4`, car stayed in lane at all times and did not cut the yellow line. 

A long step curves, it followed my driving pattern, by driving along the shape of the curve and not cutting it. During recovery it made sure to not overshoot recovery by adjusting steering angle immediately. 

#### 5. Feedback

All in all, this was fun project. But there was significant challenge in making sure car behaves well as per code in autonomous mode. As a result, I spent lot more time on data generation and pre processing than model generation. 

One thing I noticed was we are just producing steering angle in model. But most of the times driving pattern that I have (in terms of speed at various positions in track) is not same as `drive.py`.

For example, when you drive on a curve at 15mph your angle has to be different than when you drive at 20mph. When I drive in simulator mode, I tend to slow down signficantly along curve. But `drive.py` does not do the same.
For now, I got around this by creating extensive training data, pre processing, fake data generation and model architecture to detect fall off early to create smooth recovery. 

It could be more efficient to train the model to produce `speed` and `angle` and have the `drive.py` just apply them at every position of the road.

May be this could a future experiment I will do. But for now, I have got very good results and spent lot of hours on this already.
