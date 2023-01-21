# Vision Description for blind aid using LSTM : Teaching Computers to describe pictures

## Problem Statement

Receive an image as input and produces a relevant caption or short desription for that.

## Data collection

There are many open source datasets available for this problem, like Flickr 8k (containing8k images), Flickr 30k (containing 30k images), MS COCO (containing 180k images), etc. But for the purpose of this case study, I have used the Flickr 8k dataset from Kaggle. This dataset contains 8000 images each with 5 captions (as we have already seen in the Introduction section that an image can have multiple captions, all being relevant simultaneously).

These images are bifurcated as follows:

Training Set — 6000 images

Dev Set — 1000 images

Test Set — 1000 images

## Understanding the data

In the downloaded data, along with images there will be a text file contains the name of each image along with its 5 captions. Thus every line contains the \<image name\>\#i \<caption\>, where 0≤i≤4. i.e. the name of the image, caption number (0 to 4) and the actual caption as shown

| 101654506_8eb26cfb60.jpg\#0 A brown and white dog is running through the snow .                       |
|-------------------------------------------------------------------------------------------------------|
| 101654506_8eb26cfb60.jpg\#1 A dog is running in the snow                                              |
| 101654506_8eb26cfb60.jpg\#2 A dog running through snow .                                              |
| 101654506_8eb26cfb60.jpg\#3 a white and brown dog is running through a snow covered field .           |
| 101654506_8eb26cfb60.jpg\#4 The white and brown dog is running over the surface of the snow .         |
|                                                                                                       |
| 1000268201_693b08cb0e.jpg\#0 A child in a pink dress is climbing up a set of stairs in an entry way . |
| 1000268201_693b08cb0e.jpg\#1 A girl going into a wooden building .                                    |
| 1000268201_693b08cb0e.jpg\#2 A little girl climbing into a wooden playhouse .                         |
| 1000268201_693b08cb0e.jpg\#3 A little girl climbing the stairs to her playhouse .                     |
| 1000268201_693b08cb0e.jpg\#4 A little girl in a pink dress going into a wooden cabin .                |

## Data Preprocessing and encoding - Captions

Operations at this stage:

-   Basic data cleaning operations on text data
-   Created an vocabulary of all the unique words present across all the 8000\*5 (i.e. 40000) image captions (corpus) in the data set. We get 8680 unique words.
-   Write all these captions along with their image names in a new file namely, “descriptions.txt” and save it on the disk.
-   Many of these words will occur very few times, say 1, 2 or 3 times. Since we are creating a predictive model, we would not like to have all the words present in our vocabulary but the words which are more likely to occur or which are common. This helps the model become more robust to outliers and make less mistakes. Hence we consider only those words which occur at least 10 times in the entire corpus

There are 1651 unique words in our vocabulary. However, we will append 0’s (zero padding) and thus total words = 1651+1 = 1652 (one index for the 0).

-   Then we have calculated the maximum length of a caption.
-   Note that captions are something that we want to predict. So during the training period, captions will be the target variables (Y) that the model is learning to predict.

But the prediction of the entire caption, given the image does not happen at once. We will predict the caption word by word. Thus, we need to encode each word into a fixed sized vector. For which we will create two Python Dictionaries namely “wordtoix” (pronounced — word to index) and “ixtoword” (pronounced — index to word).

Stating simply, we will represent every unique word in the vocabulary by an integer (index). Each of the 1652 unique words in the corpus will be represented by an integer index between 1 to 1652.

wordtoix[‘abc’] -\> returns index of the word ‘abc’

ixtoword[k] -\> returns the word whose index is ‘k’

## Loading the training set

-   We have loaded 6000 training images and the descriptions of these images from “descriptions.txt” (saved on the hard disk) in to the Python dictionary “train_descriptions”. However, when we load them, we will add two tokens in every caption as follows

    ‘startseq’ -\> This is a start sequence token which will be added at the start of every caption.

    ‘endseq’ -\> This is an end sequence token which will be added at the end of every caption.

## Data Preprocessing - Images

-   Images are nothing but input (X) to our model and must be given in the form of a fixed sized vector. For this purpose, we opt for transfer learning by using the InceptionV3 model, trained on Imagenet dataset to perform image classification on 1000 different classes of images. However, our purpose here is not to classify the image but just get fixed-length informative vector for each image. This process is called automatic feature engineering.

Hence, we just remove the last softmax layer from the model and extract a 2048 length vector (bottleneck features) for every image as follows:

![](media/3e1887aac238c6673ff64ef3cba36f40.png)

-   Once we created the model, we pass every image to this model to get the corresponding 2048 length feature vector
-   Save all the bottleneck train features in a Python dictionary and save it on the disk using Pickle file, namely “encoded_train_images.pkl” whose keys are image names and values are corresponding 2048 length feature vector.
-   Similarly we encode all the test images and save them in the file “encoded_test_images.pkl”.

## Data Preparation using Generator Function

-   This stage is very important since we need to prepare the data in a manner which will be convenient to be given as input to the deep learning model.

**Word Embeddings**

**Model Architecture**

**Inference**

**Evaluation**

**Conclusion and Future work**

**References**
