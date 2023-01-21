**Vision Description for blind aid using LSTM : Teaching Computers to describe pictures**

**Problem Statement**

Receive an image as input and produces a relevant caption or short desription for that.

**Data collection**

There are many open source datasets available for this problem, like Flickr 8k (containing8k images), Flickr 30k (containing 30k images), MS COCO (containing 180k images), etc. But for the purpose of this case study, I have used the Flickr 8k dataset from Kaggle. This dataset contains 8000 images each with 5 captions (as we have already seen in the Introduction section that an image can have multiple captions, all being relevant simultaneously).

These images are bifurcated as follows:

Training Set — 6000 images

Dev Set — 1000 images

Test Set — 1000 images

**Understanding the data**

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

**Data Cleaning**

Operations at this stage:

-   

**Loading the training set**

**Data Preprocessing — Images**

**Data Preprocessing — Captions**

**Data Preparation using Generator Function**

**Word Embeddings**

**Model Architecture**

**Inference**

**Evaluation**

**Conclusion and Future work**

**References**
