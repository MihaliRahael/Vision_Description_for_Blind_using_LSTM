# **Vision Description for blind aid using LSTM : Teaching Computers to describe pictures**

## **Problem Statement**

 Receive an image as input and produces a relevant caption or short description for that

## **Data collection**

There are many open source datasets available for this problem, like Flickr 8k (containing8k images), Flickr 30k (containing 30k images), MS COCO (containing 180k images), etc. But for the purpose of this case study, I have used the Flickr 8k dataset from Kaggle. This dataset contains 8000 images each with 5 captions (as we have already seen in the Introduction section that an image can have multiple captions, all being relevant simultaneously).

These images are bifurcated as follows:

Training Set — 6000 images

Dev Set — 1000 images

Test Set — 1000 images

## **Understanding the data**

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

## **Data Preprocessing and encoding - Captions**

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

## **Loading the training set**

-   We have loaded 6000 training images and the descriptions of these images from “descriptions.txt” (saved on the hard disk) in to the Python dictionary “train_descriptions”. However, when we load them, we will add two tokens in every caption as follows

    ‘startseq’ -\> This is a start sequence token which will be added at the start of every caption.

    ‘endseq’ -\> This is an end sequence token which will be added at the end of every caption.

## **Data Preprocessing - Images**

-   Images are nothing but input (X) to our model and must be given in the form of a fixed sized vector. For this purpose, we opt for transfer learning by using the InceptionV3 model, trained on Imagenet dataset to perform image classification on 1000 different classes of images. However, our purpose here is not to classify the image but just get fixed-length informative vector for each image. This process is called automatic feature engineering. Hence, we just remove the last softmax layer from the model and extract a 2048 length vector (bottleneck features) for every image as follows:

![image](https://user-images.githubusercontent.com/106816732/213872813-d1e75e84-8952-4873-b9ea-7733513d5720.png)

-   Once we created the model, we pass every image to this model to get the corresponding 2048 length feature vector
-   Save all the bottleneck train features in a Python dictionary and save it on the disk using Pickle file, namely “encoded_train_images.pkl” whose keys are image names and values are corresponding 2048 length feature vector.
-   Similarly we encode all the test images and save them in the file “encoded_test_images.pkl”.

## **Data Preparation using Generator Function**

This stage is very important since we need to prepare the data in a manner which will be convenient to be given as input to the deep learning model.

Consider we have 3 images and their 3 corresponding captions as follows:

![image](https://user-images.githubusercontent.com/106816732/213872853-932c068e-84ad-41b8-940a-3c06e91fdb16.png)

*(Train image 1) Caption -\> The black cat sat on grass*

![image](https://user-images.githubusercontent.com/106816732/213872864-cf16d0a0-586c-4085-9579-95ddd2d45518.png)

*(Train image 2) Caption -\> The white cat is walking on road*

![image](https://user-images.githubusercontent.com/106816732/213872872-208e8803-54f4-454f-a2ee-2d15f792427a.png)

*(Test image) Caption -\> The black cat is walking on grass*

Now, let’s say we use the first two images and their captions to train the model and the third image to test our model. Now the questions that need to be answered are:

How do we frame this as a supervised learning problem?

What does the data matrix look like?

How many data points do we have?, etc.

First we need to convert both the images to their corresponding 2048 length feature vector as discussed above. Let “Image_1” and “Image_2” be the feature vectors of the first two images respectively.

Secondly, let’s build the vocabulary for the first two (train) captions by adding the two tokens “startseq” and “endseq” in both of them.

Caption_1 -\> “startseq the black cat sat on grass endseq”

Caption_2 -\> “startseq the white cat is walking on road endseq”

vocab = {black, cat, endseq, grass, is, on, road, sat, startseq, the, walking, white}

Let’s give an index to each word in the vocabulary:

black -1, cat -2, endseq -3, grass -4, is -5, on -6, road -7, sat -8, startseq -9, the -10, walking -11, white -12

Now let’s try to frame it as a supervised learning problem where we have a set of data points D = {Xi, Yi}, where Xi is the feature vector of data point ‘i’ and Yi is the corresponding target variable.

Let’s take the first image vector Image_1 and its corresponding caption “startseq the black cat sat on grass endseq”. Recall that, Image vector is the input and the caption is what we need to predict. But the way we predict the caption is as follows:

For the first time, we provide the image vector and the first word as input and try to predict the second word, i.e.:

Input = Image_1 + ‘startseq’; Output = ‘the’

Then we provide image vector and the first two words as input and try to predict the third word, i.e.:

Input = Image_1 + ‘startseq the’; Output = ‘cat’

And so on…

Thus, we can summarize the data matrix for one image and its corresponding caption as follows:

![image](https://user-images.githubusercontent.com/106816732/213872900-24654a34-071e-4e50-8371-0dbbc6c9f7e6.png)

*Data points corresponding to one image and its caption*

It must be noted that, one image+caption is not a single data point but are multiple data points depending on the length of the caption. Similarly if we consider both the images and their captions, our data matrix will then look as follows:

![image](https://user-images.githubusercontent.com/106816732/213872911-30904420-5f24-464b-9f89-6907658c5366.png)

*Data Matrix for both the images and captions*

So in every data point, it’s not just the image which goes as input to the system, but also, a partial caption which helps to predict the next word in the sequence. Since we are processing sequences, we will employ a Recurrent Neural Network to read these partial captions.

However, we have already discussed that we are not going to pass the actual English text of the caption, rather we are going to pass the sequence of indices where each index represents a unique word. Since we have already created an index for each word, let’s now replace the words with their indices and understand how the data matrix will look like:

![image](https://user-images.githubusercontent.com/106816732/213872924-640afd0f-b4c5-4e4e-b696-bcf604099474.png)

*Data matrix after replacing the words by their indices*

Since we would be doing **batch processing**, we need to make sure that each sequence is of equal length. Hence we need to append 0’s (zero padding) at the end of each sequence. But how many zeros should we append in each sequence? Well, this is the reason we had calculated the maximum length of a caption, which is 33. So we will append those many number of zeros which will lead to every sequence having a length of 33.

The data matrix will then look as follows:

![image](https://user-images.githubusercontent.com/106816732/213872937-6ff5c8ef-c8dd-47cf-a7c7-ac12971d0a78.png)

*Appending zeros to each sequence to make them all of same length 34*

### **Need for a Data Generator**

However, there is a big catch in this way of dataset preparation. In the above example, I have only considered 2 images and captions which have lead to 15 data points. But in our actual training dataset we have 6000 images, each having 5 captions. This makes a total of 30000 images and captions. Even if we assume that each caption on an average is just 7 words long, it will lead to a total of 30000\*7 i.e. 210000 data points.

Compute the size of the data matrix:

![image](https://user-images.githubusercontent.com/106816732/213872946-9e32c160-8f43-40a0-bff5-0b42017a8d29.png)

Size of the data matrix = n\*m

where n-\> number of data points (assumed as 210000) and m-\> length of each data point.

Clearly m= Length of image vector(2048) + Length of partial caption(x).

m = 2048 + x

But what is the value of x? Its not 33. Because every word (or index) will be mapped (embedded) to higher dimensional space through one of the word embedding techniques. During the model building stage, each word/index is mapped to a 200-long vector using a pre-trained **GLOVE word embedding** model.

Now each sequence contains 33 indices, where each index is a vector of length 200. Therefore x = 33\*200 = 6800

Hence, m = 2048 + 6800 = 8848.

Finally, size of data matrix= 210000 \* 8848= 1858080000 blocks.

Now even if we assume that one block takes 2 byte, then, to store this data matrix, we will require more than 3 GB of main memory. This is pretty huge requirement and even if we are able to manage to load this much data into the RAM, it will make the system very slow.

For this reason we use data generators a lot in Deep Learning. Data Generators are a functionality which is natively implemented in Python. The ImageDataGenerator class provided by the Keras API is nothing but an implementation of generator function in Python. So how does using a generator function solve this problem?

From the fundamentals of Deep Learning, we know that, to train a model on a particular dataset, we use some version of Stochastic Gradient Descent (SGD) like Adam, Rmsprop, Adagrad, etc. With SGD, we do not calculate the loss on the entire data set to update the gradients. Rather, in every iteration, we calculate the loss on a batch of data points (typically 64, 128, 256, etc.) to update the gradients.

This means that we do not require to store the entire dataset in the memory at once. Even if we have the current batch of points in the memory, it is sufficient for our purpose.

**A generator function in Python is used exactly for this purpose. It’s like an iterator which resumes the functionality from the point it left the last time it was called.**

## **Word Embeddings**

-   As already stated above, we will map the every word (index) to a 200-long vector and for this purpose, we will use a pre-trained GLOVE Model
-   Now, for all the 1652 unique words in our vocabulary, we create an embedding matrix which will be loaded into the model before training.
-   Notice that since we are using a pre-trained embedding layer, we need to freeze it (trainable = False), before training the model, so that it does not get updated during the backpropagation.
    ```
    model.layers[2].set_weights([embedding_matrix])  
    model.layers[2].trainable = False
    ```

## **Model Architecture**

-   Since the input consists of two parts, an image vector and a partial caption, we cannot use the Sequential API provided by the Keras library. For this reason, we use the Functional API which allows us to create Merge Models.

![image](https://user-images.githubusercontent.com/106816732/213873000-17ac50b0-9b4a-411f-9abb-a1cfadf7c48a.png)

We define the model as follows:
```
# image feature extractor model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# partial caption sequence model
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
# decoder (feed forward) model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
\# merge the two input models
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
```

The below plot helps to visualize the structure of the network and better understand the two streams of input:

![image](https://user-images.githubusercontent.com/106816732/213873053-33f86a88-3357-43b0-add9-2bd8a21ef023.png)

Finally the weights of the model will be updated through backpropagation algorithm and the model will learn to output a word, given an image feature vector and a partial caption. So in summary, we have:

Input_1 -\> Partial Caption

Input_2 -\> Image feature vector

Output -\> An appropriate word, next in the sequence of partial caption provided in the input_1 (or in probability terms we say conditioned on image vector and the partial caption)

## **How prediction works (Maximum Likelihood Estimation (MLE) and Greedy Search)**

Now we will understand how do we test (infer) our model by passing in new images, i.e. how can we generate a caption for a new test image. Recall that in the example where we saw how to prepare the data, we used only first two images and their captions. Now let’s use the third image and try to understand how we would like the caption to be generated. The third image vector and caption were as follows:

![image](https://user-images.githubusercontent.com/106816732/213873067-5fc6dc79-4539-4576-a187-68c76222abd8.png)

*Caption -\> the black cat is walking on grass*

Also the vocabulary in the example was: vocab = {black, cat, endseq, grass, is, on, road, sat, startseq, the, walking, white}

We will generate the caption iteratively, one word at a time as follows:

Iteration 1:

Input: Image vector + “startseq” (as partial caption)

Expected Output word: “the”

(This is the importance of the token ‘startseq’ which is used as the initial partial caption for any image during prediction).

But wait, here the model generates a 12-long vector in the sample example (while 1652-long vector in the original example) which is a probability distribution across all the words in the vocabulary (since we used softmax classifier). For this reason we greedily select the word with the maximum probability, given the feature vector and partial caption.

If the model is trained well, we must expect the probability for the word “the” to be maximum:

![image](https://user-images.githubusercontent.com/106816732/213873077-5f1233c1-7ec2-42f5-9c52-44af59e2aaf0.png)

This is called as **Maximum Likelihood Estimation (MLE)** i.e. we select that word which is most likely according to the model for the given input. And sometimes this method is also called as **Greedy Search**, as we greedily select the word with maximum probability.

Iteration 2:

Input: Image vector + “startseq the”

Expected Output word: “black”

![image](https://user-images.githubusercontent.com/106816732/213873082-b25d7d6a-c488-4dfc-840b-85f33e3cca6d.png)

Likewise at Iteration 8:

Input: Image vector + “startseq the black cat is walking on grass”

Expected Output word: “endseq”

![image](https://user-images.githubusercontent.com/106816732/213873097-e95121da-7c9d-4159-b04e-af520b2723c6.png)

This is where we stop the iterations. So we stop when either of the below two conditions is met:

-   We encounter an ‘endseq’ token which means the model thinks that this is the end of the caption. (Here is the importance of the ‘endseq’ token)
-   We reach a maximum threshold of the number of words generated by the model.

If any of the above conditions is met, we break the loop and report the generated caption as the output of the model for the given image.

Code for greedy Search
```
def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
```

## **Evaluation**

To understand how good the model is, let’s try to generate captions on images from the test dataset.
<img width="343" alt="image" src="https://user-images.githubusercontent.com/106816732/213873228-6306949f-16d7-426e-bb02-7669bc2c09b3.png">
<img width="285" alt="image" src="https://user-images.githubusercontent.com/106816732/213873242-b12dce11-ca00-48a3-b6b4-55eee72ddf4e.png">
<img width="294" alt="image" src="https://user-images.githubusercontent.com/106816732/213873264-15dcc188-c9a7-4828-bc22-c4fa21eede3e.png">

## **Inferences and Future work**

-   The model is not perfect in finding all aspects of the image, semantic meaning, colors, grammer of captions etc
-   Since this is a naive first-cut model, without any rigorous hyper-parameter tuning, does a decent job in generating captions for images.

We can improve the model by

-   Using a larger dataset.
-   Changing the model architecture, e.g. include an attention module.
-   Doing more hyper parameter tuning (learning rate, batch size, number of layers, number of units, dropout rate, batch normalization etc.).
-   Use the cross validation set to understand overfitting.
-   Using Beam Search instead of Greedy Search during Inference.
-   Using BLEU Score to evaluate and measure the performance of the model.

## **References**

<https://www.appliedaicourse.com/>

<https://ineuron.ai/course/Data-Science-Industry-Ready-Projects>

<https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8>

<https://cs.stanford.edu/people/karpathy/cvpr2015.pdf>

<https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/>
