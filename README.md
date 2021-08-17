# Udacity's Dog's breed classifier project


## Summary
1. [Project Definition](#Introduction)
2. [Requirements](#Requirements)
3. [Description of files](#Description)
4. [Analysis](#Analysis)
5. [Running the project](#Running)
6. [Conclusion](#Conclusion)
7. [Web site](#Website)

<a name="Introduction"></a>
## Project Definition
This project I realized is the final one of the Udacity's Data Science Nano Degree. The goal of this project is to use a CNN (Convolutionnal Neural Network) classifier 
to:
  1. Detect the breed of a dog in an image we give
  2. If we gave an image with a human face on it, the model should identify to what dog's breed the face looks like.

Udacity provided 2 datasets. The first one with dogs and the second one with human faces.

Note: The model have to have at least 60% of accuracy. The model I proposed here has 80% (you can check it in the Jupyter Notebook)
On top of that, there is a small web-app to communicate with model. On it, you can upload the image you want to run the model. Please note it can be quite long getting 
the result. You can see pictures of the website on the bottom of this file.

<a name="Requirements"></a>
## Requirements
In order to run this project, you'll need several libraries. You can find the list below:
  * python 3.6.13
  * opencv-python
  * h5py
  * matplotlib
  * numpy
  * scipy
  * tqdm
  * keras 2.1.6
  * tensorflow-gpu 1.15.0 (If you can run tensorflow on gpu only)
  * tensorflow 1.15.0
  * plotly-express
  
Be carefull, the app won't work if you have a greater version than Keras 2.2.0.
To be sure the project can run on your computer, I added a requirement.txt (with gpu and not) with all the libraries I had. You can install all those packages by 
running:
```pip install -m requirements.txt```
or
```pip install -m requirements-gpu.txt```

Alternatively, if you decide to install the package in the list without the requirements.txt, you can use:
```pip install <PackageName>```

### Anaconda
If you are a user of anaconda as I am, you can install the requirements with:
``` conda install --file requirements.txt```
or
``` conda install --file requirements-gpu.txt```

<a name="Description"></a>
## Description of files
This project contains:
  - the Jupyter Notebook I used as a first step.
  - the requirements to install required packages
  - The screenshot foler is to store images I display here
  - all others files are for the app and the model
 
<a name="Analysis"></a>
## Analysis
Note: I STRONGLY advise you to open the jupyter notebook as you will find more informations and justification in it.

### 1st Step - Datasets
After splitting the datasets into train/validation/test sets, we have the following informations:

  - There are 133 total dog categories.
  - There are 8351 total dog images.

  - There are 6680 training dog images.
  - There are 835 validation dog images.
  - There are 836 test dog images.
 
  - There are 13233 total human images.

### 2nd Step - Detect Human
By using the "CascadeClassifier" from cv2, we can easily identify human faces. Here's an example:

![face_detection](https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/screenshot/face.png)

### 3rd Step - Detect Dogs
This step is aimed to detect dogs on images. By using the ResNet50 model, we obtained a perfect score (100% accuracy). This is great for the rest of our work.

### 4th Step - Create a CNN to Classify Dog Breeds (from Scratch)
It is not difficult to find dogs but it is to find their breed ! Even for a human it is hard. That's why, the requirement for this model was at least 1% of accuracy
Mine get an accuracy of more than 9%. We can find in the jupyter Notebook the details and explanations for this model.

### 5th Step - Use a CNN to Classify Dog Breeds
Now in order to get  a better accuracy but with a raisonable training time, we will use transfer learning. The model we are going to use here is the VGG16.
Just by adding a fully connected layer for the output, in order to classify dogs (133 neurons - 1 for each breed) we obtain an accuracy of more than 40%. Great improvment !

### 6th Step - Create a CNN to Classify Dog Breeds (using Transfer Learning)
It is now time to implement our model using transfer learning.
So for this step I decided to use the ResNet50 model because it was perfect for the dog detection. I just added a fully connected layer for the output and ran it for 50 epochs with a batch sizes of 20.
With that, I get an accuracy of more than 80%. Now we have a quite serious model !

### 7th Step - Write your algorithm
Then I just wrote a simple algorithm to run the correct detection for images and to return the correct information

### 8th Step - Results
Even if you can find it in the notebook, here are the results:

<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/brad%20pitt.jpg" width="500" />
Output: 'This person looks like the breed: English Springer Spaniel'
<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/golden_retriver.jpg" width="500" />
Output: 'The preticted breed for this dog is: Golden Retriever'
<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/malinois.jpg" width="500" />
Output: 'The preticted breed for this dog is: Belgian Malinois'
<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/pitbull.jpg" width="500" />
Output: 'The preticted breed for this dog is: American Staffordshire Terrier'
<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/scarlett.jpg" width="500" />
Output: 'This person looks like the breed: Silky Terrier'
<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/cat.jpg" width="500" />
Output: "Looks like we can't detect a dog neither a human in this picture. We're sorry."

### 9th Step - Doing the application
The last step was to design the application or write a blog post. I decided to do a small app as i think this is more interesting to experiment than just reading the article. At least that's what I prefer :).

<a name="Running"></a>
## Running the project
Follow the following steps to run the project:
1. Clone the repo on your local system with: ```git clone https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/```
2. Go in the folder
3. Then, run the local server with: ```python manage.py runserver```
If it works, you should have this output:
Image
4. Now go to your local server at the following address: http://127.0.0.1/
![Server running](https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/screenshot/runserver.PNG)
And here we go, You can upload the image you want :)

<a name="Conclusion"></a>
## Conclusion
I had lost of fun doing this project. As you can see, the model is actually pretty good at finding the breed. As long as the image we give is a good one (not too
much noise, good focus etc.) the model performs really great. The fact that it give the closest breed for a human, is actually fun too. I just observed the 
breed are almost the same for this (Maltese, Hawanese etc.). But I guess human face is close to these breeds. We could try to increase accuracy by doing data augmentation, running the training for a longer epochs size and fine tune each hyperparameters. I did try data augmentation but the process was really slow and the accuracy didn't really improve. My thought is that my model wasn't complex enough and mo epochs to low.

<a name="Website"></a>
## Web site
![Website index](https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/screenshot/1stStep.PNG)
![Website prediction](https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/screenshot/Step%202.PNG)
