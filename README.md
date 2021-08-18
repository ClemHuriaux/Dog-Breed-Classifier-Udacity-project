# Udacity's Dog's breed classifier project


## Summary
1. [Project Overview](#Introduction)
2. [Project Statement](#projectStatement)
3. [Metrics](#Metrics)
4. [Data Exploration](#Analysis1)
5. [Data Visualization](#Analysis2)
6. [Data Preprocessing](#DataPre)
7. [Implementation](#Implementation)
8. [Refinement](#Refinement)
9. [Model Evaluation and Validation](#ModelEv)
10. [Justification](#Justification)
11. [Reflection](#Reflection)
12. [Improvement](#Improvement)
13. [Description of files](#Description)
14. [Requirements](#Requirements)
15. [Running the project](#Running)
16. [WebSite Sreenchots](#Website)

<a name="Introduction"></a>
## Project Overview
This project is the Capstone project of the Udacity's Data Science Nanodegree. This project is about CNN (Convolutionnal Neural Network). A CNN is a neural network architecture which is particularly good in computer visions tasks. Here we are using it for a an Image Classifier task. However, training CNNs is a long process (It can take several weeks !) so we're going to use transfer learning in this project. Transfer learning is about taking an already trained Neural Network to use it in our project. We can add layers, but it will be way faster than training the whole model.

<a name="projectStatement"></a>
## Project Statement
The goal of this project is to construct a dog's breed classifier for user. The idea is that when a user give an image of a dog, the model must find the breed of this dog. Plus, if the user give an image with a human face on it, the model should return the closest dog's breed this face looks like.
In addition, a small web-app is asked in order to allow the user an easy interaction support. Here's the step of this project:
  1. Datasets
  2. Detect Human
  3. Detect Dogs
  4. Create a CNN to Classify Dog Breeds (from Scratch)
  5. Use a CNN to Classify Dog Breeds
  6. Create a CNN to Classify Dog Breeds (using Transfer Learning)
  7. Write your algorithm

I was not particularly waiting for a great accuracy since I know CNNs' training are really long. Let's see what we have !

<a name="Metrics"></a>
## Metrics
I am using the accuracy to evaluate models for this project. Accuracy is an intuitive measurement for classification problems.

Accuracy is a good choice when datasets are not unbalanced. As we have plenty of dogs and human faces images, it seems to be a good choice here.

<a name="Analysis1"></a>
## Data Exploration
We are using 2 datasets provided by Udacity. A dataset with human faces and another with dogs images. We want to split data into train/validation/test sets. Here's a small list of our data:

  - There are 133 total dog categories.
  - There are 8351 total dog images.

  - There are 6680 training dog images.
  - There are 835 validation dog images.
  - There are 836 test dog images.
 
  - There are 13233 total human images.
 
 <a name="Analysis2"></a>
 ## Data Visualization
 We can take a look at our data and specially they are seen with cv2 by using the CascadeFaces xml. Example:
 ![face_detection](https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/screenshot/face.png)
 
 <a name="DataPre"></a>
 ## Data Preprocessing
 When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

```(nb_samples,rows,columns,channels),```

where nb_samples corresponds to the total number of images (or samples), and rows, columns, and channels correspond to the number of rows, columns, and channels for each image, respectively. 
So this is a thing we need to handle in our project. You can see it in the jupyter Notebook.

<a name="Implementation"></a>
## Implementation
I first tried to implement a Neural Network from scratch. The specs are:
<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/screenshot/model.PNG" />

As we can see I am using 4 CNNs layer with MaxPooling for dimmension reduction. For the output of this part I'm using GlobalPooling to drasticaly reduce dimmensions.
I then used a Fully connected network with 2 layer and dropout to prevent from overfitting. I used ReLU function for all layers as it works realy great.
Finnaly, my output layer is a 133 neurons one (1 for each breed).

That gives me the result of more than 9%. Which is not so great but, as this is so hard for DL, Udacity asked to get more than 1% of accuracy

<a name="Refinement"></a>
## Refinement
After we realized how hard it is to train model, it was time to improve our accuracy. So we used transfer learning.
I used the ResNet50 model as a starter. I achived a result of more than 80% of accuracy. The gap with the first one is really huge !

<a name="ModelEv"></a>
## Model Evaluation and Validation
The model is simpler here:

<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/screenshot/model2.PNG" />

I just have the pre-trained ResNet50 model and I added an output layer for the breed classification. The minimum accuracy to pass is 60%, but the model performed really well with 80%.
Let me show you some results:

<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/brad%20pitt.jpg" width="400" />
Output: 'This person looks like the breed: English Springer Spaniel'

<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/golden_retriver.jpg" width="400" />
Output: 'The preticted breed for this dog is: Golden Retriever'

<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/malinois.jpg" width="400" />
Output: 'The preticted breed for this dog is: Belgian Malinois'

<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/pitbull.jpg" width="400" />
Output: 'The preticted breed for this dog is: American Staffordshire Terrier'

<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/scarlett.jpg" width="400" />
Output: 'This person looks like the breed: Silky Terrier'

<img src="https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/reconBreed/data/cat.jpg" width="400" />
Output: "Looks like we can't detect a dog neither a human in this picture. We're sorry."

<a name="Justification"></a>
## Justification
The improvement is so huge thanks to the pre-trained model. The ResNet50 model is a complec neural network and has been train for way longer than the model we did from scratch. This is why this is so good now.

<a name="Reflection"></a>
## Reflection
We now have a great model to predict dog's breed. Thanks to transfer learning and patience, we did a usefull classifier. It was actually fun to work on it. One difficult thing was to not be lost on all details specially about shapes, transformation etc. For example I lost some times when doing the web-app, as versions between the notebook and my local virtual env were too different and I could not install Keras 2.0.2 anymore.

<a name="Improvement"></a>
## Improvement
We could try to increase accuracy by doing data augmentation, running the training for a longer epochs size and fine tune each hyperparameters. I did try data augmentation but the process was really slow and the accuracy didn't really improve. My thought is that my model wasn't complex enough and there was not enough epochs.

<a name="Description"></a>
## Description of files
This project contains:
  - the Jupyter Notebook I used as a first step.
  - the requirements to install required packages
  - The screenshot foler is to store images I display here
  - all others files are for the app and the model

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

<a name="Website"></a>
## WebSite Sreenchots
![Website index](https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/screenshot/1stStep.PNG)
![Website prediction](https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/screenshot/Step%202.PNG)
