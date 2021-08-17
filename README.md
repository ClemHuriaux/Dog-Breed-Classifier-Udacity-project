# Udacity's Dog's breed classifier project


## Summary
1. [Introduction](#Introduction)
2. [Requirements](#Requirements)
3. [Description of files](#Description)
4. [Running the project](#Running)
5. [Conclusion](#Conclusion)
6. [Web site](#Website)

<a name="Introduction"></a>
## Introduction
This project I realized is the final one of the Udacity's Data Science Nano Degree. The goal of this project is to use a CNN (Convolutionnal Neural Network) classifier 
to:
  1. Detect the breed of a dog in an image we give
  2. If we gave an image with a human face on it, the model should identify to what dog's breed the face looks like.

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
breed are almost the same for this (Maltese, Hawanese etc.). But I guess human face is close to these breeds. Anyway Thank you for this project !

<a name="Website"></a>
## Web site
![Website index](https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/screenshot/1stStep.PNG)
![Website prediction](https://github.com/ClemHuriaux/Dog-Breed-Classifier-Udacity-project/blob/master/screenshot/Step%202.PNG)
