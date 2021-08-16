import plotly as plotly
from django.shortcuts import render
import cv2
import pickle
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input
from .forms import *
import os
import plotly.express as px


def index(request):
    path_image = './media/images'
    file_in_images = os.listdir(path_image)
    if len(file_in_images):
        os.remove(f'{path_image}/{file_in_images[0]}')

    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            name_img = os.listdir(path_image)[0]
            model = CnnClass()
            img_full_path = f'{path_image}/{name_img}'
            result = model.predict(img_full_path)
            image_face = model.plot_face(img_full_path)
            return render(request, 'reconBreed/uploaded.html', {
                'img': name_img,
                'breed': result.split(":")[1],
                'img_face': image_face
            })
    else:
        form = ImageForm()
    return render(request, 'reconBreed/index.html', {'form': form})


class CnnClass:
    """ Use the Cnn model realized for the final projet of Udacity Data Science Nanodegree """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('./reconBreed/haarcascades/haarcascade_frontface_alt.xml')
        self.model = load_model('./reconBreed/cnn-model/My_Resnet_Model')
        self.model.load_weights('./reconBreed/cnn-model/weights.best.ResNet50_2.hdf5')
        self.resnet50_model = ResNet50(weights='imagenet')
        with open("./reconBreed/data/dog_names", "rb") as file:
            self.dog_names = pickle.load(file)

    def predict(self, img_path):
        """ Run the model on the image

        This function will run the model on the image we want. It will return the predicted breed if it's a dog image,
        the breed the human looks like if it's an image of a person. If its neither a dog or a person, it will say that
        the model can't predict anything

        INPUTS:
        -------
        img_path: str
            The path of the image we want to use

        RETURNS:
        --------
        str:
            The result of the prediction
        """
        if self.dog_detector(img_path):
            breed = self.resnet50_predict_breed(img_path)
            breed = breed.rsplit('.')[1].replace("_", " ").title()

            return f"The predicted breed for this dog is: {breed}"

        elif self.face_detector(img_path):
            estimated_breed = self.resnet50_predict_breed(img_path)
            estimated_breed = estimated_breed.rsplit('.')[1].replace("_", " ").title()
            return f"This person looks like the breed: {estimated_breed}"

        return "Looks like we can't detect a dog neither a human in this picture. We're sorry."

    def resnet50_predict_breed(self, img_path):
        """ Give the breed of a dog/human

        This function use the ResNet50 model (already trained) to give the breed of the dog or the human (at least the
        closest one) we want to

        INPUTS:
        -------
        img_path: str
            The path of the image we want to use

        RETURNS:
        --------
        str:
            The identified breed
        """
        bottleneck_feature = self._extract_resnet50(self._path_to_tensor(img_path))
        predicted_vector = self.model.predict(bottleneck_feature)
        return self.dog_names[np.argmax(predicted_vector)]

    def face_detector(self, img_path):
        """ Simply detect if there is a face on the image we give

        INPUTS:
        -------
        img_path: str
            The path of the image we want to use

        RETURNS:
        --------
        Boolean:
            True if there is a face, False if not
        """
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def dog_detector(self, img_path):
        """ Simply detect if there is a dog on the image we give

        INPUTS:
        -------
        img_path: str
            The path of the image we want to use

        RETURNS:
        --------
        Boolean:
            True if there is a dog, False if not
        """
        prediction = self._ResNet50_predict_labels(img_path)
        return (prediction <= 268) & (prediction >= 151)

    @staticmethod
    def plot_face(img_path):
        """ Plot the image we put in, in a plot figure

        INPUTS:
        -------
        img_path: str
            The path of the image we want to use

        RETURNS:
        -------
        Plot:
            The plot of the image
        """
        img = cv2.imread(img_path)
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        graph_div = plotly.offline.plot(px.imshow(cv_rgb), auto_open=False, output_type="div")
        return graph_div

    @classmethod
    def _paths_to_tensor(cls, img_paths):
        list_of_tensors = [cls._path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

    def _ResNet50_predict_labels(self, img_path):
        ResNet50_model = self.resnet50_model
        img = preprocess_input(self._path_to_tensor(img_path))
        return np.argmax(ResNet50_model.predict(img))

    @staticmethod
    def _path_to_tensor(img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    @staticmethod
    def _extract_resnet50(tensor):
        return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

