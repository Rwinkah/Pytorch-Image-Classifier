# Pytorch Image Classifier
 Image plant classifier built with pytorch to identify 102 unique species of flowers.

The classification can be done with either **VGG13**, **ResNet18**, **DenseNet121** or **Alexnet**, 

## Training The Models
The classifier is trained using transfer learning. Pretrained versions of the aforementioned models are loaded and are fine-tuned to serve the purpose of classifying flowers

## Getting Set-up
The only files that need to be run are the **train.py** and **predict.py**. The **dataprocessing.py** and **model_setup.py** modules contain functions that are called by them during the process of processing the data, training the data, and running predictions

- Run the train.py file
 Use the positional arguments and optional keyword arguments if you want more control over the training process. The only positional argument is the path to the training data
- Run the Predict.oy file
 After running the train.py file a model checkpoint will ben saved, run the predict.py file which will load the saved model into memory and accepts positional arguments:
 - path to the image to be classified
 - path to the trained model
 Other keyword arguments exist to exert greater control over the prediction:
 - top_k: Number of output predictions
 - gpu: Boolean to determine if the model is processed on gpu or cpu
 - category_names: class to name mapping
 The predicted class(es) is outputted to the console

 # NOTE
 This project was built in fufilment of the requirements for the awardment of Udacity's Nanodegree for AI and Machine Learning with Python