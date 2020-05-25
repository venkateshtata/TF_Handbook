from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

MODELS = {
	"vgg16": (VGG16, (224, 224)),
	"vgg19": (VGG19, (224, 224)),
	"inception": (InceptionV3, (299, 299)),
	"xception": (Xception, (299, 299)),
	"resnet": (ResNet50, (224, 224))
}


def image_load_and_convert(image_path, model):
	input_shape = MODELS[model][1]
	preprocess = imagenet_utils.preprocess_input

	image = load_img(image_path, target_size=input_shape)
	image = img_to_array(image)

	image = np.expand_dims(image, axis=0)
	image = preprocess(image)

	return image

def classify_image(image_path, model):
	img = image_load_and_convert(image_path, model)
	Network = MODELS[model][0]
	model = Network(weights='imagenet')
	preds = model.predict(img)
	P = imagenet_utils.decode_predictions(preds)
	for (i, (imagenetID, label, prob)) in enumerate(P[0]):
		print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

classify_image("dog.jpg", "vgg16")
