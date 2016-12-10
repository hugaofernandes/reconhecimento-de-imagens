
# -*- coding: utf-8 -*-
#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import os
from PIL import Image
from StringIO import StringIO

def process_image(image, blocks=4):
    if not image.mode == 'RGB':
        return None
    feature = [0] * blocks * blocks * blocks
    #feature = [0] * (blocks ** (blocks - 1))
    pixel_count = 0.0
    for pixel in image.getdata():
        ridx = int(pixel[0]/(256/blocks))
        gidx = int(pixel[1]/(256/blocks))
        bidx = int(pixel[2]/(256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    return [x/pixel_count for x in feature]

def process_image_file(image_path):
    image_fp = StringIO(open(image_path, 'rb').read())
    try:
        image = Image.open(image_fp)
        return process_image(image)
    except IOError:
        return None

def dataset(_dataset, path_dir, _class):
	for i in os.listdir(path_dir):
		_dataset = _dataset.append({'image' : process_image_file(path_dir + i), 'class' : _class}, ignore_index=True)
	return _dataset

dataset_learn = pd.DataFrame(columns=['image', 'class'])
dataset_learn = dataset(dataset_learn, './dataset_learn/cersei_lannister/', 'Cersei Lannister')
dataset_learn = dataset(dataset_learn, './dataset_learn/daenerys_targaryen/', 'Daenerys Targaryen')
dataset_learn = dataset(dataset_learn, './dataset_learn/jon_snow/', 'Jon Snow')
dataset_learn = dataset(dataset_learn, './dataset_learn/tyrion_lannister/', 'Tyrion Lannister')

dataset_test = pd.DataFrame(columns=['image', 'class'])
dataset_test = dataset(dataset_test, './dataset_test/cersei_lannister/', 'Cersei Lannister')
dataset_test = dataset(dataset_test, './dataset_test/daenerys_targaryen/', 'Daenerys Targaryen')
dataset_test = dataset(dataset_test, './dataset_test/jon_snow/', 'Jon Snow')
dataset_test = dataset(dataset_test, './dataset_test/tyrion_lannister/', 'Tyrion Lannister')

def acuracia_media(_n, _classifier, _dataset_learn):
	media = 0.0
	x = _dataset_learn['image'].values.tolist()
	y = _dataset_learn['class'].values.tolist()
	for i in range(_n):
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)
		_classifier.fit(x_train, y_train)
		_expected = y_test
		_predicted = _classifier.predict(x_test)
		media += accuracy_score(_expected, _predicted) * 100
	print (media / _n)

#acuracia_media(10, SVC(), dataset_learn)
#acuracia_media(10, KNeighborsClassifier(), dataset_learn)
#acuracia_media(10, DecisionTreeClassifier(), dataset_learn)
#acuracia_media(10, GaussianNB(), dataset_learn)
#acuracia_media(10, RandomForestClassifier(), dataset_learn)

def classifier(_classifier, _dataset_learn, _image):
	x = _dataset_learn['image'].values.tolist()
	y = _dataset_learn['class'].values.tolist()
	_classifier.fit(x, y)
	_predicted = _classifier.predict(_image)
	print (_predicted[0])

#image = process_image_file('./dataset_test/daenerys_targaryen/images5.jpg')
#classifier(RandomForestClassifier(), dataset_learn, image)

def classifier_path(_path, _classifier, _dataset):
	for i in os.listdir(_path):
		_image = process_image_file(_path + i)
		classifier(_classifier, _dataset, _image)

#classifier_path('./dataset_test/cersei_lannister/', RandomForestClassifier(), dataset_learn)
#classifier_path('./dataset_test/daenerys_targaryen/', RandomForestClassifier(), dataset_learn)
#classifier_path('./dataset_test/jon_snow/', RandomForestClassifier(), dataset_learn)
#classifier_path('./dataset_test/tyrion_lannister/', RandomForestClassifier(), dataset_learn)

def classifier_dataset_test(_dataset_learn, _dataset_test, _classifier):
	x = _dataset_learn['image'].values.tolist()
	y = _dataset_learn['class'].values.tolist()
	x_test = _dataset_test['image'].values.tolist()
	y_test = _dataset_test['class'].values.tolist()
	_classifier.fit(x, y)
	#_list = []
	result = pd.DataFrame(columns=['true', 'predicted'])
	for _image, _true in zip(x_test, y_test):
		_predicted = _classifier.predict(_image)
		#_list.append([_true, _predicted[0]])
		result = result.append({'true' : _true, 'predicted' : _predicted[0]}, ignore_index=True)
	#print (_list)
	print (result)

classifier_dataset_test(dataset_learn, dataset_test, RandomForestClassifier())


#REFERENCES:
#http://www.ippatsuman.com/2014/08/13/day-and-night-an-image-classifier-with-scikit-learn/
#http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html




