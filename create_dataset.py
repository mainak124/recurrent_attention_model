import numpy as np
import cv2
import os
import sys

#from matplotlib import pyplot as plt
import random
from random import randint
import math
import cPickle as pkl

from nabirds import load_bounding_box_annotations, load_part_annotations, load_part_names, load_class_names, load_image_labels, load_image_paths, load_image_sizes, load_hierarchy, load_photographers, load_train_test_split

IM_ORIG_SIZE = (256, 256)
IM_SIZE = (227, 227)

def preprocessImage(image, im_mean, im_size, is_train=1):

    im_resized = np.asarray(cv2.resize(image, IM_ORIG_SIZE), dtype = np.float64)
    if im_size == IM_SIZE[0]:
        x = randint(0,28)
        y = randint(0,28)
        im_cropped = im_resized[x:x+IM_SIZE[0], y:y+IM_SIZE[1]]
    elif im_size == IM_ORIG_SIZE[0]:    
        im_cropped = im_resized
    im_norm = im_cropped / 255
    im_norm = im_norm - im_mean
    rand_flip = randint(0,1)
    if is_train and rand_flip:
        im_norm = cv2.flip(im_norm, 1)
    
    processed_image = im_norm
    
    return processed_image

def get_dataset(dataset_path = '/raid/mainak/datasets/nabirds/'):
	with open(dataset_path, "rb") as data:
		train_set_x, train_set_y = pkl.load(data)

	return train_set_x, train_set_y

def get_data_length(dataset_path = '/raid/mainak/datasets/nabirds/'):

    # Load in the train / test split
    train_images, test_images = load_train_test_split(dataset_path)
    return len(train_images), len(test_images)


def find_image_mean(image_path, dataset_path = '/raid/mainak/datasets/nabirds/'):

    image_paths = load_image_paths(dataset_path, path_prefix=image_path)
    image_class_labels = load_image_labels(dataset_path)
    
    # Load in the train / test split
    train_images, test_images = load_train_test_split(dataset_path)
    
    # Visualize the images and their annotations
    image_identifiers = image_paths.keys()
    random.shuffle(image_identifiers) 
    
    count = 0

    for image_id in train_images:
        image_path = image_paths[image_id]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        im_resized = np.asarray(cv2.resize(image, IM_ORIG_SIZE), dtype = np.float64)
        # x = randint(0,28)
        # y = randint(0,28)
        # im_cropped = im_resized[x:x+IM_SIZE[0], y:y+IM_SIZE[1]]
        im_cropped = im_resized
        im_cropped = im_cropped / 255
        if (count == 0):
            im_mean = im_cropped
        else:    
            im_mean = (count * im_mean + im_cropped)/(count+1)
        count += 1
    return im_mean    


def load_image_batch(im_mean, start_idx, batch_size, image_path, im_size, dataset_path = '/raid/mainak/datasets/nabirds/', is_train=1):
    
    image_paths = load_image_paths(dataset_path, path_prefix=image_path)
    image_class_labels = load_image_labels(dataset_path)
    
    # Load in the train / test split
    train_images, test_images = load_train_test_split(dataset_path)
    
    # Visualize the images and their annotations
    image_identifiers = image_paths.keys()
    random.shuffle(image_identifiers) 
    if is_train ==1:
        image_set = train_images
    else:
        image_set = test_images
            
    if start_idx == 0:
        random.shuffle(image_set) 
    batch_images = image_set[start_idx:start_idx+batch_size]
    
    count = 0
    out_images = []
    image_labels = []

    for image_id in batch_images:
	  
        image_path = image_paths[image_id]
        # print image_path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        class_label = image_class_labels[image_id]
        #if (class_label > 21):
        #    print "Error!: ", class_label
        
        count += 1
        processed_image = preprocessImage(image, im_mean, im_size, is_train)
        out_images.append(processed_image)
        image_labels.append(class_label)
        
	#out_images_arr = np.transpose(np.asarray(out_images), (0,3,1,2)) # Make it n_images x n_ch x n_row x n_col
	out_images_arr = np.asarray(out_images) # Make it n_images x n_row x n_col x n_ch
	image_labels_arr = np.asarray(image_labels)
	
	batch_x = out_images_arr
	batch_y = image_labels_arr

    return batch_x, batch_y

if __name__ == '__main__':
  
    #for i in xrange(10):
    g = load_image_batch(im_mean=0, start_idx=0, batch_size=1, image_path='images', is_train=1)
    g = load_image_batch(im_mean=0, start_idx=1, batch_size=1, image_path='images', is_train=1)
    g = load_image_batch(im_mean=0, start_idx=0, batch_size=1, image_path='images', is_train=1)
    g = load_image_batch(im_mean=0, start_idx=1, batch_size=1, image_path='images', is_train=1)
    # if len(sys.argv) > 1:
    # 	dataset_path = sys.argv[1]
    # else:
    # 	dataset_path = '/raid/mainak/datasets/nabirds/'
    # 
    # if len(sys.argv) > 2:
    # 	image_path = sys.argv[2]
    # else:
    # 	image_path  = 'images'

    # print "Starting Data Extraction from Datapath: ", dataset_path
    # 
    # # Load in the image data
    # # Assumes that the images have been extracted into a directory called "images"
    # image_paths = load_image_paths(dataset_path, path_prefix=image_path)
    # image_sizes = load_image_sizes(dataset_path)
    # image_bboxes = load_bounding_box_annotations(dataset_path)
    # image_parts = load_part_annotations(dataset_path)
    # image_class_labels = load_image_labels(dataset_path)
    # 
    # # Load in the class data
    # class_names = load_class_names(dataset_path)
    # class_hierarchy = load_hierarchy(dataset_path)
    # 
    # # Load in the part data
    # part_names = load_part_names(dataset_path)
    # part_ids = part_names.keys()
    # part_ids.sort() 
    # 
    # # Load in the photographers
    # photographers = load_photographers(dataset_path)
    # 
    # # Load in the train / test split
    # train_images, test_images = load_train_test_split(dataset_path)
    # 
    # # Visualize the images and their annotations
    # image_identifiers = image_paths.keys()
    # random.shuffle(image_identifiers) 
    # 
    # count = 0
    # out_images = []
    # image_labels = []

    # print "Start storing the images in pickle file"
    # 
    # for image_id in image_identifiers:
	#   
    #     image_path = image_paths[image_id]
    #     #print dataset_path, image_path
    #     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #     bbox = image_bboxes[image_id]
    #     parts = image_parts[image_id]
    #     class_label = image_class_labels[image_id]
    #     class_name = class_names[class_label]
    #     
    #     count += 1
    #     processed_image = preprocessImage(image, is_train=0)
    #     out_images.append(processed_image)
    #     image_labels.append(class_label)
    #     
    #     if np.mod(count, 10) == 0:
    #         print "Count: ", count
    #     #if count > 10000:
    #     #    break

	# #out_images_arr = np.transpose(np.asarray(out_images), (0,3,1,2)) # Make it n_images x n_ch x n_row x n_col
	# out_images_arr = np.asarray(out_images) # Make it n_images x n_row x n_col x n_ch
	# image_labels_arr = np.asarray(image_labels)
	# 
	# #train_set_x = theano.shared(out_images_arr)
	# #train_set_y = theano.shared(image_labels_arr)
	# train_set_x = out_images_arr
	# train_set_y = image_labels_arr
	# f = file(dataset_path + 'birds_train_data.pkl', 'wb')
	# pkl.dump((train_set_x, train_set_y), f, protocol=pkl.HIGHEST_PROTOCOL)
	# f.close()
