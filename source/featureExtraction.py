import os
import argparse
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow.contrib.slim as slim
from models.research.slim.nets import inception_v3
import tensorflow.contrib.slim.nets
import numpy as np
import pandas as pd
import pickle



tf.logging.set_verbosity(tf.logging.INFO)
# restrict visible GPUs to GPU 0 and GPU 3
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# configure on demand memory usage
#config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# check visible CPUs / GPUs
print(device_lib.list_local_devices())
VGG_MEAN = [123.68, 116.78, 103.94]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", help="path to the dataset")
    parser.add_argument("--version", help="version of the classifier")
    parser.add_argument("--mode", help="Train or Test")
    
   
    args = parser.parse_args()
    fold = args.fold
    mode = args.mode
    version = args.version
   

    if(mode == 'Train' or mode == 'Val'):
        # Read and processs the training images
        print("Loading training set")
        '''
        data_path_pos_train, data_path_neg_train, data_path_pos_val, data_path_neg_val = DatasetSelection(fold, positive_class)

        train_filenames = data_path_pos_train + data_path_neg_train
        train_labels = [0]*len(data_path_pos_train) + [1]*len(data_path_neg_train)

        val_filenames = data_path_pos_val + data_path_neg_val
        val_labels = [0]*len(data_path_pos_val) + [1]*len(data_path_neg_val)

        num_classes = len(set(train_labels))
        '''
        train_filenames, val_filenames, train_labels, val_labels = DatasetSelectionMult(fold)
        num_classes = len(set(train_labels))

        # Verify checkpoint!!
        new_saver = tf.train.import_meta_graph('/work/gbertocco/TCC/model04/ER_Inception_M1_Mult.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('/work/gbertocco/TCC/model04/'))
        img_tensor = tf.get_default_graph().get_tensor_by_name('IteratorGetNext:0')
        prob = tf.get_default_graph().get_tensor_by_name("InceptionV1/Logits/SpatialSqueeze:0")
        is_training = tf.get_default_graph().get_tensor_by_name("Placeholder:0")

        count = 0
        tot = len(train_filenames)

        FVs = []
        for index in range(len(train_filenames)):
            print(count)
            print("==== Extracting training fv of the image number %d of a total of %d ====" % (count, tot))
            filename = train_filenames[index]
            target = train_labels[index]
            input_img = training_preprocess(_parse_function(filename))
            img = sess.run(input_img)
            res = sess.run(prob, {img_tensor: [img], is_training: False})[0]
            #res = sess.run(prob, {img_tensor: [img]})[0]
            FVs.append((res, target))
            count += 1

        pickle.dump(FVs, open("TrainFV_" + version, "wb"))

        FVs = []
        count = 0
        tot = len(val_filenames)
        for index in range(len(val_filenames)):
            print(count)
            print("==== Extracting validating fv of the image number %d of a total of %d ====" % (count, tot))
            filename = val_filenames[index]
            target = val_labels[index]
            input_img = val_preprocess(_parse_function(filename))
            img = sess.run(input_img)
            res = sess.run(prob, {img_tensor: [img], is_training: False})[0]
            #res = sess.run(prob, {img_tensor: [img]})[0]
            FVs.append((res, target))
            count += 1


        pickle.dump(FVs, open("ValFV_" + version, "wb"))

    elif(mode == 'Test'):


        filename = '/home/gbertocco/TCC/mo444-final/event_repurposing/MFC18_ER_Image_Ver1/reference/eventrepurpose/MFC18_Dev2-eventrepurpose-ref.csv'
        prefix = '/home/gbertocco/TCC/mo444-final/event_repurposing/MFC18_ER_Image_Ver1/'

        df = pd.read_csv(filename, delimiter='|')

        img_names = df['ProbeFileName']
        events = df['EventName']
        truth = df['IsTarget']

        test_names = []
        for name in img_names:
            test_names.append(prefix + name)

        labels = []
        for ev in events:
            if(ev == 'austin_marathon'):
                labels.append(0)
            elif(ev == 'boston_marathon'):
                labels.append(1)
            elif(ev == 'occupy_baltimore'):
                labels.append(2)
            elif(ev == 'occupy_portland'):
                labels.append(3)


        # Verify checkpoint!!
        new_saver = tf.train.import_meta_graph('/work/gbertocco/TCC/model04/ER_VGG_M1_Mult.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('/work/gbertocco/TCC/model04/'))
        img_tensor = tf.get_default_graph().get_tensor_by_name('IteratorGetNext:0')
        prob = tf.get_default_graph().get_tensor_by_name("vgg_16/fc8/squeezed:0")
        is_training = tf.get_default_graph().get_tensor_by_name("Placeholder:0")

        FVs = []
        count = 0
        tot = len(test_names)//4
        for index in range(len(test_names)):
            if(truth[index] == 'Y'):
                print("==== Extracting training fv of the image number %d of a total of %d ====" % (count, tot))
                filename = test_names[index]
                target = labels[index]
                input_img = val_preprocess(_parse_function(filename))
                img = sess.run(input_img)
                res = sess.run(prob, {img_tensor: [img], is_training: False})[0]
                #res = sess.run(prob, {img_tensor: [img]})[0]
                FVs.append((res, target))
                count += 1

        pickle.dump(FVs, open("TestFV_" + version, "wb"))
    





def DatasetSelection(fold, positive):


    path0 = fold + 'ER_Training_AustinMarathon_Ver1/images/'
    path1 = fold + 'ER_Training_BostonMarathon_Ver1/images/'
    #path2 = fold + 'ER_Training_OccupyBaltimore_Ver1/images/'
    #path3 = fold + 'ER_Training_OccupyPortland_Ver1/images/'

    if(positive == 0):
        data_path_pos_train = Read(path0)[:160]
        data_path_neg_train   = Read(path1)[:160] #+ Read(path2)[:160] + Read(path3)[:160]
        data_path_pos_val = Read(path0)[160:]
        data_path_neg_val = Read(path1)[160:] #+ Read(path2)[160:] + Read(path3)[160:]
    elif(positive == 1):
        data_path_pos_train = Read(path1)[:160]
        data_path_neg_train   = Read(path0)[:160] #+ Read(path2)[:160] + Read(path3)[:160]
        data_path_pos_val = Read(path1)[160:]
        data_path_neg_val = Read(path0)[160:]#+ Read(path2)[160:] + Read(path3)[160:]
    '''
    elif(positive == 2):
        data_path_pos_train = Read(path2)[:160]
        data_path_neg_train   = Read(path0)[:160] + Read(path1)[:160] + Read(path3)[:160]
        data_path_pos_val = Read(path2)[160:]
        data_path_neg_val = Read(path0)[160:]+ Read(path1)[160:] + Read(path3)[160:]
    elif(positive == 3):
        data_path_pos_train = Read(path3)[:160]
        data_path_neg_train   = Read(path0)[:160] + Read(path1)[:160] + Read(path2)[:160]
        data_path_pos_val = Read(path3)[160:]
        data_path_neg_val = Read(path0)[160:]+ Read(path1)[160:] + Read(path2)[160:]
    '''
    return data_path_pos_train, data_path_neg_train, data_path_pos_val, data_path_neg_val

def DatasetSelectionMult(fold):


    path0 = fold + 'ER_Training_AustinMarathon_Ver1/images/'
    path1 = fold + 'ER_Training_BostonMarathon_Ver1/images/'
    path2 = fold + 'ER_Training_OccupyBaltimore_Ver1/images/'
    path3 = fold + 'ER_Training_OccupyPortland_Ver1/images/'

    
    data_path_train = Read(path0)[:160] + Read(path1)[:160] + Read(path2)[:160] + Read(path3)[:160]
    data_path_val = Read(path0)[160:] + Read(path1)[160:] + Read(path2)[160:] + Read(path3)[160:]  
    
    train_labels = [0]*len(Read(path0)[:160]) + [1]*len(Read(path1)[:160]) + [2]*len(Read(path2)[:160]) + [3]*len(Read(path3)[:160])
    val_labels = [0]*len(Read(path0)[160:]) + [1]*len(Read(path1)[160:]) + [2]*len(Read(path2)[160:]) + [3]*len(Read(path3)[160:])

    return data_path_train, data_path_val, train_labels, val_labels

# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = 256.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image


# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def training_preprocess(image):
    crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)                # (4)
    flip_image = crop_image

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = flip_image - means                                     # (5)

    return centered_image

# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess(image):
    crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = crop_image - means                                     # (4)

    return centered_image

def Read(directory):
    
    filenames = os.listdir(directory)
    img_names = []
    for img in filenames:
        if img.endswith('.jpg'):
            img_names.append(directory + img)
    return img_names

def printOperationsName():
    for i in tf.get_default_graph().get_operations():
            print (i.name)

if __name__ == '__main__':
    main()
