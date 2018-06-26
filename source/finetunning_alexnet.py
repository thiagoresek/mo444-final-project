import os
import argparse
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.keras._impl.keras.applications.vgg16 import VGG16
from tensorflow.python.keras._impl.keras.layers import Dense
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import numpy as np
import pickle
from alexnet import AlexNet



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
    parser.add_argument("--positive_data", help="A number from 0 to 3 that defines which will be the positive class")
    parser.add_argument("--fold", help="path to the dataset")
    parser.add_argument("--version", help="version of the classifier")
    parser.add_argument("--mode", help="Train or Test")
    parser.add_argument("--batch_size", default=40, help="number of images in each iteration times 4")
    parser.add_argument('--model_path', default='vgg_16.ckpt', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
    parser.add_argument('--num_epochs', default=10, type=int)

    args = parser.parse_args()


    positive_class = int(args.positive_data)
    fold = args.fold
    mode = args.mode
    version = args.version
    batch_size = int(args.batch_size)
    model_path = args.model_path
    num_workers = args.num_workers
    weight_decay = args.weight_decay
    dropout_keep_prob = args.dropout_keep_prob
    num_epochs = args.num_epochs

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


    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    #graph = tf.Graph()
    #with graph.as_default():


    # Standard preprocessing for VGG on ImageNet taken from here:
    # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
    # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

    

    # ----------------------------------------------------------------------
    # DATASET CREATION using tf.contrib.data.Dataset
    # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

    # The tf.contrib.data.Dataset framework uses queues in the background to feed in
    # data to the model.
    # We initialize the dataset with a list of filenames and labels, and then apply
    # the preprocessing functions described above.
    # Behind the scenes, queues will load the filenames, preprocess them with multiple
    # threads and apply the preprocessing in parallel, and then batch the data

    # Training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_parse_function, num_parallel_calls=args.num_workers)
    train_dataset = train_dataset.map(training_preprocess, num_parallel_calls=args.num_workers)
    train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    batched_train_dataset = train_dataset.batch(int(args.batch_size))

    # Validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(_parse_function, num_parallel_calls=args.num_workers)
    val_dataset = val_dataset.map(val_preprocess, num_parallel_calls=args.num_workers)
    batched_val_dataset = val_dataset.batch(int(args.batch_size))



    # Now we define an iterator that can operator on either dataset.
    # The iterator can be reinitialized by calling:
    #     - sess.run(train_init_op) for 1 epoch on the training set
    #     - sess.run(val_init_op)   for 1 epoch on the valiation set
    # Once this is done, we don't need to feed any value for images and labels
    # as they are automatically pulled out from the iterator queues.

    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of either `train_dataset`
    # or `validation_dataset` here, because they are compatible.
    iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types, batched_train_dataset.output_shapes)
    images, labels = iterator.get_next()



    train_init_op = iterator.make_initializer(batched_train_dataset)
    val_init_op = iterator.make_initializer(batched_val_dataset)

    # Indicates whether we are in training or in test mode
    #is_training = tf.placeholder(tf.bool)
   
   

    # ---------------------------------------------------------------------
    # Now that we have set up the data, it's time to set up the model.
    # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
    # last fully connected layer (fc8) and replace it with our own, with an
    # output size num_classes=4
    # We will first train the last layer for a few epochs.
    # Then we will train the entire model on our dataset for a few epochs.

    # Get the pretrained model, specifying the num_classes argument to create a new
    # fully connected replacing the last one, called "vgg_16/fc8"
    # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
    # Here, logits gives us directly the predicted scores we wanted from the images.
    # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
    
    train_layers = ['fc8']
    #num_classes = 2
    keep_prob = 1.0
    model = AlexNet(images, keep_prob, num_classes, train_layers)
    logits = model.fc8

    print(logits)
    exit()
    # Specify where the model checkpoint is (pretrained weights).
    #model_path = args.model_path
    #assert(os.path.isfile(model_path))

    # ---------------------------------------------------------------------
    # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
    # We can then call the total loss easily
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #loss = tf.losses.get_total_loss()

    # Then we want to finetune the entire model for a few epochs.
    # We run minimize the loss only with respect to all the variables.

    starter_learning_rate = 0.001
    global_step = tf.Variable(0, trainable=False, name='gs')
    gs = tf.contrib.framework.get_variables('gs')
    gs_init = tf.variables_initializer(gs)
    sess.run(gs_init)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 500, 0.9)
    #full_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    full_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    full_train_op = full_optimizer.minimize(loss, global_step=global_step)

    # Evaluation metrics
    prediction = tf.to_int32(tf.argmax(logits, 1))
    correct_prediction = tf.equal(prediction, labels)
    conf_matrix = tf.confusion_matrix(labels, prediction, num_classes=num_classes)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #tf.get_default_graph().finalize()
    saver = tf.train.Saver()


    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    
    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)

    best_acc = 0.0
    plot_acc_val = []
    plot_acc_train = []


     # Train the entire model for a few more epochs, continuing with the *same* weights.
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        sess.run(train_init_op)
        while True:
            try:
                _ = sess.run(full_train_op)
            except tf.errors.OutOfRangeError:
                break

        # Check accuracy on the train and val sets every epoch
        train_acc, train_matrix = check_accuracy(sess, conf_matrix, train_init_op)
        val_acc, val_matrix = check_accuracy(sess, conf_matrix, val_init_op)


        plot_acc_val.append(val_acc)
        plot_acc_train.append(train_acc)

        if(val_acc > best_acc):
            best_acc = val_acc
            saver.save(sess, '/work/gbertocco/TCC/model04/ER_' + version)
            info_validation = [val_acc, val_matrix]
            pickle.dump(info_validation, open("INFO_VAL_" + version, "wb"))

        pickle.dump(plot_acc_train, open("plot_val_" + version, "wb"))
        pickle.dump(plot_acc_val, open("plot_train_" + version, "wb"))    

        print('Train normalized accuracy: %f' % train_acc)
        print('Val normalized accuracy: %f\n' % val_acc)
        print('Best (Val) normalized accuracy so far: %f\n' % best_acc)




	#net = VGG16()
	#layer = net.get_layer(name='fc2')

	#print(layer)

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



def Read(directory):
    
    filenames = os.listdir(directory)
    img_names = []
    for img in filenames:
        if img.endswith('.jpg'):
            img_names.append(directory + img)
    return img_names

	
def list_images(directory):
    
    filenames = os.listdir(directory)

    #print(len(filenames))
    labels = []
    img_names = []
    for img in filenames:

        if img.endswith('.jpg'):
            #print(img[:2])
            labels.append(int(img[:2]))
            img_names.append("./" + directory + "/" + img)

    return img_names, labels


def check_accuracy(sess, conf_matrix, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    matrix = np.zeros((4,4))
    while True:
        try:
            matrix += sess.run(conf_matrix)
            #num_correct += correct_pred.sum()
            #num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    print(matrix)
    acc = 0.0
    for ind in range(4):
        acc += matrix[ind][ind]/sum(matrix[ind, :])

    # Return the fraction of datapoints that were correctly classified
    #acc = float(num_correct) / num_samples
    normalized_acc = acc/4
    return normalized_acc, matrix

# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def _parse_function(filename, label):
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
    return resized_image, label


# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def training_preprocess(image, label):
    crop_image = tf.random_crop(image, [227, 227, 3])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)                # (4)
    flip_image = crop_image

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = flip_image - means                                     # (5)

    return centered_image, label

# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess(image, label):
    crop_image = tf.image.resize_image_with_crop_or_pad(image, 227, 227)    # (3)

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = crop_image - means                                     # (4)

    return centered_image, label

def printOperationsName():
    for i in tf.get_default_graph().get_operations():
            print (i.name)



if __name__ == '__main__':
    main()