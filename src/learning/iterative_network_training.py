import sets
import random
import numpy as np
import time

num_classes = 6
train_data_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train'
validation_data_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train'
test_data_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/demo/test/'
model_save_path = '/home/scatha/research_ws/src/lifelong_object_learning/model_weights/'
model_save_name = 'inception_v4_flickr_base_weights_b32_e50_tr100_fixed_imgnet_features_update_rgbd_no_early_stopping.h5'
batch_size = 32
img_height = 299
img_width = 299
weights_in_path = '/home/scatha/research_ws/src/lifelong_object_learning/model_weights/'
weights_in = 'inception_v4_flickr_base_weights_b32_e50_tr100_fixed_imgnet_features.h5'
weights_path = weights_in_path + weights_in
captured_images_path = '/home/scatha/research_ws/src/lifelong_object_learning/data/captured_cropped/'

training_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/captured_growing/'
object_classes = ["bowl", "calculator", "cell_phone", "coffee_mug", "notebook", "plate"]
num_instances = 4
num_instance_images = 100
epochs = 50
patience = 0

nb_train_samples = 0
nb_validation_samples = 0

testing_dirs = ['/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/captured-cropped-all',
                '/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/captured-cropped-last-instance',
                '/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/rgbd-dataset-all',
                '/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/rgbd-dataset-last-instance']
test_sizes = [3000,
               600,
             18584,
              3812]



# load model
model = create_model(weights_path=weights_path, num_classes=num_classes, weights='imagenet')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=None, sample_weight_mode=None)


# prepare data augmentation configuration       ## FUTURE: more data augmentation?
train_datagen = ImageDataGenerator(
        # rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator() #rescale=1./255)

test_datagens = [ImageDataGenerator() for i in testing_dirs]




# create list of instances to direct data capture over 
instance_list = []
for object_class in object_classes:
    for instance_num in range(num_instances):
        instance = object_class + "_" + str(instance_num+1)
        instance_list.append(instance)

while len(instance_list) != 0:

    # sequential selection
    instance = instance_list[0]
    instance_list.remove(instance)

    # random selection
    rand_index = random.randint(a, len(instance_list)-1)
    instance = instance_list[rand_index]
    instance_list.remove(instance)

    object_class = instance[:-2]
    for i in object_classes:
        if object_class == object_classes[i]:
            object_class_index = i
    instance_img_dir = captured_images_path + object_class + "/" + instance + "/images/"

    # predictions on data
    instance_filenames = listdir(instance_img_dir)
    instance_imgs = []

    for filename in instance_filenames:
        img = load_img(instance_img_dir + filename, target_size=(299, 299))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x[0])
        instance_imgs.append(x)
    instance_imgs = np.array(instance_imgs)
    reduced_instance_imgs = np.copy.deepcopy(instance_imgs)



    ## while prediction accuracy on instance_imgs is still improving:
    instance_confidences = []
    improving = True
    while improving == True:

        # find most uncertain image
        predicted_vals = model.predict(reduced_instance_imgs, batch_size=1, verbose=0)
        min_confidence = float("inf")
        min_index = None
        for index in range(len(predicted_vals)):
            confidence = predicted_vals[index][object_class_index]
            if confidence < min_confidence:
                min_confidence = confidence
                min_index = index

        instance_filename = instance_filenames[min_index]
        np.delete(reduced_instance_imgs, min_index)
        instance_filenames.remove(instance_filename)

        ## add least confident image to training set
        new_filename = training_dir + object_class + "/" + instance_filename
        img = cv2.imread(instance_img_dir + instance_filename,cv2.IMREAD_UNCHANGED)
        cv2.imwrite(new_filename,img)
        while cv2.imread(new_filename) == None:
            time.sleep(.01)

        nb_train_samples += 1
        nb_validation_samples += 1

        # batch_size=min(batch_size,nb_train_samples)

        # retrain model
        train_generator = train_datagen.flow_from_directory(
            training_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = val_datagen.flow_from_directory(
                training_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical')

        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto'), History()],
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

        # model.save_weights(model_save_path + model_save_name)

        ## test model on instance_imgs
        predicted_vals = model.predict(instance_imgs, batch_size=1, verbose=0)
        instance_vals = predicted_vals[:,object_class_index]
        ave_confidence = np.sum(instance_vals)/float(len(instance_imgs))

        if len(instance_confidences) < (patience+1):
            instance_confidences.append(ave_confidence)
        else:
            for instance_confidence in instance_confidences:
                improving = False
                if ave_confidence > instance_confidence:
                    improving = True
            instance_confidences.remove(instance_confidences[0])
            instance_confidences.append(ave_confidence)


        ### also keep track of stats

        # test set performance after each training instance
        for i in range(len(testing_dirs)):
            test_datagen = test_datagens[i]
            testing_dir = testing_dirs[i]
            test_size = test_sizes[i]

            testing_generator = test_datagen.flow_from_directory(
                testing_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical')

            metrics = model.evaluate_generator(
                testing_generator,
                steps=test_size/batch_size)

        # time taken to train each model

        # num images stored

model.save_weights(model_save_path + model_save_name)







