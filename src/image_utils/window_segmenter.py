#!/usr/bin/env python
import rospy
from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import cv2

def main():

    # text_file_path = rospy.get_param('text_file_path')
    # window_size = rospy.get_param('window_size')
    # stride = rospy.get_param('stride')
    # images_to_save_per_file = rospy.get_param('images_to_save_per_file')
    # save_image_path = rospy.get_param('save_image_path')
    # save_labels_path = rospy.get_param('save_labels_path')
    text_file_path = "/home/morgul/data_gatherer/src/data_collector/data/train_collected/metadata/"
    image_file_path = "/home/morgul/data_gatherer/src/data_collector/data/train_collected/images/"
    # text_file_path = "/home/christopher/code/data/collected/metadata/"
    window_size = 64
    stride = 5
    images_to_save_per_file = 20
    save_image_path = "/home/morgul/data_gatherer/src/data_collector/data/train/"
    save_labels_file = "/home/morgul/data_gatherer/src/data_collector/data/train/label.txt"
    # save_image_path = "/home/christopher/code/data/train/"
    # save_labels_file = "/home/christopher/code/data/train/label.txt"
    base_name = "training_"

    pos_images = []
    neg_images = []
    pos_poses = []
    neg_poses = []

    index = 0

    # grab all files in directory
    files = [f for f in listdir(text_file_path) if f.endswith(".txt")]
    for file in files:

    	file_base = splitext(file)[0]
    	image_file_name = file_base + ".png"
    	# print image_file_name

        file_pos_images = []
        file_neg_images = []

        lines = [line.rstrip('\n') for line in open(text_file_path + file)]
        # image_file = lines[0]
        image_file = image_file_path + image_file_name
        image_size = lines[1].split()
        height = int(image_size[0])
        width = int(image_size[1])

        ar_tag_points = []
        for line in lines[2:6]:
            vals = line.split()
            ar_tag_points.append((int(vals[1]), int(vals[0])))

        object_points = []
        for line in lines[6:10]:
            vals = line.split()
            object_points.append((int(vals[1]), int(vals[0])))
        # print object_points
        num_object_points = len(object_points)

        # x_pose = float(lines[10])
        # y_pose = float(lines[11])

        x_pose = index
        y_pose = index
        index += 1

        image = cv2.imread(image_file)
        # print image.shape

        # construct sub images
        left_bound = 0
        right_bound = window_size
        top_bound = 0
        bottom_bound = window_size

        while(True):

            sub_image = image[top_bound:bottom_bound, left_bound:right_bound]
            # print sub_image.shape

            # if ar_tag appears in image throw out
            ar_tag_points_in_img = 0
            for point in ar_tag_points:
                if ((point[0] >= top_bound) and (point[0] <= bottom_bound)) and ((point[1] >= left_bound) and (point[1] <= right_bound)):
                    ar_tag_points_in_img += 1
            if ar_tag_points_in_img > 0:
                top_bound += stride
                bottom_bound += stride

                if bottom_bound >= height:
                    top_bound = 0
                    bottom_bound = window_size
                    left_bound += stride
                    right_bound += stride

                    if right_bound >= width:
                        # print right_bound
                        break

                continue

            object_points_in_img = 0
            for point in object_points:
                if ((point[0] >= top_bound) and (point[0] <= bottom_bound)) and ((point[1] >= left_bound) and (point[1] <= right_bound)):
                    object_points_in_img += 1

            # if object is not image add to neg examples
            if object_points_in_img == 0:
                file_neg_images.append(sub_image)

            # if full object appears in image add to pos examples
            if object_points_in_img == num_object_points:
                file_pos_images.append(sub_image)

            # # if at least part of object appears in image add to pos examples
            # if object_points_in_img > 0:
            #   file_pos_images.append(sub_image)


            top_bound += stride
            bottom_bound += stride

            if bottom_bound >= height:
                top_bound = 0
                bottom_bound = window_size
                left_bound += stride
                right_bound += stride

                if right_bound >= width:
                    # print right_bound
                    break

        # keep subset of images per file
        file_pos_images = np.array(file_pos_images)
        file_neg_images = np.array(file_neg_images)
        print file_pos_images.shape
        print file_neg_images.shape

        np.random.shuffle(file_pos_images)
        np.random.shuffle(file_neg_images)

        file_pos_images = list(file_pos_images)
        file_neg_images = list(file_neg_images)

        num_to_save = min(images_to_save_per_file, len(file_pos_images))
        num_to_save = min(num_to_save, len(file_neg_images))

        pos_images += file_pos_images[0:num_to_save]
        neg_images += file_neg_images[0:num_to_save]

        # poses
        pose_list = [(x_pose, y_pose) for i in range(num_to_save)]
        if num_to_save > 0:
	        pos_poses += pose_list
	        neg_poses += pose_list


    # write images to file
    base_file = save_image_path + base_name
    image_index = 0
    f = open(save_labels_file, 'w')     # FIXME: integration

    for i in range(len(pos_images)):
    	image = pos_images[i]
    	pose = pos_poses[i]
        image_file = base_file + str(image_index) +".png"
        # f.write(image_file + "\t" + str(0) + "\n")
        f.write(image_file + "\t" + str(0) + "\t" + str(pose[0]) + "\t" + str(pose[1]) + "\n")
        cv2.imwrite(image_file, image)
        image_index += 1

    for i in range(len(neg_images)):
    	image = neg_images[i]
    	pose = neg_poses[i]
        image_file = base_file + str(image_index) +".png"
        # f.write(image_file + "\t" + str(1) + "\n")
        f.write(image_file + "\t" + str(1) + "\t" + str(pose[0]) + "\t" + str(pose[1]) + "\n")
        cv2.imwrite(image_file, image)
        image_index += 1
    f.close()



if __name__ == "__main__":
    main()