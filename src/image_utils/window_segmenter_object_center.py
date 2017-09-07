#!/usr/bin/env python
import rospy
from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import cv2

def main():

    window_scaling = 2.0
    object_class = "bowl"
    object_instance = "1"

    text_file_path = "/home/scatha/research_ws/src/lifelong_object_learning/data/captured/" + object_class + "/" + object_class + "_" + object_instance + "/metadata/"
    image_file_path = "/home/scatha/research_ws/src/lifelong_object_learning/data/captured/" + object_class + "/" + object_class + "_" + object_instance + "/images/"
    save_text_file_path = "/home/scatha/research_ws/src/lifelong_object_learning/data/captured/bowl/images/"
    save_image_path = "/home/scatha/research_ws/src/lifelong_object_learning/data/captured/bowl/images/"
    desired_img_size = 299

    index = -1

    # grab all files in directory
    files = [f for f in listdir(text_file_path) if f.endswith(".txt")]
    for file in files:

        index += 1

    	file_base = splitext(file)[0]
    	image_file_name = file_base + ".png"
    	# print image_file_name

        lines = [line.rstrip('\n') for line in open(text_file_path + file)]
        # image_file = lines[0]
        image_file = image_file_path + image_file_name
        image_size = lines[1].split()
        height = int(image_size[0])
        width = int(image_size[1])

        # ar_tag_points = []
        # for line in lines[2:6]:
        #     vals = line.split()
        #     ar_tag_points.append((int(vals[1]), int(vals[0])))

        object_points = []
        for line in lines[2:6]:
            vals = line.split()
            object_points.append((int(vals[1]), int(vals[0])))


        # x_pose = float(lines[10])
        # y_pose = float(lines[11])

        x_pose = index
        y_pose = index


        # print object_points
        object_center_x = 0.0
        object_center_y = 0.0
        object_min_x = float("inf")
        object_max_x = -float("inf")
        object_min_y = float("inf")
        object_max_y = -float("inf")
        for point in object_points:
            x = point[1]
            y = point[0]
            object_center_x += x
            object_center_y += y

            if x < object_min_x:
                object_min_x = x

            if x > object_max_x:
                object_max_x = x

            if y < object_min_y:
                object_min_y = y

            if y > object_max_y:
                object_max_y = y

        object_center_x = int(object_center_x/len(object_points))
        object_center_y = int(object_center_y/len(object_points))


        image = cv2.imread(image_file)
        # print image.shape

        window_size = min((object_max_x - object_min_x), (object_max_y - object_min_y))
        # tl = (object_center_x - window_size/2, object_center_y - window_size/2)
        # tr = (object_center_x + window_size/2, object_center_y - window_size/2)
        # bl = (object_center_x - window_size/2, object_center_y + window_size/2)
        # br = (object_center_x + window_size/2, object_center_y + window_size/2)
        max_window_size = window_scaling*max((object_max_x - object_min_x), (object_max_y - object_min_y))

        left_bound = object_center_x - window_size/2
        right_bound = object_center_x + window_size/2
        top_bound = object_center_y - window_size/2
        bottom_bound = object_center_y + window_size/2

        while(True):
            # new_tl = (tl[0]-1, tl[1]-1)
            # new_tr = (tr[0]+1, tl[1]-1)
            # new_bl = (bl[0]-1, tl[1]+1)
            # new_br = (br[0]+1, tl[1]+1)


            # if ar_tag appears in image throw out
            # ar_tag_points_in_img = 0
            # for point in ar_tag_points:
            #     if ((point[0] >= top_bound-1) and (point[0] <= bottom_bound+1)) and ((point[1] >= left_bound-1) and (point[1] <= right_bound+1)):
            #         ar_tag_points_in_img += 1

            # if ar_tag_points_in_img > 0:
            #     break


            # bounds check
            if top_bound-1 <= 0:
                break
            if bottom_bound+1 >= height:
                break
            if left_bound-1 <= 0:
                break
            if right_bound+1 >= width:
                break

            # stop if max window size is reached
            if (right_bound - left_bound) >= max_window_size:
                break
            if (bottom_bound - top_bound) >= max_window_size:
                break

            left_bound -= 1
            right_bound += 1
            top_bound -= 1
            bottom_bound += 1

        sub_image = image[top_bound:bottom_bound, left_bound:right_bound]

        # resized_image = cv2.resize(sub_image, (desired_img_size, desired_img_size)) 




        image_filename = save_image_path + file_base +".png"
        # text_filename = save_text_file_path + file_base +".txt"

        # f = open(text_filename, 'w')     # FIXME: integration
        # f.write(image_filename + "\t" + str(x_pose) + "\t" + str(y_pose) + "\n")
        # f.close()

        cv2.imwrite(image_filename, sub_image)


if __name__ == "__main__":
    main()