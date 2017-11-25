import csv
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
from image_helpers import transform_img

image_rel_path = './data/IMG/'
aug_img_dir = 'IMG_aug'
aug_image_rel_path = './data/IMG_aug/'

def read_image(image_path):
    """
    :param image_path:
    :return img
    """
        # Original image shape: 160x320x3
    img = cv2.imread(image_path)
    # Convert the image to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def parse_driving_log():
    """
    Walks through ./data/driving_log.csv and gets the image paths (relative to cwd) and steering angles
    :return: [], [] - list of image paths, driving angles
    """
    csv_path = './data/driving_log.csv'
    samples = []
    angles = []
    with open(csv_path) as f:
        logs = csv.reader(f, delimiter=',')
        for log in logs:
            speed = float(log[6])

            angle_augment = 0
            # Ignore data points where car was not moving
            if speed < 0.1:
                continue

            im_path = log[0]
            filename = im_path.split('/')[-1]
            rel_path = image_rel_path + filename
            angle = float(log[3])
            samples.append(rel_path)
            angles.append(angle)
    return samples, angles


def clean_aug_data():
    filelist = [ f for f in os.listdir(aug_image_rel_path) ]
    for f in filelist:
        os.remove(os.path.join(aug_image_rel_path, f))


def generate_fake_images_like(existing_paths, existing_angles, num_to_generate):
    """
    Randomly sample from from existing_paths array and copy to a new array.
    num_to_generate dictates the number of sample generated

    # Transformation is done by adding guassian blur.
    # moving image randomly in horizontal direction.
    # There should not be any transformation in x direction because later this image will be cropped out

    :param existing_paths:
    :param existing_angles:
    :param num_to_generate:
    :return:
    """
    new_image_paths = []
    new_angles = []

    for i in range(num_to_generate):
        random_index = random.randint(0, len(existing_paths) - 1)
        fpath = existing_paths[random_index]
        img = read_image(fpath)


        img = transform_img(img, 0, 15)

        filename = fpath.split('/')[-1]
        extractPath = filename.split('.')
        augfilename = aug_image_rel_path + extractPath[0] + '_' + str(random.randint(0, 10000)) + '.' + extractPath[1]
        if augfilename not in new_image_paths:
            cv2.imwrite(augfilename, img)
            new_image_paths.append(augfilename)
            new_angles.append(existing_angles[random_index])

    return new_image_paths, new_angles

def preproces_datasets(np_paths, np_angles):
    """
    Looks at distribution of the data.
    Then computes median/avg of distribution. Due to the nature of the track we found lot of images where there is no
    angle and few samples for extreme steering angle.
    In order to avoid overfitting to zero angle, we have to remove few images for zero angle.

    At same time, number of samples where |steering angle| > |0.5| is less. We need more of such images because we have
    to tech car to recover in case it is approaching curb in some situations. So here I am faking images for angles with
    very few images.

    :param np_paths:
    :param np_angles:
    :return:
    """

    clean_aug_data()

    # Step 1: Plot distribution
    min_angle = np.min(np_angles)
    max_angle = np.max(np_angles)
    bin_width = 0.1
    num_bins = int(math.floor(((max_angle - min_angle) / bin_width)))

    hist, bins = np.histogram(np_angles, num_bins)
    # Numpy returns 1 more bin than histogram array.
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.bar(bin_centers, hist, align='center', width=bin_width)
    plt.savefig('plot/training_angles_hist.png')
    plt.clf()

    # Figure out a ideal range of counts for each bin
    expected_average_per_bin = int(len(np_angles)/num_bins)

    # I came up with these multipliers based on experience from traffic classifier project
    max_per_bin = 3.5 * expected_average_per_bin
    min_per_bin = int(0.5 * expected_average_per_bin)

    print("Expected expected_average_per_bin {}, max_per_bin {}, min_per_bin {}".format(expected_average_per_bin, max_per_bin, min_per_bin))

    # Step 2:
    # Go through each training sample and categorize into bins
    # Then for each bin,
    #   - if the number of images is more than max_per_bin, then randomly
    #   remove some images.
    #   - If the number of images is less than max_per_bin, then generate fake data

    # Get training angles for each bin in histogram
    angle_indices_by_bin = {j: [] for j in range(num_bins)}

    def get_bin_from_angle(angle):
        for i in range(num_bins):
            if angle == bins[i]:
                return i
            if angle > bins[i] and angle <= bins[i+1]:
                return i

    for i in range(len(np_angles)):
        bin_num = get_bin_from_angle(np_angles[i])
        angle_indices_by_bin[bin_num].append(i)


    def delete_random(indices, keep_rate=50):
        return [i for i in range(len(indices)) if random.randint(0, 100) > keep_rate]

    indices_to_delete = []
    aug_image_paths = []
    aug_angles = []

    for bin_num in angle_indices_by_bin.keys():
        if (hist[bin_num] > max_per_bin):
            pass
        #     exceed_ratio = hist[bin_num]/max_per_bin
        #     keep_rate = int(100.0/exceed_ratio) # Larger the bin size, lower probability to keep it.
        #     deleted_bin_indices = delete_random(
        #         angle_indices_by_bin[bin_num],
        #         keep_rate=keep_rate
        #     )
        #     indices_to_delete.extend(deleted_bin_indices)
        #     hist[bin_num] -= len(deleted_bin_indices)

        # We need to generate some fake data
        elif hist[bin_num] < min_per_bin and hist[bin_num] > 0.1 * min_per_bin:
            num_images_to_generate = min_per_bin - hist[bin_num]
            # print("bin_num {}. num_images_to_generate {}".format(bin_num, num_images_to_generate))
            existing_indices = angle_indices_by_bin[bin_num]
            existing_paths = [np_paths[i] for i in existing_indices]
            existing_angles = [np_angles[i] for i in existing_indices]
            added_paths, added_angles = generate_fake_images_like(existing_paths, existing_angles, num_images_to_generate)
            aug_image_paths.extend(added_paths)
            aug_angles.extend(added_angles)
            hist[bin_num] += num_images_to_generate


    np_paths = np.delete(np_paths, indices_to_delete, axis=0)
    np_angles = np.delete(np_angles, indices_to_delete)

    aug_image_paths = np.array(aug_image_paths)
    aug_angles = np.array(aug_angles)

    np_paths = np.concatenate((np_paths, aug_image_paths))
    np_angles = np.concatenate((np_angles, aug_angles))

    print("aug_image_paths {}, deleted_paths {}, total images {}".format(len(aug_image_paths), len(indices_to_delete), len(np_paths)))
    # print("new histogram {}".format(hist))

    plt.bar(bin_centers, hist, align='center', width=bin_width)
    plt.savefig('plot/training_angles_hist_even1.png')
    plt.clf()

    return np_paths, np_angles
