from scipy.stats import wasserstein_distance
from sklearn.preprocessing import scale
from math import cos, sin
import os
# from PIL import Image
import cv2
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mutual_info_score
# generate distinct pairs without permutations (n*(n-1)/2 pairs with n dataset size)


def make_pairs(n):
    pairs_nb = int(n*(n-1)/2)
    print("generating {} pairs from {} elements".format(pairs_nb, n))
    bar = tqdm(pairs_nb)
    pairs = [0]*pairs_nb
    ind = 0
    for i in range(n):
        for j in range(i+1, n):
            pairs[ind] = [i, j]
            ind += 1
            bar.update()
    bar.close()
    return pairs, pairs_nb


def plot_hist(x, title):
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title + " histogram")
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.waitforbuttonpress()
    plt.close()


def get_cam_pos(camera_pos_file, img_nb):
    # get all camera positions for comparison
    cam_pos = [0]*img_nb
    with open(camera_pos_file, 'r') as csvfile:
        header = csvfile.readline()  # view_no,theta,phi
        line = csvfile.readline()
        line = str(line)
        line = line.split(',', 2)
        line[-1] = line[-1].replace("\n", "")  # remove line separator

        while line != ['']:
            view_no = line[0]
            theta = float(line[1])
            phi = float(line[2])
            x = radius * sin(theta)*cos(phi)
            y = radius * sin(theta) * sin(phi)
            z = radius * cos(theta)
            cam_pos[int(view_no)] = [x, y, z]
            line = csvfile.readline()
            line = str(line)
            line = line.split(',', 2)
            line[-1] = line[-1].replace("\n", "")  # remove line separator
        csvfile.close()
    return cam_pos


def get_ambiguities(folder, pairs, pairs_nb, ambiguities_only=True):
    # Compute all errors, do not forget to normalize them
    img_MI = [0] * pairs_nb
    img_errors = [0] * pairs_nb
    cam_errors = [0] * pairs_nb

    print("computing pairs comparison")
    bar = tqdm(pairs_nb)
    for i in range(pairs_nb):
        img_pair = pairs[i]
        img_1_name = "viewpoint" + str(img_pair[0]) + ".png"
        img_1 = cv2.imread(os.path.join("data", folder, "images", img_1_name))
        img_2_name = "viewpoint" + str(img_pair[1]) + ".png"
        img_2 = cv2.imread(os.path.join("data", folder, "images", img_2_name))

        # img_1_name = os.path.join(
        #     folder, "viewpoint" + str(img_pair[0]) + ".png")
        # img_2_name = os.path.join(
        #     folder, "viewpoint" + str(img_pair[1]) + ".png")
        # img_1 = Image.open(img_1_name)
        # img_1 = list(img_1.getdata())
        # img_2 = Image.open(img_2_name)
        # img_2 = list(img_2.getdata())

        # Test various image error metrics
        # img_errors[i] = mean_squared_error(img_1, img_2)
        img_errors[i] = (np.square(img_1-img_2)).mean()
        # img_errors[i] = wasserstein_distance(img_1, img_2)
        img_MI[i] = mutual_info_score(
            np.reshape(img_1, -1), np.reshape(img_2, -1))

        # Test ends
        camera_error = mean_squared_error(
            cam_pos[pairs[i][0]], cam_pos[pairs[i][1]])
        cam_errors[i] = camera_error
        bar.update()
    bar.close()

    # plot errors before normalization
    plot_hist(img_errors, "image distances")
    plot_hist(cam_errors, "camera distances")

    # normalize all errors between
    print("max img error before normalization: {}".format(max(img_errors)))
    img_errors = scale(img_errors, axis=0, with_mean=True,
                       with_std=True, copy=True)
    print("max cam error before normalization : {}".format(max(cam_errors)))
    cam_errors = scale(cam_errors, axis=0, with_mean=True,
                       with_std=True, copy=True)
    img_MI = scale(img_MI, axis=0, with_mean=True,
                   with_std=True, copy=True)
    # compute ambiguity metrics (if the img_errors contains mutual information instead of errors, compute sum instead of difference)
    ambiguities = cam_errors + (img_MI - img_errors) / 2
    if ambiguities_only:
        return ambiguities
    else:
        return img_errors, img_MI, cam_errors, ambiguities


if __name__ == "__main__":
    dataset_name = "cube_64_R_4_random_angles"
    camera_pos_file = "camera_data_cube_64_R_4_random_angles.csv"
    folder = os.path.join("data", dataset_name, "images")
    radius = 4
    # list image files
    img_files = os.listdir(folder)
    # Generate pairs for proper number of files
    img_nb = img_files.__len__()
    # for 5000 images, compute all pairwise errors takes 18 hours (better compute pairwise errors at training time for batches of size ~100)
    pairs, pairs_nb = make_pairs(100)  # switch back to img_nb
    # generate ambiguity table as empty dictionary of size pairs_nb
    s = set(range(pairs_nb))
    ambiguity_table = dict.fromkeys(s)

    camera_pos_file = os.path.join("data", camera_pos_file)
    cam_pos = get_cam_pos(camera_pos_file, img_nb)

    img_errors, img_MI, cam_errors, ambiguities = get_ambiguities(
        dataset_name, pairs, pairs_nb, ambiguities_only=False)

    # save pairwise errors and ambiguities in file
    file_name = "data/pairwise_ambiguities_{0}.csv".format(dataset_name)
    if not os.path.isfile(file_name):
        os.mknod(file_name)

    with open(file_name, 'w') as csvfile:
        fieldnames = ['img_pair', 'img_mean_square_normalized',
                      'camera_mean_square_normalized', 'ambiguity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(0, pairs_nb):
            writer.writerow(
                {'img_pair': pairs[i], 'img_mean_square_normalized': img_errors[i],
                 'camera_mean_square_normalized': cam_errors[i], 'ambiguity': ambiguities[i]})

    plot_hist(img_errors, "image distances normalized")
    plot_hist(cam_errors, "camera distances normalized")
    plot_hist(ambiguities, "ambiguities")
