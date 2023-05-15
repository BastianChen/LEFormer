import random
import cv2
import os
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Split dataset into train and test sets')
    parser.add_argument('--dataset_type', type=str, required=True, choices=['sw', 'qtpl'], help='type of the dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='path of the dataset')
    parser.add_argument('--save_path', type=str, required=True, help='path to save the train and test sets')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = args.dataset_path
    save_path = args.save_path

    if args.dataset_type == 'sw':
        num_images_to_extract = 3519
        total_images_length = 17596
        dataset_img_path = dataset_label_path = dataset_path
    elif args.dataset_type == 'qtpl':
        num_images_to_extract = 610
        total_images_length = 6773
        dataset_img_path = os.path.join(dataset_path, "train_img")
        dataset_label_path = os.path.join(dataset_path, "train_label")
    else:
        raise ValueError(
            '--dataset_type must be sw or qtpl.')

    random_set = set()
    while len(random_set) < num_images_to_extract:
        random_set.add(random.randint(0, total_images_length - 1))

    num_training = 0
    num_validation = 0
    for index in range(total_images_length):
        if index == 5:
            break
        image_name = f"{dataset_img_path}/{index}.jpg"
        image = cv2.imread(image_name)
        if args.dataset_type == 'sw':
            label_name = f"{dataset_label_path}/{index}_vis.png"
            gt_name = f"{dataset_path}/{index}.png"
            label = cv2.imread(label_name, 0)
            gt = cv2.imread(gt_name, 0)
        else:
            label_name = f"{dataset_label_path}/{index}.png"
            label = cv2.imread(label_name, 0)
            gt = np.where(label == 38, 1, 0)

        if index in random_set:
            cv2.imwrite(f"{save_path}/images/validation/val_{num_validation}.jpg", image)
            cv2.imwrite(f"{save_path}/annotations/validation/val_{num_validation}.png", label)
            cv2.imwrite(f"{save_path}/gt/validation/val_{num_validation}.png", gt)
            num_validation += 1
        else:
            cv2.imwrite(f"{save_path}/images/training/training_{num_training}.jpg", image)
            cv2.imwrite(f"{save_path}/annotations/training/training_{num_training}.png", label)
            cv2.imwrite(f"{save_path}/gt/training/training_{num_training}.png", gt)
            num_training += 1


if __name__ == '__main__':
    main()
