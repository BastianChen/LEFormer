import random
import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm


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
    if args.dataset_type == 'sw':
        for index in tqdm(range(total_images_length)):
            image_name = f"{dataset_img_path}/{index}.jpg"
            image = cv2.imread(image_name)
            label_name = f"{dataset_label_path}/{index}_vis.png"
            binary_label_name = f"{dataset_path}/{index}.png"
            label = cv2.imread(label_name, 0)
            binary_label = cv2.imread(binary_label_name, 0)

            if index in random_set:
                images_validation_path = f"{save_path}/images/validation"
                annotations_validation_path = f"{save_path}/annotations/validation"
                binary_annotations_validation_path = f"{save_path}/binary_annotations/validation"
                for item in [images_validation_path, annotations_validation_path, binary_annotations_validation_path]:
                    if not os.path.exists(item):
                        os.makedirs(item, exist_ok=True)
                cv2.imwrite(f"{images_validation_path}/val_{num_validation}.jpg", image)
                cv2.imwrite(f"{annotations_validation_path}/val_{num_validation}.png", label)
                cv2.imwrite(f"{binary_annotations_validation_path}/val_{num_validation}.png", binary_label)
                num_validation += 1
            else:
                images_training_path = f"{save_path}/images/training"
                annotations_training_path = f"{save_path}/annotations/training"
                binary_annotations_training_path = f"{save_path}/binary_annotations/training"
                for item in [images_training_path, annotations_training_path, binary_annotations_training_path]:
                    if not os.path.exists(item):
                        os.makedirs(item, exist_ok=True)
                cv2.imwrite(f"{images_training_path}/training_{num_training}.jpg", image)
                cv2.imwrite(f"{annotations_training_path}/training_{num_training}.png", label)
                cv2.imwrite(f"{binary_annotations_training_path}/training_{num_training}.png", binary_label)
                num_training += 1
    else:
        index = 0
        for file_name in tqdm(os.listdir(dataset_img_path)):
            image_name = f"{dataset_img_path}/{file_name}"
            image = cv2.imread(image_name)
            label_name = f"{dataset_label_path}/{file_name}"
            label = cv2.imread(label_name, 0)
            binary_label = np.where(label == 38, 1, 0)

            if index in random_set:
                images_validation_path = f"{save_path}/images/validation"
                annotations_validation_path = f"{save_path}/annotations/validation"
                binary_annotations_validation_path = f"{save_path}/binary_annotations/validation"
                for item in [images_validation_path, annotations_validation_path, binary_annotations_validation_path]:
                    if not os.path.exists(item):
                        os.makedirs(item, exist_ok=True)
                cv2.imwrite(f"{images_validation_path}/val_{num_validation}.jpg", image)
                cv2.imwrite(f"{annotations_validation_path}/val_{num_validation}.png", label)
                cv2.imwrite(f"{binary_annotations_validation_path}/val_{num_validation}.png", binary_label)
                num_validation += 1
            else:
                images_training_path = f"{save_path}/images/training"
                annotations_training_path = f"{save_path}/annotations/training"
                binary_annotations_training_path = f"{save_path}/binary_annotations/training"
                for item in [images_training_path, annotations_training_path, binary_annotations_training_path]:
                    if not os.path.exists(item):
                        os.makedirs(item, exist_ok=True)
                cv2.imwrite(f"{images_training_path}/training_{num_training}.jpg", image)
                cv2.imwrite(f"{annotations_training_path}/training_{num_training}.png", label)
                cv2.imwrite(f"{binary_annotations_training_path}/training_{num_training}.png", binary_label)
                num_training += 1
            index += 1


if __name__ == '__main__':
    main()
