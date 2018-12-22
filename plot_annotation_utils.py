# -*- coding: utf-8 -*-
"""
Created on Wen Dec 18 2018
@author: Rt-Rakesh

This script is a utility to plot all the bbox in the images along with the labels.
Usage: Should be used by calling in jupyter notbooks.
"""

import os
import pandas as pd
import cv2
from tqdm import tqdm


def plot_rec(coor, img, label=''):
    """
    This Function plots the annoations on the images along with label.
    Args:
    1.coor: --tuple The coornidates of the  bbox, it should have the data in the following format (xmin,ymin,xmax,ymax).
    2.image: --np array The image object containing the image in np.array must be provided.
    3.label: -- str The label for the bbox to be mentioned here.

    Returns:
    The image with the annotaions and label written on the image.
    """
    x1, y1, x2, y2 = coor
    draw_img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(draw_img, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
    return draw_img


def plot_annotation_labelwise(csv_path, annotated_files_out_folder_path, original_images_input_folder_path, first_5_only=False):
    """
    This Function plots the annotations on the images along with label and saves it labelwise.
    Args:
    1.csv path: --str The path to the csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,label).
    2.annotated_files_out_folder_path: --str The path to directory where the new annotated images will be saved.
    3.original_images_input_folder_path: --str The path to images directory.
    4.first_5_only: --Boolean Default: False This parameter by default will allow for plotting of all the annotations.Chenge it to True to plot only 5 images per label.
    Returns:
    Label wise images are plotted with the annotaions and labels and stored in the folder mentioned.
    """
    data_df = pd.read_csv(csv_path)
    lable_list = set(data_df.label)
    for i in tqdm(lable_list, desc='Processing labels for all images.'):
        path = os.path.join(annotated_files_out_folder_path, 'labelwise_annotations', str(i))
        if not os.path.exists(path):
            os.makedirs(path)
        if first_5_only:
            temp_df = data_df.loc[data_df['label'] == i].head()
        else:
            temp_df = data_df.loc[data_df['label'] == i]
        if len(temp_df) > 0:
            for j, t in temp_df.iterrows():
                image_path = os.path.join(original_images_input_folder_path, str(t.path))
                img = cv2.imread(image_path)
                x1 = t.xmin
                y1 = t.ymin
                x2 = t.xmax
                y2 = t.ymax
                label = str(t.label)
                anno_image = plot_rec((x1, y1, x2, y2), img, label)
                cv2.imwrite(os.path.join(path, str(t.path)), anno_image)


def plot_annotation(csv_path, annotated_files_out_folder_path, original_images_input_folder_path):
    """
    This Function plots the annoations on the images along with label.
    Args:
    1.csv path: --str The path to the csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,label).
    2.annotated_files_out_folder_path: --str The path to directory where the new annotated images will be saved.
    3.original_images_input_folder_path: --str The path to images directory.
    4.first_5_only: --Boolean Default: False This parameter by default will allow for plotting of all the annotations.Chenge it to True to plot only 5 images per label.
    Returns:
    Label wise images are plotted with the annotaions and labels and stored in the folder mentioned.
    """
    data_df = pd.read_csv(csv_path)
    image_list = set(data_df.path)
    path = os.path.join(annotated_files_out_folder_path, 'annotated_images')
    if not os.path.exists(path):
        os.makedirs(path)
    for i in tqdm(image_list):
        temp_df = data_df.loc[data_df['path'] == i]
        image_path = os.path.join(original_images_input_folder_path, str(i))
        img = cv2.imread(image_path)
        if len(temp_df) > 0:
            for j, t in temp_df.iterrows():
                x1 = t.xmin
                y1 = t.ymin
                x2 = t.xmax
                y2 = t.ymax
                label = str(t.label)
                img = plot_rec((x1, y1, x2, y2), img, label)
            cv2.imwrite(os.path.join(path, str(i)), img)
