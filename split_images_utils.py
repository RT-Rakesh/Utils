# -*- coding: utf-8 -*-
"""
Created on Wen Jan 08 2019
@author: Rt-Rakesh

This script is a utility to split the images in the test csv to a seperate folder for infernce rqquirment.
Usage: Should be used by calling in jupyter notbooks.
"""
import pandas as pd
import os
import shutil
from tqdm import tqdm


def split_images(csv_path, image_path, save_path):
    """
        This Function splits the images from the entire set of images.
    -----
    Args:
    -----
        1.csv_path: --str The path to the csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,label).
        2.image_path: --str  The path to the folder with the images.
        3.save_path: -- str The path to save the split images.

    ---------
    Returns:
    ---------
        The images listed in the csv will be copied to a new folder called test_images.
    """
    df = pd.read_csv(csv_path)
    if not os.path.exists(os.path.join(save_path, 'test_images')):
        os.makedirs(os.path.join(save_path, 'test_images'))
    for i in tqdm(list(set(df.path))):
        if os.path.exists(os.path.join(image_path, i)):
            shutil.copy(os.path.join(image_path, i), os.path.join(os.path.join(save_path, 'test_images'), i))
