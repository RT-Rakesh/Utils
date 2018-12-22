# -*- coding: utf-8 -*-
"""
Created on Wen Dec 22 2018
@author: Rt-Rakesh

This script is a utility to generate label map from data in csv format.
Usage: Could be used by calling in jupyter notbooks.
"""
import pandas as pd
import os


def generate_label_map(csv_path, output_folder):
    """
    This Function generates label map for the data in csv format(path,xmin,ymin,xmax,ymax,label).
    Args:
    1.csv_path: --str The path to the data in csv format(path,xmin,ymin,xmax,ymax,label).
    2.output_folder: --str The path to the folder where the label map is stored.

    Returns:
    The label_map.pbtxt is generated.
    """
    filename = "label_map.pbtxt"
    output_name = os.path.join(output_folder, filename)
    file = pd.read_csv(csv_path, index_col=0)
    categories = file['label'].unique()
    end = '\n'
    s = ' '
    class_map = {}
    for ID, name in enumerate(categories):
        out = ''
        out += 'item' + s + '{' + end
        out += s*2 + 'id:' + ' ' + (str(ID+1)) + end
        out += s*2 + 'name:' + ' ' + '\'' + name + '\'' + end
        out += '}' + end*2
        with open(output_name, 'a') as f:
            f.write(out)
        class_map[name] = ID+1
    print("Label Map generated Successfully")
