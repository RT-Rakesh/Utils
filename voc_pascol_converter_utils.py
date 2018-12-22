# -*- coding: utf-8 -*-
"""
Created on Wen Dec 12 2018
@author: Rt-Rakesh

This script is a utility to convert data to and from the format accepted by the label img tool.
Usage: Should be used by calling in jupyter notbooks.
"""

import pandas as pd
import xml.etree.cElementTree as ET
from tqdm import tqdm
import glob
import os


def convert_csv_2_pascol_voc_xml(csv_path, image_path, xml_folder):
    """
    This Function writes the annoations in a xml format compatible with label me toolself.
    Args:
    1.csv path: --str The path to the csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,label).
    2.image_path: --str The path to directory containing the images must be mentioned.
    3.xml_folder: -- str The output folder path to be mentioned here.

    Returns:
    The xml corresponding to the input annotaions are written in the path mentioned.
    """
    data = pd.read_csv(csv_path)
    path_list = list(set(data.path))
    for imgs in tqdm(path_list):
        temp_data = data[data['path'] == imgs].reset_index(drop=True)
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = "roboreader_training_Image"
        ET.SubElement(annotation, "filename").text = str(imgs)
        ET.SubElement(annotation, "path").text = image_path+str(imgs)
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Unknown"
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = 0
        ET.SubElement(size, "height").text = 0
        ET.SubElement(size, "depth").text = str(3)
        ET.SubElement(annotation, "segmented").text = str(0)
        for i, rows in temp_data.iterrows():
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = str(rows['label'])
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(rows['xmin'])
            ET.SubElement(bndbox, "ymin").text = str(rows['ymin'])
            ET.SubElement(bndbox, "xmax").text = str(rows['xmax'])
            ET.SubElement(bndbox, "ymax").text = str(rows['ymax'])
        tree = ET.ElementTree(annotation)
        tree.write(os.path.join(xml_folder, imgs.split('.')[0]+".xml"))


def convert_pascol_voc_xml_to_csv(xml_folder_path, out_path=False, return_df=False):
    """
    This Function converts the annoations xml into csv format(path,xmin,ymin,xmax,ymax,label).
    Args:
    1.xml_folder_path: --str The path to the xml folder.
    2.out_path: --str Optional Argument --The path to the csv file must be saved.(eg.path/train_data.csv). If not given saves the csv as data.csv .
    3.return_df: -- Bool Default value: False Change this to True if the convert xml are required in dataframe.

    Returns:
    The csv corresponding to the input annotaions are written in the path mentioned.
    Returns the dataframe.
    """
    xml_list = []
    for xml_file in tqdm(glob.glob(xml_folder_path + '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('path').text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     member[0].text,
                     )
            xml_list.append(value)
    column_name = ['path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
    data_df = pd.DataFrame(xml_list, columns=column_name)
    data_df['path'] = [i.split('/')[-1] for i in data_df.path]
    if out_path:
        data_df.to_csv(out_path, index=False)
    else:
        data_df.to_csv("./data.csv", index=False)
    # f = open(out_path+"training_data.csv", "w")
    # for item in xml_list:
    #     line = ','.join(str(x) for x in item)
    #     f.write(line + '\n')
    # f.close()
    print('Successfully converted XML to CSV.')
    if return_df:
        return data_df
