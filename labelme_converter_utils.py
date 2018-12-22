# -*- coding: utf-8 -*-
"""
Created on Wen Dec 12 2018
@author: Rt-Rakesh

This script is a utility to convert data to and from the format accepted by the LabelMe tool.
Usage: Could be used by calling in jupyter notbooks.
"""

import pandas as pd
import xml.etree.cElementTree as ET
from tqdm import tqdm
import os


def convert_csv_2_labelme_xml(csv_path, xml_folder):
    """
    This Function writes the annoations in a xml format compatible with label me toolself.
    Args:
    1.csv path: --str The path to the csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,label).
    2.xml_folder: -- str The output folder path to be mentioned here.

    Returns:
    The xml corresponding to the input annotaions are written in the path mentioned.
    """

    data = pd.read_csv(csv_path)
    path_list = list(set(data.path))
    for imgs in path_list:
        temp_data = data[data['path'] == imgs].reset_index(drop=True)
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "filename").text = str(imgs)
        ET.SubElement(annotation, "folder").text = "roboreader_examples"
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "sourceImage").text = "The MIT-CSAIL database of objects and scenes"
        ET.SubElement(source, "sourceAnnotation").text = "LabelMe Webtool"
        for i, rows in tqdm(temp_data.iterrows()):
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = str(rows['label'])
            ET.SubElement(obj, "deleted").text = "0"
            ET.SubElement(obj, "verified").text = "0"
            ET.SubElement(obj, "occluded").text = "no"
            ET.SubElement(obj, "attributes")
            parts = ET.SubElement(obj, "parts")
            ET.SubElement(parts, "hasparts")
            ET.SubElement(parts, "ispartof")
            ET.SubElement(obj, "date").text = "31-Dec-2099 11:59:59"
            ET.SubElement(obj, "id").text = str(int(i)+1)
            ET.SubElement(obj, "type").text = "bounding_box"
            polygon = ET.SubElement(obj, "polygon")
            ET.SubElement(polygon, "username").text = "anonymous"
            pt = ET.SubElement(polygon, "pt")
            ET.SubElement(pt, "x").text = str(rows['xmin'])
            ET.SubElement(pt, "y").text = str(rows['ymin'])
            pt = ET.SubElement(polygon, "pt")
            ET.SubElement(pt, "x").text = str(rows['xmax'])
            ET.SubElement(pt, "y").text = str(rows['ymin'])
            pt = ET.SubElement(polygon, "pt")
            ET.SubElement(pt, "x").text = str(rows['xmax'])
            ET.SubElement(pt, "y").text = str(rows['ymax'])
            pt = ET.SubElement(polygon, "pt")
            ET.SubElement(pt, "x").text = str(rows['xmin'])
            ET.SubElement(pt, "y").text = str(rows['ymax'])
        tree = ET.ElementTree(annotation)
        tree.write(os.path.join(xml_folder, imgs.split('.')[0]+".xml"))
