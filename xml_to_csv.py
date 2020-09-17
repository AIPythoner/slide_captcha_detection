# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 00:52:02 2018

@author: Xiang Guo
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(xml_path):
    xml_list = []
    for xml_file in glob.glob(xml_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # filename = os.path.basename(xml_file).replace('item', '').replace('.xml', '')
        # filename = 'item' + filename + '.jpg'
        for member in root.findall('object'):
            value = (
                root.find('filename').text,
                # filename,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
            )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # xml_path = r'./train/train_label'
    xml_path = r'E:\deepLearingData\character_position\test_label'
    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv('./bank_position/test.csv', index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()
    """
    python model_main.py --logtostderr --model_dir=./train/training/ --pipeline_config_path=.\train\training\ssd_mobilenet_v1_coco.config

    """
