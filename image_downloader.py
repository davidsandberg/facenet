from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import shutil
import argparse
import os
import glob
import pandas as pd
import numpy as np
from feature_extractor.batch import download_images
from activelearning.others.downloader import Downloader

def fetch_images(csv_path, image_path, n_workers=100):
    """This function is used to download all imgs from csv file
       if the image is not exists.

    Args:
      csv_path: the csv path
      img_path: the path to save downloaded images
      
    Returns:
        Nothing
    """
    
    csv_reference_df = pd.read_csv(csv_path, 
                                   index_col=False, 
                                   encoding='utf-8')
    image_url_list = csv_reference_df['ImgUrl'].unique().tolist()
    image_url_clean_list = []
    
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    
    # Check the image is whether exist in the image pool
    for n in image_url_list:
        image_filename = n.split('/')[-1]
        temp_path = os.path.join(image_path, image_filename)
        
        if not os.path.isfile(temp_path):
            image_url_clean_list.append(n)
    
    # Download image if not in the image pool
    download_images(urls=image_url_clean_list, 
                    n_workers=n_workers, 
                    output_folder=image_path)

def prepare_images(filename_list=['test_data_sku_CCNA_20180619.csv','test_data_sku_CCNA_20180612.csv','test_data_sku_CCNA_20180614.csv','test_data_sku_CCNA_20180616.csv','test_data_sku_CCNA_20180623.csv','test_data_sku_CCNA_20180627.csv','test_data_sku_CCNA_20180629.csv','test_data_sku_CCNA_20180709.csv','test_data_sku_RCCB_20180817.csv'], INPUT_DIR = '/home/caffe/caffe/examples/sku_classification/', OUTPUT_DIR = '/datadrive/images/ccna_test_data_latest_full'):
    output_images_dir = os.path.join(OUTPUT_DIR, 'all_images')
    output_cropped_images_dir = os.path.join(OUTPUT_DIR, 'all_cropped_images')

    # The path to save the single .csv file containing all target .csv files
    sum_filename_out = '/datadrive/ccna_all_test_datasets.csv' # you can it what you want

    # The path head sku list no reference images, the first col is sku ID and NO header
    csv_filename_in = '/home/caffe/facenet/CCNA_head_sku_list.csv'# You need ask 

    # Locate .cvs and true labels for cropping
    filepath_list = [os.path.join(INPUT_DIR, n) for n in filename_list]

    # Read head SKU list without reference images
    sku_id_df = pd.read_csv(csv_filename_in, 
                            index_col=False, 
                            header=None,
                            encoding='utf-8')
    sku_id_df = sku_id_df.loc[:,0].tolist()
    sum_df = pd.DataFrame()

    print(' -> Reading the head SKU list with no reference \
          images from: {}'.format(csv_filename_in))

    for n in filepath_list:
        temp_df = pd.read_csv(n, index_col=False, encoding='utf-8')
        temp_df = temp_df.loc[temp_df.SystemId.isin(sku_id_df)]
        sum_df = sum_df.append(temp_df)

        print(' -> Obtaining target .csv dataset from: {}'.format(n))

    # Save information to temp loaction
    sum_df.to_csv(sum_filename_out, index=False, encoding='utf-8')

    print(' -> The combination all .csv datasets and \
          save it into: {}'.format(sum_filename_out))

    # Begin downloading
    fetch_images(sum_filename_out, output_images_dir)

    # Uncomment blow code if you want clean saved all cropped images
    #if os.path.exists(output_cropped_images_dir):
    #    shutil.rmtree(output_cropped_images_dir)

    # Begin cropping
    Downloader.crop_save_sku(output_cropped_images_dir,
                             output_images_dir,
                             sum_filename_out,
                             ignore_ids = [-1, 0, 2, 1, 1265, 1000050])

    print('All done')

prepare_images()