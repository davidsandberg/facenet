import os
import json
import uuid
import datetime
import logging
from botocore.exceptions import ClientError
from db_schema import Image, Face, Duplicate, FaceFeature
from imageanalysis.photodna import photodna_hash

from memsql.common import database
import base64
import binascii
import numpy as np
import binascii
import time
import json
import app_context
import boto3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_NAME = 'cfs_test'


def convert_to_vector(pdna):
    if not pdna:
        return None

    if len(pdna) == 144:
        pdna_decoded = pdna
    elif len(pdna) == 192:
        pdna_decoded = base64.b64decode(pdna)
    elif len(pdna) == 288:
        pdna_decoded = binascii.unhexlify(pdna)
    else:
        raise ValueError('Unknown PhotoDNA format')

    vector = np.fromstring(pdna_decoded, '144B').astype('float32')

    # encode in hex to prevent encoding issues with MemSQL clients
    return vector.tobytes().encode('hex')


def calc_dot_product(photo_dna_hex):
  # test dot_product example form docs:
  print run_query("SELECT DOT_PRODUCT(JSON_ARRAY_PACK('[1.0, 0.5, 2.0]'), JSON_ARRAY_PACK('[1.0, 0.5, 2.0]')) as dp")

  print run_query('SELECT DOT_PRODUCT(unhex("'+photo_dna_hex+'"), unhex("'+photo_dna_hex+'")) as dp FROM image')
  # test on our image.
  r = run_query('SELECT photodna, DOT_PRODUCT(unhex("'+photo_dna_hex+'"), photodna ) as dp FROM image')
  print r


def calc_euclidean(photo_dna_hex):
  # test expmpale fomr docs.
  print run_query("SELECT EUCLIDEAN_DISTANCE(JSON_ARRAY_PACK('[1.0, 0.5, 2.0]'), JSON_ARRAY_PACK('[0.7, 0.2, 1.7]')) as ed")
  r = run_query('SELECT photodna, EUCLIDEAN_DISTANCE(unhex("'+photo_dna_hex+'"), photodna ) as ed FROM image')
  print r


def test_face_feature():
  face = run_query('SELECT id, feature, face_id FROM face_feature LIMIT 1')
  print '\n drs face:'
  print face
  r = run_query('SELECT id, face_id, EUCLIDEAN_DISTANCE(unhex("'+face.feature+'"), feature ) as ed FROM face_feature')
  print r



def setup_env():
    s3 = boto3.resource('s3')
    env_file = s3.Object('thorn-cfs-config', 'analysis.json')
    envs = json.loads(env_file.get()['Body'].read().decode('utf-8'))

    for env in envs:
        if env not in os.environ:
            os.environ[env] = envs[env]



# Convert to a vector that can be used in MemSQL dot products.
# This can be stored in any blob column, such as VARBINARY(576).
# https://docs.memsql.com/sql-reference/v5.8/dot_product/
def convert_to_vector(pdna_decoded):
    if not pdna_decoded:
        return None

    vector = np.fromstring(pdna_decoded, '144B').astype('float32')
    return vector.tobytes()

def connect():
  master_agg = '192.168.65.1'#  10.0.3.186:3306
  conn = database.connect(host=master_agg, user='root', password='pdnatest', database=DB_NAME)
  return conn


def run_query(query):
  print '\nquery ' + query
  c = connect()
  r = c.query(query)
  c.close()
  return r


if __name__ == '__main__':
  setup_env()

  # GEt photodna hash for an image:
  full_image_path = '/cfs/93D373B0-33F4-4281-BD1E-A408C1A71C49_4_mod.jpg'
  photo_dna_base64 = photodna_hash(filename_or_image=full_image_path)
  photo_dna_bytes = binascii.a2b_base64(photo_dna_base64)
  photo_dna_hex = convert_to_vector(photo_dna_bytes).encode('hex')


  # calc_dot_product(photo_dna_hex)
  # calc_euclidean(photo_dna_hex)
  test_face_feature()

  # # check for exact photdna match
  # print run_query('SELECT photodna FROM image where photodna = unhex("'+photo_dna_hex+'")')
