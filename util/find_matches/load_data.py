import os
import sys
import json

import numpy as np

import db_connection


def main():
    db = db_connection.connect()
    cursor = db.cursor()

    try:
        with open(sys.argv[1]) as f:
            images = json.load(f)

        for (filename, feature_vector) in images.items():
            store_image(cursor, filename, feature_vector)

        db.commit()
    finally:
        cursor.close()
        db.close()


def store_image(cursor, filename, feature_vector):
    statement = '''
        INSERT INTO images
            (filename, feature_vector)
            VALUES
            (%s, %s)
    '''

    cursor.execute(statement, (filename, convert_to_memsql_vector(feature_vector)))


def convert_to_memsql_vector(vector):
    return np.array(vector).astype('float32').tobytes()


if __name__ == '__main__':
    main()
