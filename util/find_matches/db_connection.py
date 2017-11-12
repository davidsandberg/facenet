import os

import pymysql


def connect():
    return pymysql.connect(
        user=os.getenv('MYSQL_USER') or 'root',
        host=os.getenv('MYSQL_HOST') or '127.0.0.1',
        database=os.getenv('MYSQL_DATABASE') or 'cfs_test'
    )
