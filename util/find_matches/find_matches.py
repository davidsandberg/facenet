import sys

import pymysql

import db_connection


def main():
    db = db_connection.connect()
    cursor = db.cursor(pymysql.cursors.DictCursor)

    statement = '''
        select
        	a.filename as a,
        	b.filename as b,
        	DOT_PRODUCT(a.feature_vector, b.feature_vector) as dot_dist,
        	EUCLIDEAN_DISTANCE(a.feature_vector, b.feature_vector) as euc_dist
        from images a inner join images b
        on a.id != b.id and a.id < b.id
        having euc_dist < %s
        order by euc_dist
    '''

    cursor.execute(statement, (sys.argv[1]))
    print 'a,b,dot_dist,euc_dist'
    for row in cursor:
        print '{},{},{},{}'.format(row['a'], row['b'], row['dot_dist'], row['euc_dist'])


if __name__ == '__main__':
    main()
