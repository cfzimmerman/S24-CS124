import sys
import numpy
import json


def gen_test_data():
    if len(sys.argv) != 5:
        print(
            "***.py [int num rows] [int num cols] [int min val] [int max val]")
        return

    num_rows = int(sys.argv[1])
    num_cols = int(sys.argv[2])
    min_val = int(sys.argv[3])
    max_val = int(sys.argv[4])

    left = [numpy.random.randint(min_val, max_val, num_cols).tolist()
            for _ in range(num_rows)]
    right = [numpy.random.randint(min_val, max_val, num_rows).tolist()
             for _ in range(num_cols)]

    print(json.dumps({
        "left": left,
        "right": right,
        "sum": numpy.add(left, right).tolist(),
        "diff": numpy.subtract(left, right).tolist(),
        "prod": numpy.matmul(left, right).tolist(),
    }))


gen_test_data()
