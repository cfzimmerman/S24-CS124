import sys
import numpy
import json


def gen_test_data():
    if len(sys.argv) != 4:
        print(
            "***.py [int dimension] [int min val] [int max val]")
        return

    dim = int(sys.argv[1])
    min_val = int(sys.argv[2])
    max_val = int(sys.argv[3])

    left = [numpy.random.randint(min_val, max_val, dim).tolist()
            for _ in range(dim)]
    right = [numpy.random.randint(min_val, max_val, dim).tolist()
             for _ in range(dim)]

    print(json.dumps({
        "left": left,
        "right": right,
        "sum": numpy.add(left, right).tolist(),
        "diff": numpy.subtract(left, right).tolist(),
        "prod": numpy.matmul(left, right).tolist(),
    }))


gen_test_data()
