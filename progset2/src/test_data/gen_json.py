import sys
import numpy
import json


def gen_square():
    assert len(sys.argv) == 4
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


def gen_asymmetric():
    assert len(sys.argv) == 5
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
        "prod": numpy.matmul(left, right).tolist(),
    }))


if len(sys.argv) == 4:
    gen_square()
elif len(sys.argv) == 5:
    gen_asymmetric()
else:
    print("Produces matrices and their arithmetic results for use in tests.\n")
    print(
        "Generate square matrices:\n\
        ***.py [int dimension] [int min val] [int max val]")
    print(
        "Generate asymmetric matrices:\n\
        ***.py [int num rows] [int num cols] [int min val] [int max val]")
