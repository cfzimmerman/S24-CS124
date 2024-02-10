'''
This scripts generates weighted data that can be used to test heap correctness. 
'''

import sys
import json
from numbers import Integral
from random import random, randint


def generate():
    if len(sys.argv) != 5:
        print(
            "usage: python gen_wt_nums.py [filepath] [sizeof output] [rrange min] [rrange max]")
        return

    output_file = open(sys.argv[1], "w")

    output_size: int = int(sys.argv[2])
    rrange_min: int = int(sys.argv[3])
    rrange_max: int = int(sys.argv[4])

    if rrange_min > rrange_max:
        print("rrange_min cannot exceed rrange_max: ", rrange_min, rrange_max)
        return

    lst = []
    for _ in range(output_size):
        lst.append({
            "weight": random(),
            "val": randint(rrange_min, rrange_max)
        })

    output_file.write(json.dumps(lst))
    output_file.close()


generate()
