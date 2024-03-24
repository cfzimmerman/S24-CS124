import math

"""
Mini script used to estimate a theoretical value for n_0 at which
Strassen's algorithm should call the O(n^3) matrix multiplication
algorithm.
"""


def recurrence(n: int, base: int) -> int:
    if n <= base:
        return 3 * (n ** 3) + (n ** 2) + n
    return 7 * recurrence(n / 2, base) + 18 * (n ** 2)


def search_log():
    for input_size in [256, 512, 1024, 2048, 8192]:
        base = input_size

        opt_base = math.inf
        opt_rec = math.inf
        while 4 < base:
            rec = recurrence(input_size, base)
            # print(
            #    f"n = {input_size}, n_0 = {base}, T(n) = {rec}")
            if rec < opt_rec:
                opt_rec = rec
                opt_base = base
            base /= 2
        print(f"n = {input_size}, optimal T(n) = {opt_rec}, n_0 = {opt_base}")


search_log()
