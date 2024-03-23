import sys
import numpy as np


def gen_raw_sq(dim: int, v_min: int, v_max: int) -> None:
    assert (dim >= 0)
    rng = np.random.default_rng()
    left = [rng.integers(low=v_min, high=v_max, size=dim)
            for _ in range(dim)]
    right = [rng.integers(low=v_min, high=v_max, size=dim)
             for _ in range(dim)]
    prod = np.matmul(left, right)
    for group in [left, right, prod]:
        for row in group:
            for entry in row:
                sys.stdout.write(str(entry) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Produces square matrices with one entry per stdout line.")
        print("A single call produces a 3n^2 list of NxN entries for \
two matrices and their product.\n")
        print("python ***.py [uint dimension] [int min] [int max]")
        print("python ***.py 4 0 10")
    else:
        dim = int(sys.argv[1])
        v_min = int(sys.argv[2])
        v_max = int(sys.argv[3])
        gen_raw_sq(dim, v_min, v_max)
