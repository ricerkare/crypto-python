import numpy, random
from conversion_functions import roll_uint64

# Testing 
x = numpy.random.randint(0, 0xffffffffffffffff, dtype=numpy.uint64)
for n in range(64):
    print("roll(x, " + str(n) + "):" + \
          format(roll_uint64(x, numpy.uint64(n)), "064b"))

