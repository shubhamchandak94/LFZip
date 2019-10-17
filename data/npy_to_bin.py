import numpy as np
import struct
import sys

if len(sys.argv) < 3:
    print("Usage: python npy_to_bin.py infile_name outfile_name")
    sys.exit(1)
infile=sys.argv[1]
outfile=sys.argv[2]

arr = np.load(infile)
assert arr.dtype == 'float32'

fout = open(outfile, 'wb')
arr_flat = arr.flatten()
for i in arr_flat:
    fout.write(struct.pack('f',i))
fout.close()
