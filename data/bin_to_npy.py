import numpy as np
import struct
import sys

if len(sys.argv) < 3:
    print("Usage: python npy_to_bin.py infile_name outfile_name [nseries]")
    sys.exit(1)
infile=sys.argv[1]
outfile=sys.argv[2]

if len(sys.argv) > 3:
    nseries = int(sys.argv[3])
else:
    nseries = 1

with open(infile, 'rb') as fin:
    buf = fin.read()


if nseries == 1:
    assert len(buf)%4 == 0
    arr_len = len(buf)/4
    arr = np.array(list(struct.iter_unpack('f',buf)), dtype=np.float32)

    np.save(outfile, arr.flatten())
else:
    assert len(buf)%(4*nseries) == 0
    arr_len = len(buf)/(4*nseries)
    l = list(struct.iter_unpack('f',buf))
    arr = np.array((nseries,len(buf)//(4*nseries),dtype=np.float32))
    for j in range(nseries):
        arr[j] = l[j*arr_len:(j+1)*arr_len]
    np.save(outfile, arr)
