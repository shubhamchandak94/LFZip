pip install --upgrade pip
pip install \
    tensorflow==1.15 \
    tensorflow-gpu==1.15
pip install tqdm
pip install \
      h5py \
      matplotlib \
      mkl \
      nose \
      notebook \
      Pillow \
      pandas \
      pydot \
      pyyaml \
      numpy \
      scipy \
      scikit-learn \
      six \
      theano \
      padasip \
      keras
#libbsc for entropy coding
(cd libbsc && make CC=g++-9)
# compile nlms_helper for actual compression with nlms
g++ nlms_helper.py -std=c++11 -o nlms_helper.out -Wall -O3 -march=native
