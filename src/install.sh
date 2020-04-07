pip install -q --upgrade pip
pip install -q --upgrade setuptools
pip install -q tensorflow==1.15
pip install -q tensorflow-gpu==1.15
pip install -q tqdm
pip install -q \
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
# libbsc for entropy coding
(cd libbsc && make)
# compile nlms_helper for actual compression with nlms
g++ nlms_helper.cpp -std=c++11 -o nlms_helper.out -Wall -O3 -march=native
