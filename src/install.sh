pip install --upgrade pip
pip install \
    tensorflow==1.15 \
    tensorflow-gpu==1.15 \
    tqdm
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
(cd libbsc && make)
