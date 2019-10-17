pip install --upgrade pip
pip install \
    tensorflow\
    tensorflow-gpu\
    tqdm
conda install \
      h5py \
      matplotlib \
      mkl \
      nose \
      notebook \
      Pillow \
      pandas \
      pydot \
      pygpu \
      pyyaml \
      scikit-learn \
      six \
      theano
pip install padasip sklearn keras
#libbsc for read_seq compression
(cd libbsc && make)
