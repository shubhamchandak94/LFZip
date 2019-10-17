pip install --upgrade pip
pip install \
    tensorflow==1.15 \
    tensorflow-gpu==1.15 \
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
