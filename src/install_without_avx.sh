# use with python 3.6 (conda create --name snakes python=3.6 \ conda activate snakes)
pip install --upgrade pip
pip install \
    tensorflow==1.5\
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
pip install padasip sklearn keras==2.1.5
