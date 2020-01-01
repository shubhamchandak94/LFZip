/*
NLMS helper written in C++ to speed things up.
Used in nlms_compress.py replacing part of code in nlms_compress_python.py.
The code related to the NLMS filtering is based on the source code available
at https://github.com/matousc89/padasip.
*/

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

float dot_product(const float *a, const float *b, const uint32_t n) {
  float ret = 0.0;
  for (uint32_t i = 0; i < n; i++) ret += (*(a + i)) * (*(b + i));
  return ret;
}

class NLMS_base {
  uint32_t n;
  float mu;
  float eps;
  std::vector<float> w;

 public:
  NLMS_base(const uint32_t n_, const float mu_ = 0.1, const float eps_ = 1.0);
  void adapt(const float true_val, const float *arr);
  float predict(const float *arr);
};

class NLMS_predictor {
  NLMS_base *filter;
  uint32_t n;
  uint32_t nseries;
  uint32_t j;  // for multivariate series, which variable is this
 public:
  NLMS_predictor(const uint32_t n_, const float mu, const uint32_t nseries,
                 const uint32_t j);
  float predict(const std::vector<float> &recon_arr, const uint32_t idx);
  ~NLMS_predictor() { delete filter; }
};

struct Params {
  uint32_t len;
  uint8_t nseries;
  std::vector<float> maxerror;
  std::vector<float> maxerror_original;
  std::vector<int64_t> min_bin_idx;
  std::vector<int64_t> max_bin_idx;
  std::vector<uint8_t> quantization_bytes;
  std::vector<NLMS_predictor *> predictor;
  std::string basedir;
  bool decompress_flag;
  ~Params() {
    for (size_t i = 0; i < predictor.size(); i++) delete predictor[i];
  }
};

void get_params(Params &p, int argc, char **argv);

void compress(Params &p);

void decompress(Params &p);

int main(int argc, char **argv) {
  Params p;
  get_params(p, argc, argv);
  if (p.decompress_flag)
    decompress(p);
  else
    compress(p);
  return 0;
}

void compress(Params &p) {
  std::string bin_idx_file_prefix = p.basedir + "/bin_idx.";
  std::string float_file_prefix = p.basedir + "/float.";
  std::vector<std::ofstream> f_bin_idx(p.nseries), f_float(p.nseries);
  std::vector<float> reconstruction(p.nseries * p.len);
  for (uint32_t j = 0; j < p.nseries; j++) {
    f_bin_idx[j].open(bin_idx_file_prefix + std::to_string(j),
                      std::ios::binary);
    f_float[j].open(float_file_prefix + std::to_string(j), std::ios::binary);
  }
  std::string data_file = p.basedir + "/data.bin";
  std::string reconstruction_file = p.basedir + "/recon.bin";
  std::ifstream fin_data(data_file, std::ios::binary);
  for (uint32_t i = 0; i < p.len; i++) {
    for (uint32_t j = 0; j < p.nseries; j++) {
      float dataval;
      fin_data.read((char *)&dataval, sizeof(float));
      float predval = p.predictor[j]->predict(reconstruction, i);
      float diff = dataval - predval;
      int64_t bin_idx = int64_t(std::round((diff / (2.0 * p.maxerror[j]))));
      if (p.min_bin_idx[j] <= bin_idx && bin_idx <= p.max_bin_idx[j]) {
        reconstruction[j + p.nseries * i] =
            predval + (float)(p.maxerror[j] * bin_idx * 2.0);
        // check if numeric precision issues present, if yes, just store
        // original data as it is
        if (std::abs(reconstruction[j + p.nseries * i] - dataval) <=
            p.maxerror_original[j]) {
          if (p.quantization_bytes[j] == 1) {
            int8_t bin_idx_1 = (int8_t)bin_idx;
            f_bin_idx[j].write((char *)&bin_idx_1, sizeof(int8_t));
          } else {
            int16_t bin_idx_2 = (int16_t)bin_idx;
            f_bin_idx[j].write((char *)&bin_idx_2, sizeof(int16_t));
          }
          continue;
        }
      }
      if (p.quantization_bytes[j] == 1) {
        int8_t bin_idx_1 = (int8_t)(p.min_bin_idx[j]-1);
        f_bin_idx[j].write((char *)&bin_idx_1, sizeof(int8_t));
      } else {
        int16_t bin_idx_2 = (int16_t)(p.min_bin_idx[j]-1);
        f_bin_idx[j].write((char *)&bin_idx_2, sizeof(int16_t));
      }
      f_float[j].write((char *)&dataval, sizeof(float));
      reconstruction[j + p.nseries * i] = dataval;
    }
  }
  fin_data.close();
  for (uint32_t j = 0; j < p.nseries; j++) {
    f_bin_idx[j].close();
    f_float[j].close();
  }
  std::ofstream f_reconstruction(reconstruction_file, std::ios::binary);
  f_reconstruction.write((char *)&reconstruction[0], sizeof(float) * p.len * p.nseries);
  f_reconstruction.close();
}

void decompress(Params &p) {
  std::string bin_idx_file_prefix = p.basedir + "/bin_idx.";
  std::string float_file_prefix = p.basedir + "/float.";
  std::vector<std::ifstream> f_bin_idx(p.nseries), f_float(p.nseries);
  std::vector<float> reconstruction(p.nseries * p.len);
  for (uint32_t j = 0; j < p.nseries; j++) {
    f_bin_idx[j].open(bin_idx_file_prefix + std::to_string(j),
                      std::ios::binary);
    f_float[j].open(float_file_prefix + std::to_string(j), std::ios::binary);
  }
  std::string reconstruction_file = p.basedir + "/recon.bin";
  for (uint32_t i = 0; i < p.len; i++) {
    for (uint32_t j = 0; j < p.nseries; j++) {
      float predval = p.predictor[j]->predict(reconstruction, i);
      int64_t bin_idx;
      if (p.quantization_bytes[j] == 1) {
        int8_t bin_idx_1;
        f_bin_idx[j].read((char *)&bin_idx_1, sizeof(int8_t));
        bin_idx = bin_idx_1;
      } else {
        int16_t bin_idx_2;
        f_bin_idx[j].read((char *)&bin_idx_2, sizeof(int16_t));
        bin_idx = bin_idx_2;
      }
      if (bin_idx == p.min_bin_idx[j] - 1) {
        float dataval;
        f_float[j].read((char *)&dataval, sizeof(float));
        reconstruction[j + p.nseries * i] = dataval;
      } else {
        reconstruction[j + p.nseries * i] =
            predval + (float)(p.maxerror[j] * bin_idx * 2.0);
      }
    }
  }
  for (uint32_t j = 0; j < p.nseries; j++) {
    f_bin_idx[j].close();
    f_float[j].close();
  }
  std::ofstream f_reconstruction(reconstruction_file, std::ios::binary);
  f_reconstruction.write((char *)&reconstruction[0], sizeof(float) * p.len * p.nseries);
  f_reconstruction.close();
}

void get_params(Params &p, int argc, char **argv) {
  if (argc != 3)
    throw std::runtime_error("Incorrect number of arguments to nlms_helper");
  std::string mode = std::string(argv[1]);
  if (mode == "c")
    p.decompress_flag = false;
  else if (mode == "d")
    p.decompress_flag = true;
  else
    throw std::runtime_error(
        "Incorrect first argument to nlms_helper: expected c or d");
  p.basedir = std::string(argv[2]);
  std::string params_file = p.basedir + "/params";
  std::string params_maxerror_original_file =
      p.basedir + "/params.maxerror_original";
  std::ifstream fin_params(params_file, std::ios::binary);
  std::ifstream fin_params_maxerror_original;
  fin_params.read((char *)&p.nseries, sizeof(uint8_t));
  fin_params.read((char *)&p.len, sizeof(uint32_t));
  p.maxerror.resize(p.nseries);
  p.min_bin_idx.resize(p.nseries);
  p.max_bin_idx.resize(p.nseries);
  p.quantization_bytes.resize(p.nseries);
  p.predictor.resize(p.nseries);
  if (!p.decompress_flag) {
    fin_params_maxerror_original.open(params_maxerror_original_file,
                                      std::ios::binary);
    p.maxerror_original.resize(p.nseries);
  }
  for (uint8_t j = 0; j < p.nseries; j++) {
    fin_params.read((char *)&p.maxerror[j], sizeof(float));
    uint32_t n_nlms;
    float mu_nlms;
    fin_params.read((char *)&n_nlms, sizeof(uint32_t));
    fin_params.read((char *)&mu_nlms, sizeof(float));
    p.predictor[j] = new NLMS_predictor(n_nlms, mu_nlms, p.nseries, j);
    fin_params.read((char *)&p.quantization_bytes[j], sizeof(uint8_t));
    if (p.quantization_bytes[j] == 1) {
      p.max_bin_idx[j] = 127;
      p.min_bin_idx[j] = -127;
    } else if (p.quantization_bytes[j] == 2) {
      p.max_bin_idx[j] = 32767;
      p.min_bin_idx[j] = -32767;
    } else {
      throw std::runtime_error(
          "Invalid value of quantization_bytes encountered");
    }
    if (!p.decompress_flag)
      fin_params_maxerror_original.read((char *)&p.maxerror_original[j],
                                        sizeof(float));
  }
}

NLMS_base::NLMS_base(const uint32_t n_, const float mu_, const float eps_) {
  n = n_;
  mu = mu_;
  eps = eps_;
  if ((mu >= 1000.0 || mu <= 0.0) || (eps >= 1000.0 || eps <= 0.0))
    throw std::runtime_error("Invalid value of mu or eps.");
  w.resize(n, 0.0);
}

void NLMS_base::adapt(const float true_val, const float *arr) {
  float y = dot_product(&w[0], arr, n);
  float e = true_val - y;
  float nu = mu / (eps + dot_product(arr, arr, n));
  for (uint32_t i = 0; i < n; i++) w[i] += nu * e * (*(arr + i));
}

float NLMS_base::predict(const float *arr) {
  return dot_product(arr, &w[0], n);
}

NLMS_predictor::NLMS_predictor(const uint32_t n_, const float mu,
                               const uint32_t nseries_, const uint32_t j_) {
  filter = new NLMS_base(n_ * nseries_ + j_, mu);
  n = n_;
  nseries = nseries_;
  j = j_;
}

float NLMS_predictor::predict(const std::vector<float> &recon_arr,
                              const uint32_t idx) {
  if (idx > n) {
    filter->adapt(recon_arr[j + nseries * (idx - 1)],
                  &recon_arr[nseries * (idx - n - 1)]);
    return filter->predict(&recon_arr[nseries * (idx - n)]);
  } else if (idx > 0 && idx <= n) {
    return recon_arr[j + nseries * (idx - 1)];
  } else {
    return 0.0;
  }
}
