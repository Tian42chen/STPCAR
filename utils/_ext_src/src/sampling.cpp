// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "sampling.h"
#include "utils.h"

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), points.data<float>(),
                                 idx.data<int>(), output.data<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return output;
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx,
                              const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.type().is_cuda()) {
    gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                      idx.size(1), grad_out.data<float>(),
                                      idx.data<int>(), output.data<float>());
  } else {
    // TORCH_CHECK(false, "CPU not supported");
    for (int i = 0; i < grad_out.size(0); i++) {
      for (int l = 0; l < grad_out.size(1); l++) {
        for (int j = 0; j < idx.size(1); j++) {
          int a = idx[i][j].item<int>();
          output[i][l][a] += grad_out[i][l][j];
        }
      }
    }
  }

  return output;
}
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, points.data<float>(),
        tmp.data<float>(), output.data<int>());
  } else {
    // TORCH_CHECK(false, "CPU not supported");
    for (int i = 0; i < points.size(0); i++) {
      std::vector<int> indices(points.size(1));
      std::iota(indices.begin(), indices.end(), 0);
      std::random_shuffle(indices.begin(), indices.end());

      std::vector<int> samples(nsamples);
      samples[0] = indices[0];
      for (int j = 1; j < nsamples; j++) {
        float max_dist = -1;
        int max_idx = -1;
        for (int k = 0; k < indices.size(); k++) {
          if (std::find(samples.begin(), samples.end(), indices[k]) != samples.end()) {
            continue;
          }
          float dist = 1e10;
          for (int l = 0; l < j; l++) {
            float d = 0;
            for (int m = 0; m < 3; m++) {
              float diff = points[i][indices[k]][m].item<float>() - points[i][samples[l]][m].item<float>();
              d += diff * diff;
            }
            dist = std::min(dist, d);
          }
          if (dist > max_dist) {
            max_dist = dist;
            max_idx = indices[k];
          }
        }
        samples[j] = max_idx;
      }

      for (int j = 0; j < nsamples; j++) {
        output[i][j] = samples[j];
      }
    }
  }

  return output;
}
