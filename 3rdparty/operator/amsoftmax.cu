#include "./amsoftmax-inl.h"
#include <math.h>

namespace mshadow {
namespace cuda {

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


template<typename DType>
__global__ void AmSoftmaxForwardKernel(const Tensor<gpu, 2, DType> x,
                                      const Tensor<gpu, 2, DType> w,
                                      const Tensor<gpu, 1, DType> label,
                                      Tensor<gpu, 2, DType> out,
                                      Tensor<gpu, 2, DType> oout,
                                      const DType margin,
                                      const DType s) {
  const int n = x.size(0); //batch size
  const int feature_dim = x.size(1); //embedding size, 512 for example
  const int m = w.size(0);//num classes
  const DType cos_m = cos(margin);
  const DType sin_m = sin(margin);
  CUDA_KERNEL_LOOP(i, n) {
    const int yi = static_cast<int>(label[i]);
    const DType fo_i_yi = out[i][yi];
    oout[i][0] = fo_i_yi;
    if(fo_i_yi>=0.0) {
      const DType cos_t = fo_i_yi / s;
      const DType sin_t = sqrt(1.0-cos_t*cos_t);
      out[i][yi] = fo_i_yi*cos_m - (s*sin_t*sin_m);
    }
  }
}

template<typename DType>
inline void AmSoftmaxForward(const Tensor<gpu, 2, DType> &x,
                            const Tensor<gpu, 2, DType> &w,
                            const Tensor<gpu, 1, DType> &label,
                            const Tensor<gpu, 2, DType> &out,
                            const Tensor<gpu, 2, DType> &oout,
                            const DType margin,
                            const DType s) {
  const int n = x.size(0);
  const int m = w.size(0);
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((n + kBaseThreadNum - 1) / kBaseThreadNum);
  AmSoftmaxForwardKernel<<<dimGrid, dimBlock>>>(x, w, label, out, oout, margin, s);
}


template<typename DType>
__global__ void AmSoftmaxBackwardXKernel(const Tensor<gpu, 2, DType> x,
                                        const Tensor<gpu, 2, DType> w,
                                        const Tensor<gpu, 1, DType> label,
                                        const Tensor<gpu, 2, DType> out,
                                        const Tensor<gpu, 2, DType> oout,
                                        const Tensor<gpu, 2, DType> o_grad,
                                        Tensor<gpu, 2, DType> x_grad,
                                        const Tensor<gpu, 2, DType> workspace,
                                        const DType margin,
                                        const DType s) {
  const int nthreads = x.size(0) * x.size(1);
  //const int nthreads = x.size(0);
  const int feature_dim = x.size(1);
  const DType cos_m = cos(margin);
  const DType nsin_m = sin(margin)*-1.0;
  const DType ss = s*s;
  CUDA_KERNEL_LOOP(idx, nthreads) {
    const int i = idx / feature_dim;
    const int l = idx % feature_dim;
    //const int i = idx;
    const int yi = static_cast<int>(label[i]);
    if(oout[i][0]>=0.0) {
      //x_grad[i][l] -= o_grad[i][yi] * w[yi][l];
      //c = 1-cost*cost, = sint*sint
      const DType cost = oout[i][0]/s;
      const DType c = 1.0-cost*cost;
      const DType dc_dx = -2.0/ss*oout[i][0]*w[yi][l];
      const DType d_sint_dc = 1.0/(2*sqrt(c));
      const DType d_sint_dx = dc_dx*d_sint_dc;
      const DType df_dx = cos_m*w[yi][l] + s*nsin_m*d_sint_dx;
      x_grad[i][l] += o_grad[i][yi] * (df_dx - w[yi][l]);
    }
  }
}

template<typename DType>
__global__ void AmSoftmaxBackwardWKernel(const Tensor<gpu, 2, DType> x,
                                        const Tensor<gpu, 2, DType> w,
                                        const Tensor<gpu, 1, DType> label,
                                        const Tensor<gpu, 2, DType> out,
                                        const Tensor<gpu, 2, DType> oout,
                                        const Tensor<gpu, 2, DType> o_grad,
                                        Tensor<gpu, 2, DType> w_grad,
                                        const Tensor<gpu, 2, DType> workspace,
                                        const DType margin,
                                        const DType s) {
  const int nthreads = w.size(0) * w.size(1);
  const int n = x.size(0);
  const int feature_dim = w.size(1);
  const DType cos_m = cos(margin);
  const DType nsin_m = sin(margin)*-1.0;
  const DType ss = s*s;
  CUDA_KERNEL_LOOP(idx, nthreads) {
    const int j = idx / feature_dim;
    const int l = idx % feature_dim;
    DType dw = 0;
    for (int i = 0; i < n; ++i) {
      const int yi = static_cast<int>(label[i]);
      if (yi == j&&oout[i][0]>=0.0) {
        const DType cost = oout[i][0]/s;
        const DType c = 1.0-cost*cost;
        const DType dc_dw = -2.0/ss*oout[i][0]*x[i][l];
        const DType d_sint_dc = 1.0/(2*sqrt(c));
        const DType d_sint_dw = dc_dw*d_sint_dc;
        const DType df_dw = cos_m*x[i][l] + s*nsin_m*d_sint_dw;
        dw += o_grad[i][yi] * (df_dw - x[i][l]);
      }
    }
    w_grad[j][l] += dw;
  }
}

template<typename DType>
inline void AmSoftmaxBackward(const Tensor<gpu, 2, DType> &x,
                             const Tensor<gpu, 2, DType> &w,
                             const Tensor<gpu, 1, DType> &label,
                             const Tensor<gpu, 2, DType> &out,
                             const Tensor<gpu, 2, DType> &oout,
                             const Tensor<gpu, 2, DType> &o_grad,
                             const Tensor<gpu, 2, DType> &x_grad,
                             const Tensor<gpu, 2, DType> &w_grad,
                             const Tensor<gpu, 2, DType> &workspace,
                             const DType margin,
                             const DType s) {
  const int n = x.size(0);
  const int feature_dim = x.size(1);
  const int m = w.size(0);
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((n + kBaseThreadNum - 1) / kBaseThreadNum);
  dimGrid.x = ((n * feature_dim + kBaseThreadNum - 1) / kBaseThreadNum);
  AmSoftmaxBackwardXKernel<<<dimGrid, dimBlock>>>(x, w, label, out, oout, o_grad, x_grad, workspace,
                                                 margin, s);
  dimGrid.x = ((m * feature_dim + kBaseThreadNum - 1) / kBaseThreadNum);
  AmSoftmaxBackwardWKernel<<<dimGrid, dimBlock>>>(x, w, label, out, oout, o_grad, w_grad, workspace,
                                                 margin, s);
}

}  // namespace cuda

template<typename DType>
inline void AmSoftmaxForward(const Tensor<gpu, 2, DType> &x,
                            const Tensor<gpu, 2, DType> &w,
                            const Tensor<gpu, 1, DType> &label,
                            const Tensor<gpu, 2, DType> &out,
                            const Tensor<gpu, 2, DType> &oout,
                            const DType margin,
                            const DType s) {
  cuda::AmSoftmaxForward(x, w, label, out, oout, margin, s);
}

template<typename DType>
inline void AmSoftmaxBackward(const Tensor<gpu, 2, DType> &x,
                             const Tensor<gpu, 2, DType> &w,
                             const Tensor<gpu, 1, DType> &label,
                             const Tensor<gpu, 2, DType> &out,
                             const Tensor<gpu, 2, DType> &oout,
                             const Tensor<gpu, 2, DType> &o_grad,
                             const Tensor<gpu, 2, DType> &x_grad,
                             const Tensor<gpu, 2, DType> &w_grad,
                             const Tensor<gpu, 2, DType> &workspace,
                             const DType margin,
                             const DType s) {
  cuda::AmSoftmaxBackward(x, w, label, out, oout, o_grad, x_grad, w_grad, workspace, margin, s);
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(AmSoftmaxParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AmSoftmaxOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet
