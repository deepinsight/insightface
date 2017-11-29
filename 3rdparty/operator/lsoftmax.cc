/*!
 * Copyright (c) 2016 by Contributors
 * \file lsoftmax.cc
 * \brief LSoftmax from <Large-Margin Softmax Loss for Convolutional Neural Networks>
 * \author luoyetx
 */
#include "./lsoftmax-inl.h"

namespace mshadow {

template <typename DType>
inline void LSoftmaxForward(const Tensor<cpu, 2, DType> &x,
                            const Tensor<cpu, 2, DType> &w,
                            const Tensor<cpu, 1, DType> &label,
                            const Tensor<cpu, 2, DType> &out,
                            const Tensor<cpu, 1, DType> &x_norm,
                            const Tensor<cpu, 1, DType> &w_norm,
                            const Tensor<cpu, 1, DType> &k_table,
                            const Tensor<cpu, 1, DType> &c_table,
                            const int margin,
                            const DType beta) {
  LOG(FATAL) << "Not Implemented.";
}

template <typename DType>
inline void LSoftmaxBackward(const Tensor<cpu, 2, DType> &x,
                             const Tensor<cpu, 2, DType> &w,
                             const Tensor<cpu, 1, DType> &label,
                             const Tensor<cpu, 1, DType> &x_norm,
                             const Tensor<cpu, 1, DType> &w_norm,
                             const Tensor<cpu, 2, DType> &o_grad,
                             const Tensor<cpu, 2, DType> &x_grad,
                             const Tensor<cpu, 2, DType> &w_grad,
                             const Tensor<cpu, 2, DType> &workspace,
                             const Tensor<cpu, 1, DType> &k_table,
                             const Tensor<cpu, 1, DType> &c_table,
                             const int margin,
                             const DType beta) {
  LOG(FATAL) << "Not Implemented.";
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(LSoftmaxParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new LSoftmaxOp<cpu, DType>(param);
  })
  return op;
}

Operator *LSoftmaxProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(LSoftmaxParam);

MXNET_REGISTER_OP_PROPERTY(LSoftmax, LSoftmaxProp)
.describe("LSoftmax from <Large-Margin Softmax Loss for Convolutional Neural Networks>")
.add_argument("data", "Symbol", "data")
.add_argument("weight", "Symbol", "weight")
.add_argument("label", "Symbol", "label")
.add_arguments(LSoftmaxParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
