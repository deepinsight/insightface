#include "./amsoftmax-inl.h"

namespace mshadow {

template <typename DType>
inline void AmSoftmaxForward(const Tensor<cpu, 2, DType> &x,
                            const Tensor<cpu, 2, DType> &w,
                            const Tensor<cpu, 1, DType> &label,
                            const Tensor<cpu, 2, DType> &out,
                            const Tensor<cpu, 2, DType> &oout,
                            const DType margin,
                            const DType s) {
  LOG(FATAL) << "Not Implemented.";
}

template <typename DType>
inline void AmSoftmaxBackward(const Tensor<cpu, 2, DType> &x,
                             const Tensor<cpu, 2, DType> &w,
                             const Tensor<cpu, 1, DType> &label,
                             const Tensor<cpu, 2, DType> &out,
                             const Tensor<cpu, 2, DType> &oout,
                             const Tensor<cpu, 2, DType> &o_grad,
                             const Tensor<cpu, 2, DType> &x_grad,
                             const Tensor<cpu, 2, DType> &w_grad,
                             const Tensor<cpu, 2, DType> &workspace,
                             const DType margin,
                             const DType s) {
  LOG(FATAL) << "Not Implemented.";
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(AmSoftmaxParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AmSoftmaxOp<cpu, DType>(param);
  })
  return op;
}

Operator *AmSoftmaxProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(AmSoftmaxParam);

MXNET_REGISTER_OP_PROPERTY(AmSoftmax, AmSoftmaxProp)
.describe("AmSoftmax from <TODO>")
.add_argument("data", "Symbol", "data")
.add_argument("weight", "Symbol", "weight")
.add_argument("label", "Symbol", "label")
.add_arguments(AmSoftmaxParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
