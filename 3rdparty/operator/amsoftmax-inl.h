/*!
 * Copyright (c) 2018 by Contributors
 * \file amsoftmax-inl.h
 * \brief AmSoftmax from <TODO>
 * \author Jia Guo
 */
#ifndef MXNET_OPERATOR_AMSOFTMAX_INL_H_
#define MXNET_OPERATOR_AMSOFTMAX_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace amsoftmax_enum {
enum AmSoftmaxOpInputs {kData, kWeight, kLabel};
enum AmSoftmaxOpOutputs {kOut, kOOut};
enum AmSoftmaxResource {kTempSpace};
}

struct AmSoftmaxParam : public dmlc::Parameter<AmSoftmaxParam> {
  float margin;
  float s;
  int num_hidden;
  int verbose;
  float eps;
  DMLC_DECLARE_PARAMETER(AmSoftmaxParam) {
    DMLC_DECLARE_FIELD(margin).set_default(0.5).set_lower_bound(0.0)
    .describe("AmSoftmax margin");
    DMLC_DECLARE_FIELD(s).set_default(64.0).set_lower_bound(1.0)
    .describe("s to X");
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output");
    DMLC_DECLARE_FIELD(verbose).set_default(0)
    .describe("Log for beta change");
    DMLC_DECLARE_FIELD(eps).set_default(1e-10f)
    .describe("l2 eps");
  }
};

template<typename xpu, typename DType>
class AmSoftmaxOp : public Operator {
 public:
  explicit AmSoftmaxOp(AmSoftmaxParam param) {
    this->param_ = param;
    count_ = 0;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);
    CHECK_EQ(req.size(), 2);
    CHECK_EQ(req[amsoftmax_enum::kOut], kWriteTo);
    Stream<xpu> *stream = ctx.get_stream<xpu>();
    const int n = in_data[amsoftmax_enum::kData].size(0); //batch size
    const int m = in_data[amsoftmax_enum::kWeight].size(0);//num classes
    Tensor<xpu, 2, DType> x = in_data[amsoftmax_enum::kData].FlatTo2D<xpu, DType>(stream);
    Tensor<xpu, 2, DType> w = in_data[amsoftmax_enum::kWeight].FlatTo2D<xpu, DType>(stream);
    Tensor<xpu, 1, DType> label = in_data[amsoftmax_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), stream);
    Tensor<xpu, 2, DType> out = out_data[amsoftmax_enum::kOut].FlatTo2D<xpu, DType>(stream);
    Tensor<xpu, 2, DType> oout = out_data[amsoftmax_enum::kOOut].get_with_shape<xpu, 2, DType>(Shape2(n,1), stream);
    //Tensor<xpu, 2, DType> workspace = ctx.requested[amsoftmax_enum::kTempSpace].get_space_typed<xpu, 2, DType>(Shape2(n, 1), stream);
#if defined(__CUDACC__)
    CHECK_EQ(stream->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    // original fully connected
    out = dot(x, w.T());
    if (ctx.is_train) {
      const DType margin = static_cast<DType>(param_.margin);
      const DType s = static_cast<DType>(param_.s);
      AmSoftmaxForward(x, w, label, out, oout, margin, s);
    }
  }

  //virtual void GradNorm(mshadow::Tensor<xpu, 2, DType> grad, mshadow::Stream<xpu>* s) {
  //  using namespace mshadow;
  //  using namespace mshadow::expr;
  //  Tensor<cpu, 2, DType> grad_cpu(grad.shape_);
  //  AllocSpace(&grad_cpu);
  //  Copy(grad_cpu, grad, s);
  //  DType grad_norm = param_.eps;
  //  for(uint32_t i=0;i<grad_cpu.shape_[0];i++) {
  //    for(uint32_t j=0;j<grad_cpu.shape_[1];j++) {
  //      grad_norm += grad_cpu[i][j]*grad_cpu[i][j];
  //    }
  //  }
  //  grad_norm = sqrt(grad_norm);
  //  grad_cpu /= grad_norm;
  //  Copy(grad, grad_cpu, s);
  //  FreeSpace(&grad_cpu);
  //}

  virtual DType GradNorm(mshadow::Tensor<xpu, 2, DType> grad, mshadow::Stream<xpu>* s) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Tensor<cpu, 2, DType> grad_cpu(grad.shape_);
    AllocSpace(&grad_cpu);
    Copy(grad_cpu, grad, s);
    DType grad_norm = param_.eps;
    for(uint32_t i=0;i<grad_cpu.shape_[0];i++) {
      for(uint32_t j=0;j<grad_cpu.shape_[1];j++) {
        grad_norm += grad_cpu[i][j]*grad_cpu[i][j];
      }
    }
    grad_norm = sqrt(grad_norm);
    //grad_cpu /= grad_norm;
    //Copy(grad, grad_cpu, s);
    FreeSpace(&grad_cpu);
    return grad_norm;
  }
  virtual void Print(mshadow::Tensor<xpu, 2, DType> tensor, mshadow::Stream<xpu>* s) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Tensor<cpu, 2, DType> tensor_cpu(tensor.shape_);
    AllocSpace(&tensor_cpu);
    Copy(tensor_cpu, tensor, s);
    for(uint32_t i=0;i<tensor_cpu.shape_[0];i++) {
      for(uint32_t j=0;j<tensor_cpu.shape_[1];j++) {
        std::cout<<tensor_cpu[i][j]<<",";
      }
      std::cout<<std::endl;
    }
    FreeSpace(&tensor_cpu);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);
    CHECK_GE(in_grad.size(), 2);
    CHECK_GE(req.size(), 2);
    CHECK_EQ(req[amsoftmax_enum::kData], kWriteTo);
    CHECK_EQ(req[amsoftmax_enum::kWeight], kWriteTo);
    Stream<xpu> *stream = ctx.get_stream<xpu>();
    const int n = in_data[amsoftmax_enum::kData].size(0);
    const int m = in_data[amsoftmax_enum::kWeight].size(0);
    Tensor<xpu, 2, DType> x = in_data[amsoftmax_enum::kData].FlatTo2D<xpu, DType>(stream);
    Tensor<xpu, 2, DType> w = in_data[amsoftmax_enum::kWeight].FlatTo2D<xpu, DType>(stream);
    Tensor<xpu, 1, DType> label = in_data[amsoftmax_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), stream);
    Tensor<xpu, 2, DType> out = out_data[amsoftmax_enum::kOut].FlatTo2D<xpu, DType>(stream);
    Tensor<xpu, 2, DType> oout = out_data[amsoftmax_enum::kOOut].get_with_shape<xpu, 2, DType>(Shape2(n,1), stream);
    Tensor<xpu, 2, DType> o_grad = out_grad[amsoftmax_enum::kOut].FlatTo2D<xpu, DType>(stream);
    Tensor<xpu, 2, DType> x_grad = in_grad[amsoftmax_enum::kData].FlatTo2D<xpu, DType>(stream);
    Tensor<xpu, 2, DType> w_grad = in_grad[amsoftmax_enum::kWeight].FlatTo2D<xpu, DType>(stream);
    Tensor<xpu, 2, DType> workspace = ctx.requested[amsoftmax_enum::kTempSpace].get_space_typed<xpu, 2, DType>(Shape2(n, 1), stream);
#if defined(__CUDACC__)
    CHECK_EQ(stream->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    // original fully connected
    x_grad = dot(o_grad, w);
    w_grad = dot(o_grad.T(), x);
    // large margin fully connected
    const DType margin = static_cast<DType>(param_.margin);
    const DType s = static_cast<DType>(param_.s);
    AmSoftmaxBackward(x, w, label, out, oout, o_grad, x_grad, w_grad, workspace, margin, s);
    count_+=1;
    if (param_.verbose) {
      if(count_%param_.verbose==0) {
        DType n = GradNorm(x_grad, stream);
        LOG(INFO)<<"x_grad norm:"<<n;
        n = GradNorm(w_grad, stream);
        LOG(INFO)<<"w_grad norm:"<<n;
        //Print(oout, stream);
      }
    }
  }


 private:
  AmSoftmaxParam param_;
  uint32_t count_;
};

template<typename xpu>
Operator *CreateOp(AmSoftmaxParam param, int dtype);

#if DMLC_USE_CXX11
class AmSoftmaxProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "weight", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "ooutput"};
  }

  int NumOutputs() const override {
    return 2;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, label, weight]";
    const TShape &dshape = in_shape->at(amsoftmax_enum::kData);
    const TShape &lshape = in_shape->at(amsoftmax_enum::kLabel);
    CHECK_EQ(dshape.ndim(), 2) << "data shape should be (batch_size, feature_dim)";
    CHECK_EQ(lshape.ndim(), 1) << "label shape should be (batch_size,)";
    const int n = dshape[0];
    const int feature_dim = dshape[1];
    const int m = param_.num_hidden;
    SHAPE_ASSIGN_CHECK(*in_shape, amsoftmax_enum::kWeight, Shape2(m, feature_dim));
    out_shape->clear();
    out_shape->push_back(Shape2(n, m));  // output
    out_shape->push_back(Shape2(n, 1));  // output
    aux_shape->clear();
    return true;
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[amsoftmax_enum::kOut], 
            in_data[amsoftmax_enum::kData],
            in_data[amsoftmax_enum::kWeight], in_data[amsoftmax_enum::kLabel]};
  }

  std::string TypeString() const override {
    return "AmSoftmax";
  }

  OperatorProperty *Copy() const override {
    auto ptr = new AmSoftmaxProp();
    ptr->param_ = param_;
    return ptr;
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  AmSoftmaxParam param_;
};
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif 
