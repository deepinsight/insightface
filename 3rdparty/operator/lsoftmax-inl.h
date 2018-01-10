/*!
 * Copyright (c) 2016 by Contributors
 * \file lsoftmax-inl.h
 * \brief LSoftmax from <Large-Margin Softmax Loss for Convolutional Neural Networks>
 * \author luoyetx
 */
#ifndef MXNET_OPERATOR_LSOFTMAX_INL_H_
#define MXNET_OPERATOR_LSOFTMAX_INL_H_

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

namespace lsoftmax_enum {
enum LSoftmaxOpInputs {kData, kWeight, kLabel};
enum LSoftmaxOpOutputs {kOut, kDataNorm, kWeightNorm};
enum LSoftmaxResource {kTempSpace};
}

struct LSoftmaxParam : public dmlc::Parameter<LSoftmaxParam> {
  int margin;
  float beta;
  float beta_min;
  float scale;
  int num_hidden;
  bool grad_norm;
  int verbose;
  float eps;
  DMLC_DECLARE_PARAMETER(LSoftmaxParam) {
    DMLC_DECLARE_FIELD(margin).set_default(2).set_lower_bound(1)
    .describe("LSoftmax margin");
    DMLC_DECLARE_FIELD(beta).set_default(1).set_lower_bound(0)
    .describe("LSoftmax beta, same as lambda to weight original value");
    DMLC_DECLARE_FIELD(beta_min).set_default(0).set_lower_bound(0)
    .describe("Minimum beta");
    DMLC_DECLARE_FIELD(scale).set_default(1).set_range(0, 1)
    .describe("Scale of beta during training for every iteration");
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output");
    DMLC_DECLARE_FIELD(grad_norm).set_default(false)
    .describe("do grad norm");
    DMLC_DECLARE_FIELD(verbose).set_default(0)
    .describe("Log for beta change");
    DMLC_DECLARE_FIELD(eps).set_default(1e-10f)
    .describe("l2 eps");
  }
};

template<typename xpu, typename DType>
class LSoftmaxOp : public Operator {
 public:
  explicit LSoftmaxOp(LSoftmaxParam param) {
    this->param_ = param;
    // setup global lookup table
    k_table_.clear();
    c_table_.clear();
    k_table_.push_back(1);
    c_table_.push_back(1);
    const int margin = param.margin;
    const double pi = std::atan(1) * 4;
    double factor = 1;
    for (int i = 1; i <= margin; ++i) {
      factor = factor * (margin - i + 1) / i;
      k_table_.push_back(std::cos(i * pi / margin));
      c_table_.push_back(factor);
    }
    //next_beta_ = param.beta * 0.1f;
    count_ = 0;
    if(const char* env_p = std::getenv("BETA")) {
      float _beta = std::atof(env_p);
      if (param_.verbose) {
        LOG(INFO)<<"beta:"<<_beta;
        LOG(INFO)<<param_.margin<<","<<param_.beta<<","<<param_.beta_min<<","<<param_.scale;
      }
      param_.beta = _beta;
    }
    else if(const char* env_p = std::getenv("GLOBAL_STEP")) {
      int nbatch = std::atoi(env_p);
      if (param_.verbose) {
        LOG(INFO)<<"nbatch:"<<nbatch;
      }
      float _beta = param.beta*std::pow((double)param.scale, (double)nbatch);
      param_.beta = std::max(_beta, param_.beta_min);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);
    CHECK_EQ(req.size(), 3);
    CHECK_EQ(req[lsoftmax_enum::kOut], kWriteTo);
    CHECK(req[lsoftmax_enum::kDataNorm] == kNullOp ||
          req[lsoftmax_enum::kDataNorm] == kWriteTo);
    CHECK(req[lsoftmax_enum::kWeightNorm] == kNullOp ||
          req[lsoftmax_enum::kWeightNorm] == kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = in_data[lsoftmax_enum::kData].size(0);
    const int m = in_data[lsoftmax_enum::kWeight].size(0);
    Tensor<xpu, 2, DType> x = in_data[lsoftmax_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> w = in_data[lsoftmax_enum::kWeight].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> label = in_data[lsoftmax_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 2, DType> out = out_data[lsoftmax_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> x_norm = out_data[lsoftmax_enum::kDataNorm].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 1, DType> w_norm = out_data[lsoftmax_enum::kWeightNorm].get_with_shape<xpu, 1, DType>(Shape1(m), s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    // original fully connected
    out = dot(x, w.T());
    if (ctx.is_train) {
      // large margin fully connected
      const int margin = param_.margin;
      if(const char* env_p = std::getenv("BETA")) {
        float _beta = std::atof(env_p);
        param_.beta = _beta;
      }
      const DType beta = static_cast<DType>(param_.beta);
      //LOG(INFO)<<"beta:"<<beta<<std::endl;
      Tensor<cpu, 1, DType> k_table_cpu(k_table_.data(), Shape1(k_table_.size()));
      Tensor<cpu, 1, DType> c_table_cpu(c_table_.data(), Shape1(c_table_.size()));
      Tensor<xpu, 1, DType> k_table_xpu(Shape1(k_table_.size()));
      Tensor<xpu, 1, DType> c_table_xpu(Shape1(c_table_.size()));
      k_table_xpu.set_stream(s);
      c_table_xpu.set_stream(s);
      AllocSpace(&k_table_xpu);
      AllocSpace(&c_table_xpu);
      Copy(k_table_xpu, k_table_cpu, s);
      Copy(c_table_xpu, c_table_cpu, s);
      LSoftmaxForward(x, w, label, out, x_norm, w_norm, k_table_xpu, c_table_xpu, margin, beta);
      FreeSpace(&k_table_xpu);
      FreeSpace(&c_table_xpu);
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
    CHECK_EQ(out_data.size(), 3);
    CHECK_GE(in_grad.size(), 2);
    CHECK_GE(req.size(), 2);
    CHECK_EQ(req[lsoftmax_enum::kData], kWriteTo);
    CHECK_EQ(req[lsoftmax_enum::kWeight], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = in_data[lsoftmax_enum::kData].size(0);
    const int m = in_data[lsoftmax_enum::kWeight].size(0);
    Tensor<xpu, 2, DType> x = in_data[lsoftmax_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> w = in_data[lsoftmax_enum::kWeight].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> label = in_data[lsoftmax_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 1, DType> x_norm = out_data[lsoftmax_enum::kDataNorm].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 1, DType> w_norm = out_data[lsoftmax_enum::kWeightNorm].get_with_shape<xpu, 1, DType>(Shape1(m), s);
    Tensor<xpu, 2, DType> o_grad = out_grad[lsoftmax_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> x_grad = in_grad[lsoftmax_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> w_grad = in_grad[lsoftmax_enum::kWeight].FlatTo2D<xpu, DType>(s);
    // workspace is used for cos_t, cos_mt, k, sin2_t, fo and cos_t_m for every data point
    Tensor<xpu, 2, DType> workspace = ctx.requested[lsoftmax_enum::kTempSpace].get_space_typed<xpu, 2, DType>(Shape2(6, n), s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    // original fully connected
    x_grad = dot(o_grad, w);
    w_grad = dot(o_grad.T(), x);
    // large margin fully connected
    const int margin = param_.margin;
    const DType beta = static_cast<DType>(param_.beta);
    count_+=1;
    if (param_.verbose) {
      if(count_%param_.verbose==0) {
        LOG(INFO)<<"["<<count_<<"]current beta:"<<beta;
        DType n = GradNorm(x_grad, s);
        LOG(INFO)<<"x_grad norm:"<<n;
        n = GradNorm(w_grad, s);
        LOG(INFO)<<"w_grad norm:"<<n;
      }
    }
    Tensor<cpu, 1, DType> k_table_cpu(k_table_.data(), Shape1(k_table_.size()));
    Tensor<cpu, 1, DType> c_table_cpu(c_table_.data(), Shape1(c_table_.size()));
    Tensor<xpu, 1, DType> k_table_xpu(Shape1(k_table_.size()));
    Tensor<xpu, 1, DType> c_table_xpu(Shape1(c_table_.size()));
    k_table_xpu.set_stream(s);
    c_table_xpu.set_stream(s);
    AllocSpace(&k_table_xpu);
    AllocSpace(&c_table_xpu);
    Copy(k_table_xpu, k_table_cpu, s);
    Copy(c_table_xpu, c_table_cpu, s);
    LSoftmaxBackward(x, w, label, x_norm, w_norm, o_grad, x_grad, w_grad, workspace,
                     k_table_xpu, c_table_xpu, margin, beta);
    FreeSpace(&k_table_xpu);
    FreeSpace(&c_table_xpu);
    //if(param_.grad_norm) {
    //  GradNorm(x_grad, s);
    //  GradNorm(w_grad, s);
    //}
    // dirty hack, should also work for multi device
    if(std::getenv("BETA")==NULL) {
      param_.beta *= param_.scale;
      param_.beta = std::max(param_.beta, param_.beta_min);
    }
    //LOG(INFO)<<"w_grad:"<<w_grad.shape_[0]<<","<<w_grad.shape_[1];
    //norm = reduce_with_axis<red::sum, false>(F<mxnet::op::mshadow_op::square>(w_grad), 2);
    //norm = F<mxnet::op::mshadow_op::square_root>(norm + param_.eps);
    //out = data / broadcast_with_axis(norm, 1, dshape[2]);
    //if (param_.beta < next_beta_) {
    //  next_beta_ *= 0.1f;
    //  if (param_.verbose) {
    //    LOG(INFO) << "LSoftmax changes beta to " << param_.beta;
    //  }
    //}
  }

  //Tensor<xpu, 2, DType> grad_norm(const Tensor<xpu, 2, DType> grad) {
  //}




 private:
  LSoftmaxParam param_;
  // global lookup table
  std::vector<DType> k_table_;
  std::vector<DType> c_table_;
  //float next_beta_;
  uint32_t count_;
};  // class LSoftmaxOp

template<typename xpu>
Operator *CreateOp(LSoftmaxParam param, int dtype);

#if DMLC_USE_CXX11
class LSoftmaxProp : public OperatorProperty {
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
    return {"output", "data_norm", "weight_norm"};
  }

  int NumOutputs() const override {
    return 3;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, label, weight]";
    const TShape &dshape = in_shape->at(lsoftmax_enum::kData);
    const TShape &lshape = in_shape->at(lsoftmax_enum::kLabel);
    CHECK_EQ(dshape.ndim(), 2) << "data shape should be (batch_size, feature_dim)";
    CHECK_EQ(lshape.ndim(), 1) << "label shape should be (batch_size,)";
    const int n = dshape[0];
    const int feature_dim = dshape[1];
    const int m = param_.num_hidden;
    SHAPE_ASSIGN_CHECK(*in_shape, lsoftmax_enum::kWeight, Shape2(m, feature_dim));
    out_shape->clear();
    out_shape->push_back(Shape2(n, m));  // output
    out_shape->push_back(Shape1(n));  // data norm
    out_shape->push_back(Shape1(m));  // weight norm
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
    return {out_grad[lsoftmax_enum::kOut], out_data[lsoftmax_enum::kDataNorm],
            out_data[lsoftmax_enum::kWeightNorm], in_data[lsoftmax_enum::kData],
            in_data[lsoftmax_enum::kWeight], in_data[lsoftmax_enum::kLabel]};
  }

  std::string TypeString() const override {
    return "LSoftmax";
  }

  OperatorProperty *Copy() const override {
    auto ptr = new LSoftmaxProp();
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
  LSoftmaxParam param_;
};  // class LSoftmaxProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_LSOFTMAX_INL_H_
