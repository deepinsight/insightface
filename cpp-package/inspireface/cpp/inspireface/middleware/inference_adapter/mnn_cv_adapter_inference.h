#ifndef MNN_SIMPLE_INFER_H
#define MNN_SIMPLE_INFER_H

#include "opencv2/opencv.hpp"
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/MNNForwardType.h>

class MNNCVAdapterInference {
public:
    MNNCVAdapterInference(const std::string &model, int thread, const float mean[], const float normal[],
                          bool use_model_bin = false) {
#ifdef FORWARD_CUDA
        backend_ = MNN_FORWARD_CUDA;
#else
        backend_ = MNN_FORWARD_CPU;
#endif

        if (use_model_bin) {
            detect_model_ = std::shared_ptr<MNN::Interpreter>(
                MNN::Interpreter::createFromBuffer(model.c_str(), model.size()));
        } else {
            detect_model_ = std::shared_ptr<MNN::Interpreter>(
                MNN::Interpreter::createFromFile(model.c_str()));
        }

        _config.type = backend_;
        _config.numThread = thread;
        MNN::BackendConfig backendConfig;
        backendConfig.precision = MNN::BackendConfig::Precision_High;
        backendConfig.power = MNN::BackendConfig::Power_High;
        _config.backendConfig = &backendConfig;
        for (int i = 0; i < 3; i++) {
            this->mean[i] = mean[i];
            this->normal[i] = normal[i];
        }
    }

    ~MNNCVAdapterInference() {
        detect_model_->releaseModel();
        detect_model_->releaseSession(sess);
    }

    void Init(const std::string &input, const std::vector<std::string> &outputs, int width,
              int height) {
        sess = detect_model_->createSession(_config);
        tensor_shape_ = {1, 3, height, width};
        input_ = detect_model_->getSessionInput(sess, input.c_str());

        // Resize input tensor
        detect_model_->resizeTensor(input_, tensor_shape_);
        
        // Store output tensors and resize them
        for (const auto& output_name : outputs) {
            auto output_tensor = detect_model_->getSessionOutput(sess, output_name.c_str());
            output_tensors_.emplace_back(output_tensor);
        }

        detect_model_->resizeSession(sess);

        width_ = width;
        height_ = height;
    }

    std::vector<std::vector<float>> Infer(const cv::Mat &mat) {
        assert(mat.rows == height_);
        assert(mat.cols == width_);
        MNN::CV::ImageProcess::Config config;
        config.destFormat = MNN::CV::ImageFormat::BGR;
        config.sourceFormat = MNN::CV::BGR;
        for (int i = 0; i < 3; i++) {
            config.mean[i] = mean[i];
            config.normal[i] = normal[i];
        }

        std::unique_ptr<MNN::CV::ImageProcess> process(
                MNN::CV::ImageProcess::create(config));
        process->convert(mat.data, mat.cols, mat.rows, (int) mat.step1(), input_);
        detect_model_->runSession(sess);
        
        std::vector<std::vector<float>> results;

        for (auto output : output_tensors_) {
            auto dimType = input_->getDimensionType();

            if (output->getType().code != halide_type_float) {
                dimType = MNN::Tensor::TENSORFLOW;
            }
            std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output, dimType));
            output->copyToHostTensor(outputUser.get());
            auto type = outputUser->getType();
            auto size = outputUser->elementSize();
            std::vector<float> tempValues(size);
            if (type.code == halide_type_float) {
                auto values = outputUser->host<float>();
                for (int i = 0; i < size; ++i) {
                    tempValues[i] = values[i];
                }
            }
            results.push_back(tempValues);
        }

        return results;
    }

    std::vector<std::vector<std::vector<float>>> BatchInfer(const std::vector<cv::Mat> &images) {
        int batch_size = images.size();
        tensor_shape_[0] = batch_size;  // Update batch size
        detect_model_->resizeTensor(input_, tensor_shape_);
        detect_model_->resizeSession(sess);

        MNN::CV::ImageProcess::Config config;
        config.destFormat = MNN::CV::ImageFormat::BGR;
        config.sourceFormat = MNN::CV::BGR;
        for (int i = 0; i < 3; i++) {
            config.mean[i] = mean[i];
            config.normal[i] = normal[i];
        }

        std::unique_ptr<MNN::CV::ImageProcess> process(
                MNN::CV::ImageProcess::create(config));
        
        std::shared_ptr<MNN::Tensor> inputUser(new MNN::Tensor(input_, MNN::Tensor::TENSORFLOW));
        auto size_h = inputUser->height();
        auto size_w = inputUser->width();
        auto bpp = inputUser->channel();

        for (int batch = 0; batch < batch_size; ++batch) {
            const auto& mat = images[batch];
            assert(mat.rows == height_);
            assert(mat.cols == width_);

            // No need to setScale since the images are already resized
            process->convert(mat.data, mat.cols, mat.rows, (int)mat.step1(), inputUser->host<uint8_t>() + inputUser->stride(0) * batch * inputUser->getType().bytes(), size_w, size_h, bpp, 0, inputUser->getType());
        }

        input_->copyFromHostTensor(inputUser.get());
        detect_model_->runSession(sess);
        
        std::vector<std::vector<std::vector<float>>> all_results(batch_size);

        for (auto output : output_tensors_) {
            auto dimType = input_->getDimensionType();

            if (output->getType().code != halide_type_float) {
                dimType = MNN::Tensor::TENSORFLOW;
            }
            std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output, dimType));
            output->copyToHostTensor(outputUser.get());
            auto type = outputUser->getType();
            auto size = outputUser->elementSize() / batch_size;

            for (int batch = 0; batch < batch_size; ++batch) {
                std::vector<float> tempValues(size);
                if (type.code == halide_type_float) {
                    auto values = outputUser->host<float>() + batch * outputUser->stride(0);
                    for (int i = 0; i < size; ++i) {
                        tempValues[i] = values[i];
                    }
                }
                all_results[batch].push_back(tempValues);
            }
        }

        return all_results;
    }

    float mean[3]{};
    float normal[3]{};

private:
    std::shared_ptr<MNN::Interpreter> detect_model_;
    MNN::Tensor *input_{};
    std::vector<MNN::Tensor*> output_tensors_;
    MNN::Session *sess{};
    std::vector<int> tensor_shape_;
    MNN::ScheduleConfig _config;
    MNNForwardType backend_;
    int width_{};
    int height_{};
};

#endif // MNN_SIMPLE_INFER_H
