#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <inspirecv/inspirecv.h>
#include <inspireface/include/inspireface/session.h>
#include <inspireface/include/inspireface/launch.h>
#include <inspireface/middleware/thread/resource_pool.h>
#include <inspireface/include/inspireface/spend_timer.h>
#include <thread>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> <loop_count> <thread_num>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    int loop = std::stoi(argv[3]);
    int thread_num = std::stoi(argv[4]);

    if (thread_num > 10) {
        std::cerr << "Error: thread_num cannot be greater than 10" << std::endl;
        return -1;
    }
    if (loop < 1000) {
        std::cerr << "Error: loop count must be at least 1000" << std::endl;
        return -1;
    }

    INSPIREFACE_CONTEXT->Load(model_path);
    inspirecv::Image image = inspirecv::Image::Create(image_path);
    inspirecv::FrameProcess process =
      inspirecv::FrameProcess::Create(image.Data(), image.Height(), image.Width(), inspirecv::BGR, inspirecv::ROTATION_0);

    inspire::parallel::ResourcePool<inspire::Session> sessionPool(thread_num, [](inspire::Session& session) {

    });

    for (int i = 0; i < thread_num; ++i) {
        inspire::CustomPipelineParameter param;
        param.enable_recognition = true;
        param.enable_liveness = true;
        param.enable_mask_detect = true;
        param.enable_face_attribute = true;
        param.enable_face_quality = true;
        inspire::Session session = inspire::Session::Create(inspire::DetectModuleMode::DETECT_MODE_ALWAYS_DETECT, 1, param);
        sessionPool.AddResource(std::move(session));
    }

    std::vector<std::thread> threads;
    int tasksPerThread = loop / thread_num;
    int remainingTasks = loop % thread_num;

    // Run the task in parallel
    for (int i = 0; i < thread_num; ++i) {
        int taskCount = tasksPerThread + (i < remainingTasks ? 1 : 0);
        threads.emplace_back([&, taskCount]() {
            for (int j = 0; j < taskCount; ++j) {
                auto sessionGuard = sessionPool.AcquireResource();
                std::vector<inspire::FaceTrackWrap> results;
                int32_t ret;
                ret = sessionGuard->FaceDetectAndTrack(process, results);
                if (ret != 0) {
                    std::cerr << "FaceDetectAndTrack failed" << std::endl;
                    break;
                }
                if (results.size() == 0) {
                    std::cerr << "Not found face" << std::endl;
                    break;
                }
            }
        });
    }

    // Print basic information before starting
    std::cout << "\n=== Configuration Information ===" << std::endl;
    std::cout << "Model Path: " << model_path << std::endl;
    std::cout << "Image Path: " << image_path << std::endl;
    std::cout << "Total Loop Count: " << loop << std::endl;
    std::cout << "Number of Threads: " << thread_num << std::endl;
    std::cout << "Tasks per Thread: " << tasksPerThread << std::endl;
    std::cout << "Remaining Tasks: " << remainingTasks << std::endl;
    std::cout << "==============================\n" << std::endl;

    inspire::SpendTimer timer("Number of threads: " + std::to_string(thread_num) + ", Number of tasks: " + std::to_string(loop));
    timer.Start();
    for (auto& thread : threads) {
        thread.join();
    }
    timer.Stop();
    std::cout << timer << std::endl;

    // Convert microseconds to milliseconds and print
    double milliseconds = timer.Total() / 1000.0;
    std::cout << "Total execution time: " << milliseconds << " ms" << std::endl;

    return 0;
}
