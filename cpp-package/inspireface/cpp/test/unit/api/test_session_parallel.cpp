/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "unit/test_helper/test_help.h"
#include "inspireface/middleware/thread/resource_pool.h"
#include <thread>

TEST_CASE("test_SessionParallel", "[Session][Parallel]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    auto image1 = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    auto image2 = inspirecv::Image::Create(GET_DATA("data/bulk/jntm.jpg"));

    int loop = 100;

    // Run it once to make sure the similarity is stable
    HFSessionCustomParameter parameter = {0};
    parameter.enable_recognition = 1;
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    HFSession session;
    HResult ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
    REQUIRE(ret == HSUCCEED);
    float expectedSimilarity = GenerateRandomNumbers(1, 0, 100)[0] / 100.0f;
    ret = CompareTwoFaces(session, image1, image2, expectedSimilarity);
    REQUIRE(ret);
    TEST_PRINT("Expected similarity: {}", expectedSimilarity);

    ret = HFReleaseInspireFaceSession(session);
    REQUIRE(ret == HSUCCEED);

    SECTION("Serial") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);

        float similarity = 0.0f;
        inspire::SpendTimer timeSpend("Serial loop: " + std::to_string(loop));
        timeSpend.Start();
        for (int i = 0; i < loop; ++i) {
            ret = CompareTwoFaces(session, image1, image2, similarity);
            REQUIRE(ret);
            REQUIRE(similarity == Approx(expectedSimilarity).epsilon(0.01));
        }
        timeSpend.Stop();
        std::cout << timeSpend << std::endl;
        ;

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("Parallel") {
        int N = 4;  // Use 4 sessions in parallel
#ifdef ISF_RKNPU_RV1106
        N = 1;  // Use 1 session in parallel
#endif
        inspire::parallel::ResourcePool<HFSession> sessionPool(N, [](HFSession& session) {
            auto ret = HFReleaseInspireFaceSession(session);
            if (ret != HSUCCEED) {
                TEST_ERROR_PRINT("Failed to release session: {}", ret);
            }
        });

        // Example Initialize N sessions to the resource pool
        for (int i = 0; i < N; ++i) {
            HFSessionCustomParameter parameter = {0};
            parameter.enable_recognition = 1;
            HFSession session;
            HResult ret = HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_ALWAYS_DETECT, 3, -1, -1, &session);
            REQUIRE(ret == HSUCCEED);
            sessionPool.AddResource(std::move(session));
        }

        // Create a thread pool to execute a task
        std::vector<std::thread> threads;
        std::atomic<int> completed(0);
        float similaritySum = 0.0f;
        std::mutex similarityMutex;

        inspire::SpendTimer timeSpend("Parallel loop: " + std::to_string(loop) + ", thread: " + std::to_string(N));
        timeSpend.Start();

        // Start worker thread
        int tasksPerThread = loop / N;
        int remainingTasks = loop % N;

        for (int i = 0; i < N; ++i) {
            int taskCount = tasksPerThread + (i < remainingTasks ? 1 : 0);
            threads.emplace_back([&, taskCount]() {
                for (int j = 0; j < taskCount; ++j) {
                    auto sessionGuard = sessionPool.AcquireResource();
                    float similarity = 0.0f;
                    HResult ret = CompareTwoFaces(*sessionGuard, image1, image2, similarity);
                    REQUIRE(ret);

                    REQUIRE(similarity == Approx(expectedSimilarity).epsilon(0.01));
                    {
                        std::lock_guard<std::mutex> lock(similarityMutex);
                        similaritySum += similarity;
                    }

                    completed++;
                }
            });
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        timeSpend.Stop();
        std::cout << timeSpend << std::endl;
        ;

        // Optional: Output average similarity(stability)
        TEST_PRINT("Average similarity: {}", (similaritySum / loop));
    }
}

TEST_CASE("test_SessionParallel_Memory", "[Session][Parallel][Memory]") {
    size_t memoryUsage = getCurrentMemoryUsage();
    TEST_PRINT("Current memory usage: {}MB", memoryUsage);
    int loop = 4;
#ifdef ISF_RKNPU_RV1106
    loop = 1;
#endif
    std::vector<HFSession> sessions;
    for (int i = 0; i < loop; ++i) {
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFSession session;
        HResult ret = HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_ALWAYS_DETECT, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        sessions.push_back(session);
        size_t memoryUsage = getCurrentMemoryUsage();
        TEST_PRINT("[alloc{}] Current memory usage: {}MB", i + 1, memoryUsage);
    }
    // Release all sessions
    for (int i = 0; i < loop; ++i) {
        auto ret = HFReleaseInspireFaceSession(sessions[i]);
        REQUIRE(ret == HSUCCEED);
        size_t memoryUsage = getCurrentMemoryUsage();
        TEST_PRINT("[free{}] Current memory usage: {}MB", i + 1, memoryUsage);
    }
}
