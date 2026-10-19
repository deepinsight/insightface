#include <chrono>
#include <dlfcn.h>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <shared-library>" << std::endl;
        return 2;
    }

    dlerror();
    const auto start = std::chrono::steady_clock::now();
    void* handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start);

    if (handle == nullptr) {
        const char* error = dlerror();
        std::cerr << "load_result=failure elapsed_us=" << elapsed.count()
                  << " error=" << (error == nullptr ? "unknown" : error) << std::endl;
        return 1;
    }

    if (dlclose(handle) != 0) {
        const char* error = dlerror();
        std::cerr << "close_result=failure error="
                  << (error == nullptr ? "unknown" : error) << std::endl;
        return 1;
    }

    std::cout << "load_result=success elapsed_us=" << elapsed.count() << std::endl;
    return 0;
}
