#include <inspireface.h>

int main() {
    const char* resourcePath = "test_res/pack/Pikachu";
    HResult ret = HFReloadInspireFace(resourcePath);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Failed to launch InspireFace: %d", ret);
        return 1;
    }

    // Switch to another resource
    ret = HFReloadInspireFace("test_res/pack/Megatron");
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Failed to reload InspireFace: %d", ret);
        return 1;
    }

    return 0;
}
