#ifdef IF_TEST_FRAMEWORK
#include <InspireFace/inspireface.h>
#else
#include "inspireface.h"
#endif
int main(void) {
    HFUInt32 level = 0;
    HFImageStream stream = 0;
    if (HFQueryCAPILevel(&level) != HSUCCEED || level != HF_C_API_LEVEL) return 1;
    if (HFCreateImageStreamEmpty(&stream) != HSUCCEED || !stream) return 2;
    return HFReleaseImageStream(stream) == HSUCCEED ? 0 : 3;
}
