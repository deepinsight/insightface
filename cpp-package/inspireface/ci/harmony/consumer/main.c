#include <inspireface.h>

int main(void) {
    HFInspireFaceVersion version = {0};
    return HFQueryInspireFaceVersion(&version);
}
