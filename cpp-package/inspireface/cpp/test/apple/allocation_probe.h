#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) int IFTestAllocationObservationSupported(void);
__attribute__((visibility("default"))) void IFTestConsumeAllocation(void *memory);
__attribute__((visibility("default"))) void IFTestBeginAllocations(void);
__attribute__((visibility("default"))) uint64_t IFTestEndAllocations(void);
#ifdef __cplusplus
}
#endif
