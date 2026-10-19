// Test-only dyld interposition. Never linked into either SDK framework.
#include "allocation_probe.h"

#include <TargetConditionals.h>
#include <malloc/malloc.h>
#include <stdlib.h>
static _Thread_local int observing;
static _Thread_local uint64_t allocations;
int IFTestAllocationObservationSupported(void) {
#if __has_feature(address_sanitizer) || TARGET_OS_IPHONE
    // ASan replaces allocation entry points; iOS static consumers do not support this dyld probe.
    return 0;
#else
    return 1;
#endif
}
void IFTestConsumeAllocation(void *memory) {
    if (memory) *(volatile unsigned char *)memory = 1;
}
void IFTestBeginAllocations(void) {
    allocations = 0;
    observing = 1;
}
uint64_t IFTestEndAllocations(void) {
    observing = 0;
    return allocations;
}
#define COUNT()                       \
    do {                              \
        if (observing) ++allocations; \
    } while (0)
static void *probe_malloc(size_t size) {
    COUNT();
    return malloc(size);
}
static void *probe_calloc(size_t count, size_t size) {
    COUNT();
    return calloc(count, size);
}
static void *probe_realloc(void *memory, size_t size) {
    COUNT();
    return realloc(memory, size);
}
static void *probe_zone_malloc(malloc_zone_t *zone, size_t size) {
    COUNT();
    return malloc_zone_malloc(zone, size);
}
static void *probe_zone_calloc(malloc_zone_t *zone, size_t count, size_t size) {
    COUNT();
    return malloc_zone_calloc(zone, count, size);
}
static void *probe_zone_realloc(malloc_zone_t *zone, void *memory, size_t size) {
    COUNT();
    return malloc_zone_realloc(zone, memory, size);
}
static void *probe_zone_memalign(malloc_zone_t *zone, size_t alignment, size_t size) {
    COUNT();
    return malloc_zone_memalign(zone, alignment, size);
}
#define INTERPOSE(replacement, original)  \
    __attribute__((used)) static struct { \
        const void *new_function;         \
        const void *old_function;         \
    } interpose_##original                \
        __attribute__((section("__DATA,__interpose"))) = {(const void *)&replacement, (const void *)&original}
#if !TARGET_OS_IPHONE
INTERPOSE(probe_malloc, malloc);
INTERPOSE(probe_calloc, calloc);
INTERPOSE(probe_realloc, realloc);
INTERPOSE(probe_zone_malloc, malloc_zone_malloc);
INTERPOSE(probe_zone_calloc, malloc_zone_calloc);
INTERPOSE(probe_zone_realloc, malloc_zone_realloc);
INTERPOSE(probe_zone_memalign, malloc_zone_memalign);

#endif
