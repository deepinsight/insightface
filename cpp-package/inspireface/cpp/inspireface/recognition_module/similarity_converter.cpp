#include <mutex>
#include "similarity_converter.h"

namespace inspire {

SimilarityConverter* SimilarityConverter::instance = nullptr;
std::mutex SimilarityConverter::instanceMutex;

}  // namespace inspire
