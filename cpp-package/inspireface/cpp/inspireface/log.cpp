/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include "log.h"

namespace inspire {

// Static Logger initialization
LogManager* LogManager::instance = nullptr;
std::mutex LogManager::mutex;

}  // namespace inspire