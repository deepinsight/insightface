#include "resource_manage.h"

std::unique_ptr<inspire::ResourceManager> inspire::ResourceManager::instance;
std::mutex inspire::ResourceManager::mutex;