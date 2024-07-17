//
// Created by tunm on 2024/4/17.
//

#include "launch.h"
#include "log.h"
#include "herror.h"


namespace inspire {

std::mutex Launch::mutex_;
std::shared_ptr<Launch> Launch::instance_ = nullptr;

InspireArchive& Launch::getMArchive() {
    return m_archive_;
}

std::shared_ptr<Launch> Launch::GetInstance() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!instance_) {
        instance_ = std::shared_ptr<Launch>(new Launch());
    }
    return instance_;
}

int32_t Launch::Load(const std::string &path) {
    if (!m_load_) {
        m_archive_.ReLoad(path);
        if (m_archive_.QueryStatus() == SARC_SUCCESS) {
            m_load_ = true;
            return HSUCCEED;
        } else {
            return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
        }
    } else {
        INSPIRE_LOGW("There is no need to call launch more than once, as subsequent calls will not affect the initialization.");
        return HSUCCEED;
    }
}

bool Launch::isMLoad() const {
    return m_load_;
}

void Launch::Unload() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (m_load_) {
        // Assuming InspireArchive has a method to clear its resources
        m_archive_.Release(); 
        m_load_ = false;
        INSPIRE_LOGI("All resources have been successfully unloaded and system is reset.");
    } else {
        INSPIRE_LOGW("Unload called but system was not loaded.");
    }
}

}   // namespace inspire