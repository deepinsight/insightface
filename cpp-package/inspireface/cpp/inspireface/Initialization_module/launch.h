// Created by tunm on 2024/04/17.
#pragma once
#ifndef INSPIREFACE_LAUNCH_H
#define INSPIREFACE_LAUNCH_H
#include "middleware/model_archive/inspire_archive.h"
#include <mutex>

#ifndef INSPIRE_API
#define INSPIRE_API
#endif

#define INSPIRE_LAUNCH inspire::Launch::GetInstance()

namespace inspire {

// The Launch class acts as the main entry point for the InspireFace system.
// It is responsible for loading static resources such as models, configurations, and parameters.
class INSPIRE_API Launch {
public:
    Launch(const Launch&) = delete;             // Delete the copy constructor to prevent copying.
    Launch& operator=(const Launch&) = delete;  // Delete the assignment operator to prevent assignment.

    // Retrieves the singleton instance of Launch, ensuring that only one instance exists.
    static std::shared_ptr<Launch> GetInstance();

    // Loads the necessary resources from a specified path.
    // Returns an integer status code: 0 on success, non-zero on failure.
    int32_t Load(const std::string &path);

    // Provides access to the loaded InspireArchive instance.
    InspireArchive& getMArchive();

    // Checks if the resources have been successfully loaded.
    bool isMLoad() const;

    // Unloads the resources and resets the system to its initial state.
    void Unload();

private:
    Launch() : m_load_(false) {} ///< Private constructor for the singleton pattern.

    static std::mutex mutex_;                         ///< Mutex for synchronizing access to the singleton instance.
    static std::shared_ptr<Launch> instance_;         ///< The singleton instance of Launch.

    InspireArchive m_archive_;  ///< The archive containing all necessary resources.
    bool m_load_;               ///< Flag indicating whether the resources have been successfully loaded.
};


}   // namespace inspire

#endif //INSPIREFACE_LAUNCH_H
