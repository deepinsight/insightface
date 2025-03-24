/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIREFACE_ENVIRO_H
#define INSPIREFACE_ENVIRO_H

#include <string>

/**
 * @class Enviro
 * @brief A singleton class for managing environment-specific configurations.
 *
 * This class provides methods to set and get various environment-specific configurations
 * such as the resource pack name, test result directory, and runtime full path.
 */
class Enviro {
public:
    static Enviro &getInstance() {
        static Enviro instance;
        return instance;
    }

    Enviro(Enviro const &) = delete;
    void operator=(Enviro const &) = delete;

    std::string getPackName() const {
        return packName;
    }

    void setPackName(const std::string &name) {
        packName = name;
    }

    const std::string &getTestResDir() const {
        return testResDir;
    }

    void setTestResDir(const std::string &dir) {
        Enviro::testResDir = dir;
    }

    const std::string &getTestRuntimeFullPath() const {
        return runtimeFullPath;
    }

    void setTestRuntimeFullPath(const std::string &path) {
        Enviro::runtimeFullPath = path;
    }

private:
    Enviro() {}

    std::string packName{"Pikachu"};
    std::string testResDir{"test_res"};
    std::string runtimeFullPath;
};

#endif  // INSPIREFACE_ENVIRO_H
