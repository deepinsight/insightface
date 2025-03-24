#ifndef INSPIRE_FACE_SYSTEM_H
#define INSPIRE_FACE_SYSTEM_H

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <string>
#include <initializer_list>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace inspire {
namespace os {

template <typename... Args>
inline std::string PathJoin(Args... args) {
    std::string result;
#if defined(_WIN32)
    const char sep = '\\';
#else
    const char sep = '/';
#endif
    std::vector<std::string> paths = {args...};
    bool first = true;
    for (const auto& p : paths) {
        if (p.empty())
            continue;

        if (first) {
            result = p;
            first = false;
            continue;
        }

        if (result.back() != sep)
            result += sep;
        result += p;
    }

    return result;
}

inline std::pair<std::string, std::string> PathSplit(const std::string& path) {
    std::string directory, filename;
    size_t pos = path.rfind('/');

#ifdef _WIN32
    size_t backslash_pos = path.rfind('\\');
    if ((pos == std::string::npos) || (backslash_pos != std::string::npos && backslash_pos > pos)) {
        pos = backslash_pos;
    }
#endif

    if (pos == std::string::npos) {
        directory = "";
        filename = path;
    } else if (pos == 0) {
        directory = path.substr(0, 1);
        filename = path.substr(1);
    } else {
        directory = path.substr(0, pos);
        filename = path.substr(pos + 1);
    }

    return {directory, filename};
}

inline std::pair<std::string, std::string> SplitExt(const std::string& path) {
    std::string basename, extension;

    size_t pos = path.rfind('.');

    if (pos == std::string::npos) {
        basename = path;
        extension = "";
    } else {
        basename = path.substr(0, pos);
        extension = path.substr(pos);
    }

    return {basename, extension};
}

inline std::string Dirname(const std::string& path) {
    return PathSplit(path).first;
}

inline std::string Basename(const std::string& path) {
    return PathSplit(path).second;
}

#if defined(_WIN32)
inline std::wstring Utf8ToWideChar(const std::string& utf8str) {
    int size_required = MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), utf8str.size(), NULL, 0);
    std::wstring ws_translated_str(size_required, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), utf8str.size(), &ws_translated_str[0], size_required);
    return ws_translated_str;
}
#endif

inline bool IsExists(const std::string& path) {
#if defined(_WIN32)
    std::wstring ws_path = Utf8ToWideChar(path);
    return _waccess(ws_path.c_str(), 0) == 0;
#else
    return access(path.c_str(), F_OK) == 0;
#endif
}

inline bool IsDir(const std::string& path) {
#if defined(_WIN32)
    return GetFileAttributesA(path.c_str()) == FILE_ATTRIBUTE_DIRECTORY;
#else
    struct stat s;
    if (stat(path.c_str(), &s) != 0)
        return false;
    return S_ISDIR(s.st_mode);
#endif
}

inline bool IsFile(const std::string& path) {
#if defined(_WIN32)
    return GetFileAttributesA(path.c_str()) == FILE_ATTRIBUTE_ARCHIVE;
#else
    struct stat s;
    if (stat(path.c_str(), &s) != 0)
        return false;
    return S_ISREG(s.st_mode);
#endif
}

}  // namespace os
}  // namespace inspire

#endif  // INSPIRE_FACE_SYSTEM_H
