/**
 * Created by Jingyu Yan
 * @date 2025-03-23
 */
#pragma once
#ifndef MODELLOADERTAR_CPP_MICROTAR_H
#define MODELLOADERTAR_CPP_MICROTAR_H

#ifndef INSPIRE_API
#define INSPIRE_API
#endif

#include <string>
#include <vector>
#include <memory>

namespace inspire {

enum {
    SARC_SUCCESS = 0,
    SARC_FAILURE = -1,
    SARC_OPEN_FAIL = -2,
    SARC_READ_FAIL = -3,
    SARC_WRITE_FAIL = -4,
    SARC_SEEK_FAIL = -5,
    SARC_BAD_CHKSUM = -6,
    SARC_NULL_RECORD = -7,
    SARC_NOTFOUND = -8,
    SARC_LOAD_FILE_FAIL = -9,
    SARC_NOT_LOAD = -10,
};

class INSPIRE_API CoreArchive {
public:
    explicit CoreArchive(const std::string& archiveFile);
    explicit CoreArchive();
    ~CoreArchive();

    // Copy construction and assignment operations are disabled
    CoreArchive(const CoreArchive&) = delete;
    CoreArchive& operator=(const CoreArchive&) = delete;

    // Enable mobile construction and assignment operations
    CoreArchive(CoreArchive&& other) noexcept;
    CoreArchive& operator=(CoreArchive&& other) noexcept;

    int32_t Reset(const std::string& archiveFile);
    std::vector<char>& GetFileContent(const std::string& filename);
    int32_t QueryLoadStatus() const;
    const std::vector<std::string>& GetSubfilesNames() const;
    void Close();
    void PrintSubFiles();

private:
    class Impl;
    std::unique_ptr<Impl> m_pImpl;

};  // class CoreArchive
}  // namespace inspire

#endif  // MODELLOADERTAR_CPP_MICROTAR_H