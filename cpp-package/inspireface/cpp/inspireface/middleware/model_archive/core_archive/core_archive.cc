/**
 * Created by Jingyu Yan
 * @date 2025-03-23
 */
#include "core_archive.h"
#include "microtar/microtar.h"
#include "log.h"
#include <unordered_map>
#include <iostream>

namespace inspire {

class CoreArchive::Impl {
public:
    Impl() : m_load_file_status_(SARC_NOT_LOAD), m_tar_(nullptr) {}

    explicit Impl(const std::string& archiveFile) : m_load_file_status_(SARC_NOT_LOAD), m_tar_(nullptr) {
        Reset(archiveFile);
    }

    ~Impl() {
        Close();
    }

    int32_t Reset(const std::string& archiveFile) {
        Close();
        std::vector<char>().swap(m_empty_);
        m_tar_ = std::make_shared<mtar_t>();
        m_load_file_status_ = mtar_open(m_tar_.get(), archiveFile.c_str(), "r");
        if (m_load_file_status_ != MTAR_ESUCCESS) {
            INSPIRE_LOGE("Invalid archive file: %d", m_load_file_status_);
            Close();
            m_tar_.reset();
            return m_load_file_status_;
        }
        mtar_header_t h;
        m_load_file_status_ = mtar_read_header(m_tar_.get(), &h);
        if (m_load_file_status_ != MTAR_ESUCCESS) {
            INSPIRE_LOGE("Error reading root from archive.");
            return m_load_file_status_;
        }
        //        m_file_archive_root_ = std::string(h.name);
        size_t index = 0;
        while ((mtar_read_header(m_tar_.get(), &h)) != MTAR_ENULLRECORD) {
            m_load_file_status_ = mtar_next(m_tar_.get());
            if (m_load_file_status_ != MTAR_ESUCCESS) {
                INSPIRE_LOGE("Failed to scan the file: %d", m_load_file_status_);
                Close();
                return m_load_file_status_;
            }
            m_subfiles_names_.emplace_back(h.name);
            index++;
        }
        return m_load_file_status_;
    }

    std::vector<char>& GetFileContent(const std::string& filename) {
        auto index = filenameFuzzyMatching(filename);
        if (index != std::string::npos) {
            auto fullFilename = m_subfiles_names_[index];
            auto ret = lazyReadFile(fullFilename);
            if (ret != MTAR_ESUCCESS) {
                INSPIRE_LOGE("Failed to load file");
            }
            return m_file_content_cache_map_[fullFilename];
        }
        return m_empty_;
    }

    int32_t QueryLoadStatus() const {
        return m_load_file_status_;
    }

    const std::vector<std::string>& GetSubfilesNames() const {
        return m_subfiles_names_;
    }

    void Close() {
        if (m_tar_ && m_tar_.get() != nullptr) {
            mtar_close(m_tar_.get());
        }
        m_tar_.reset();
        m_load_file_status_ = SARC_NOT_LOAD;
        m_subfiles_names_.clear();
        m_file_content_cache_map_.clear();
    }

    void PrintSubFiles() {
        std::cout << "Subfiles: " << m_subfiles_names_.size() << std::endl;
        for (int i = 0; i < m_subfiles_names_.size(); ++i) {
            std::cout << m_subfiles_names_[i] << std::endl;
        }
    }

private:
    size_t filenameFuzzyMatching(const std::string& filename) {
        for (size_t i = 0; i < m_subfiles_names_.size(); ++i) {
            if (m_subfiles_names_[i].find(filename) != std::string::npos) {
                return i;
            }
        }
        return std::string::npos;
    }

    int32_t lazyReadFile(const std::string& filename) {
        if (m_file_content_cache_map_.find(filename) != m_file_content_cache_map_.end()) {
            return MTAR_ESUCCESS;
        }
        mtar_header_t h;
        auto ret = mtar_find(m_tar_.get(), filename.c_str(), &h);
        if (ret == MTAR_ESUCCESS) {
            std::vector<char> content(h.size);
            ret = mtar_read_data(m_tar_.get(), content.data(), h.size);
            if (ret == MTAR_ESUCCESS) {
                m_file_content_cache_map_[filename] = std::move(content);  // Load and store the file contents
                return MTAR_ESUCCESS;
            } else {
                INSPIRE_LOGE("Failed to load file: %d", ret);
            }
        } else {
            INSPIRE_LOGE("Failed to find file: %d", ret);
        }

        return SARC_LOAD_FILE_FAIL;
    }

    std::string m_file_archive_root_;            ///< Archive file path
    std::vector<std::string> m_subfiles_names_;  ///< Name list of subfiles
    std::shared_ptr<mtar_t> m_tar_;              ///< mtar context
    int32_t m_load_file_status_;                 ///< Initiation status code

    std::vector<char> m_empty_;  ///< Const empty

    std::unordered_map<std::string, std::vector<char>> m_file_content_cache_map_;  ///< File buffer cache
};

CoreArchive::CoreArchive() : m_pImpl(std::make_unique<Impl>()) {}

CoreArchive::CoreArchive(const std::string& archiveFile) : m_pImpl(std::make_unique<Impl>(archiveFile)) {}

CoreArchive::~CoreArchive() = default;

CoreArchive::CoreArchive(CoreArchive&& other) noexcept = default;

CoreArchive& CoreArchive::operator=(CoreArchive&& other) noexcept = default;

int32_t CoreArchive::Reset(const std::string& archiveFile) {
    return m_pImpl->Reset(archiveFile);
}

std::vector<char>& CoreArchive::GetFileContent(const std::string& filename) {
    return m_pImpl->GetFileContent(filename);
}

int32_t CoreArchive::QueryLoadStatus() const {
    return m_pImpl->QueryLoadStatus();
}

const std::vector<std::string>& CoreArchive::GetSubfilesNames() const {
    return m_pImpl->GetSubfilesNames();
}

void CoreArchive::Close() {
    m_pImpl->Close();
}

void CoreArchive::PrintSubFiles() {
    m_pImpl->PrintSubFiles();
}

}  // namespace inspire