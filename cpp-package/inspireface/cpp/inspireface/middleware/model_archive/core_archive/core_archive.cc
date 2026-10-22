/**
 * Created by Jingyu Yan
 * @date 2025-03-23
 */
#include "core_archive.h"
#include "microtar/microtar.h"
#include "log.h"
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <iostream>

namespace inspire {

class CoreArchive::Impl {
public:
    struct ArchiveState {
        ~ArchiveState() {
            if (tar_open) {
                mtar_close(&tar);
            }
        }

        ArchiveState() = default;
        ArchiveState(const ArchiveState&) = delete;
        ArchiveState& operator=(const ArchiveState&) = delete;

        mtar_t tar = {};
        bool tar_open{false};
        int32_t load_status{SARC_NOT_LOAD};
        std::vector<std::string> subfiles_names;
        std::unordered_set<std::string> full_names;
        std::unordered_set<std::string> ambiguous_full_names;
        // An empty mapped value marks an ambiguous basename.
        std::unordered_map<std::string, std::string> basename_index;
        std::unordered_map<std::string, std::shared_ptr<std::vector<char>>> file_content_cache;
        std::mutex read_mutex;
    };

    Impl() : state_(std::make_shared<ArchiveState>()) {}

    explicit Impl(const std::string& archiveFile) : Impl() {
        Reset(archiveFile);
    }

    ~Impl() = default;

    int32_t Reset(const std::string& archiveFile) {
        auto replacement = std::make_shared<ArchiveState>();
        replacement->load_status = mtar_open(&replacement->tar, archiveFile.c_str(), "r");
        if (replacement->load_status != MTAR_ESUCCESS) {
            INSPIRE_LOGE("Invalid archive file: %d", replacement->load_status);
            Publish(replacement);
            return replacement->load_status;
        }
        replacement->tar_open = true;

        mtar_header_t header = {};
        while (true) {
            const int32_t header_status = mtar_read_header(&replacement->tar, &header);
            if (header_status == MTAR_ENULLRECORD) {
                replacement->load_status = SARC_SUCCESS;
                break;
            }
            if (header_status != MTAR_ESUCCESS) {
                INSPIRE_LOGE("Failed to scan archive header: %d", header_status);
                replacement->load_status = header_status;
                break;
            }

            // PAX/global metadata records may repeat the same synthetic name
            // (for example "././@PaxHeader"). They are not loadable files and
            // must not participate in model-name resolution.
            if (header.type == MTAR_TREG || header.type == 0) {
                const std::string full_name(header.name);
                if (full_name.empty()) {
                    INSPIRE_LOGE("Archive contains an empty regular filename");
                    replacement->load_status = SARC_FAILURE;
                    break;
                }
                const bool unique_full_name = replacement->full_names.emplace(full_name).second;
                if (!unique_full_name) {
                    // Keep scanning for compatibility with archives containing
                    // repeated metadata records rewritten as regular files,
                    // but never resolve an ambiguous exact name.
                    replacement->ambiguous_full_names.emplace(full_name);
                }
                replacement->subfiles_names.emplace_back(full_name);
                const std::string basename = Basename(full_name);
                const auto basename_entry = replacement->basename_index.find(basename);
                if (basename_entry == replacement->basename_index.end()) {
                    replacement->basename_index.emplace(basename, full_name);
                } else if (!unique_full_name || basename_entry->second != full_name) {
                    basename_entry->second.clear();
                }
            }

            const int32_t next_status = mtar_next(&replacement->tar);
            if (next_status != MTAR_ESUCCESS) {
                INSPIRE_LOGE("Failed to scan archive file: %d", next_status);
                replacement->load_status = next_status;
                break;
            }
        }

        const int32_t result = replacement->load_status;
        Publish(replacement);
        return result;
    }

    std::shared_ptr<const std::vector<char>> GetFileContentShared(const std::string& filename) const {
        const auto state = Acquire();
        if (!state || state->load_status != SARC_SUCCESS || filename.empty()) {
            return EmptyContent();
        }
        const std::string full_filename = ResolveFilename(*state, filename);
        if (full_filename.empty()) {
            return EmptyContent();
        }

        std::lock_guard<std::mutex> lock(state->read_mutex);
        const auto cached = state->file_content_cache.find(full_filename);
        if (cached != state->file_content_cache.end()) {
            return cached->second;
        }

        mtar_header_t header = {};
        int32_t status = mtar_find(&state->tar, full_filename.c_str(), &header);
        if (status != MTAR_ESUCCESS) {
            INSPIRE_LOGE("Failed to find archive file %s: %d", full_filename.c_str(), status);
            return EmptyContent();
        }
        auto content = std::make_shared<std::vector<char>>(header.size);
        if (header.size != 0) {
            status = mtar_read_data(&state->tar, content->data(), header.size);
            if (status != MTAR_ESUCCESS) {
                INSPIRE_LOGE("Failed to load archive file %s: %d", full_filename.c_str(), status);
                return EmptyContent();
            }
        }
        state->file_content_cache.emplace(full_filename, content);
        return content;
    }

    std::vector<char>& GetFileContent(const std::string& filename) {
        // Keep the legacy reference-returning API alive without exposing a
        // buffer that Close()/Reset() can immediately destroy.
        thread_local std::shared_ptr<const std::vector<char>> pinned_content;
        pinned_content = GetFileContentShared(filename);
        if (pinned_content->empty()) {
            // Do not const-cast the process-wide immutable empty sentinel: a
            // legacy caller may legally mutate the returned vector.
            thread_local std::vector<char> mutable_empty;
            mutable_empty.clear();
            return mutable_empty;
        }
        return const_cast<std::vector<char>&>(*pinned_content);
    }

    int32_t QueryLoadStatus() const {
        const auto state = Acquire();
        return state ? state->load_status : SARC_NOT_LOAD;
    }

    const std::vector<std::string>& GetSubfilesNames() const {
        // Preserve the reference until this thread's next call without keeping
        // the archive's FILE open after Close(), Reset(), or destruction.
        thread_local std::vector<std::string> pinned_names;
        const auto state = Acquire();
        if (state) {
            pinned_names = state->subfiles_names;
        } else {
            pinned_names.clear();
        }
        return pinned_names;
    }

    void Close() {
        Publish(std::make_shared<ArchiveState>());
    }

    void PrintSubFiles() {
        const auto state = Acquire();
        const size_t count = state ? state->subfiles_names.size() : 0;
        std::cout << "Subfiles: " << count << std::endl;
        if (state) {
            for (const auto& filename : state->subfiles_names) {
                std::cout << filename << std::endl;
            }
        }
    }

private:
    static std::string Basename(const std::string& filename) {
        const size_t separator = filename.find_last_of('/');
        return separator == std::string::npos ? filename : filename.substr(separator + 1);
    }

    static std::string ResolveFilename(const ArchiveState& state, const std::string& filename) {
        if (state.full_names.find(filename) != state.full_names.end() &&
            state.ambiguous_full_names.find(filename) == state.ambiguous_full_names.end()) {
            return filename;
        }
        // Directory-prefixed archives remain compatible, but only a complete,
        // unambiguous basename is accepted. Substring matching is forbidden.
        if (filename.find('/') == std::string::npos) {
            const auto basename = state.basename_index.find(filename);
            if (basename != state.basename_index.end()) {
                return basename->second;
            }
        }
        return {};
    }

    static std::shared_ptr<const std::vector<char>> EmptyContent() {
        static const auto empty = std::make_shared<const std::vector<char>>();
        return empty;
    }

    std::shared_ptr<ArchiveState> Acquire() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

    void Publish(std::shared_ptr<ArchiveState> state) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_.swap(state);
    }

    mutable std::mutex state_mutex_;
    std::shared_ptr<ArchiveState> state_;
};

CoreArchive::CoreArchive() : m_pImpl(std::make_unique<Impl>()) {}

CoreArchive::CoreArchive(const std::string& archiveFile) : m_pImpl(std::make_unique<Impl>(archiveFile)) {}

CoreArchive::~CoreArchive() = default;

CoreArchive::CoreArchive(CoreArchive&& other) noexcept = default;

CoreArchive& CoreArchive::operator=(CoreArchive&& other) noexcept = default;

int32_t CoreArchive::Reset(const std::string& archiveFile) {
    return m_pImpl->Reset(archiveFile);
}

std::shared_ptr<const std::vector<char>> CoreArchive::GetFileContentShared(const std::string& filename) const {
    return m_pImpl->GetFileContentShared(filename);
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
