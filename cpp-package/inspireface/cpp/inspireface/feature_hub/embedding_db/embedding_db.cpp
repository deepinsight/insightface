#include "embedding_db.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "isf_check.h"
#include "sqlite-vec.h"

#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace inspire {

std::shared_ptr<EmbeddingDB> EmbeddingDB::instance_ = nullptr;
std::mutex EmbeddingDB::instanceMutex_;

std::shared_ptr<EmbeddingDB> EmbeddingDB::AcquireInstance() {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    return instance_;
}

EmbeddingDB &EmbeddingDB::GetInstance() {
    static thread_local std::shared_ptr<EmbeddingDB> pinned_instance;
    pinned_instance = AcquireInstance();
    INSPIREFACE_CHECK_MSG(pinned_instance, "EmbeddingDB not initialized. Call Init() first.");
    return *pinned_instance;
}

bool EmbeddingDB::Init(const std::string &dbPath, size_t vectorDim, IdMode idMode) {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    if (instance_) {
        INSPIRE_LOGW("EmbeddingDB already initialized, skipping duplicate initialization");
        return instance_->IsInitialized();
    }

    std::shared_ptr<EmbeddingDB> candidate(new EmbeddingDB(dbPath, vectorDim, "cosine", idMode));
    if (!candidate->IsInitialized()) {
        return false;
    }
    instance_ = std::move(candidate);
    return true;
}

void EmbeddingDB::Deinit() {
    std::shared_ptr<EmbeddingDB> retired;
    {
        std::lock_guard<std::mutex> lock(instanceMutex_);
        retired.swap(instance_);
    }
}

EmbeddingDB::EmbeddingDB(const std::string &dbPath, size_t vectorDim, const std::string &distanceMetric, IdMode idMode)
: vectorDim_(vectorDim), tableName_("vec_items"), idMode_(idMode) {
    if (vectorDim_ == 0) {
        INSPIRE_LOGE("EmbeddingDB vector dimension must be positive");
        return;
    }

    const int extension_status = sqlite3_auto_extension(reinterpret_cast<void (*)()>(sqlite3_vec_init));
    if (extension_status != SQLITE_OK) {
        INSPIRE_LOGE("Failed to register sqlite-vec extension: %d", extension_status);
        return;
    }

    const int open_status = sqlite3_open(dbPath.c_str(), &db_);
    if (open_status != SQLITE_OK) {
        INSPIRE_LOGE("Failed to open embedding database '%s': %s", dbPath.c_str(), db_ ? sqlite3_errmsg(db_) : "unknown error");
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
        return;
    }

    const std::string create_table_sql =
      "CREATE VIRTUAL TABLE IF NOT EXISTS " + tableName_ + " USING vec0(embedding float[" + std::to_string(vectorDim_) +
      "] distance_metric=" + distanceMetric + ")";
    if (!ExecuteSQLUnlocked(create_table_sql) || !ValidateSchemaUnlocked()) {
        sqlite3_close(db_);
        db_ = nullptr;
        return;
    }
    initialized_ = true;
}

EmbeddingDB::~EmbeddingDB() {
    std::lock_guard<std::mutex> lock(dbMutex_);
    initialized_ = false;
    if (db_) {
        const int status = sqlite3_close(db_);
        if (status != SQLITE_OK) {
            INSPIRE_LOGE("Failed to close embedding database: %d", status);
        }
        db_ = nullptr;
    }
}

bool EmbeddingDB::IsValidVector(const std::vector<float> &vector) const {
    if (vector.size() != vectorDim_) {
        INSPIRE_LOGE("Vector dimension mismatch. Expected: %zu, Got: %zu", vectorDim_, vector.size());
        return false;
    }
    for (const float value : vector) {
        if (!std::isfinite(value)) {
            INSPIRE_LOGE("Feature vector contains a non-finite value");
            return false;
        }
    }
    return true;
}

bool EmbeddingDB::InsertVector(const std::vector<float> &vector, int64_t &allocId) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    return InsertVectorUnlocked(0, vector, allocId);
}

bool EmbeddingDB::InsertVector(int64_t id, const std::vector<float> &vector, int64_t &allocId) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    return InsertVectorUnlocked(id, vector, allocId);
}

bool EmbeddingDB::InsertVectorUnlocked(int64_t id, const std::vector<float> &vector, int64_t &allocId) {
    allocId = -1;
    if (!initialized_ || !db_ || !IsValidVector(vector)) {
        return false;
    }

    const std::string sql = idMode_ == IdMode::AUTO_INCREMENT ? "INSERT INTO " + tableName_ + "(embedding) VALUES (?)"
                                                              : "INSERT INTO " + tableName_ + "(rowid, embedding) VALUES (?, ?)";
    sqlite3_stmt *statement = nullptr;
    int status = sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement, nullptr);
    if (status != SQLITE_OK) {
        INSPIRE_LOGE("Failed to prepare vector insert: %s", sqlite3_errmsg(db_));
        sqlite3_finalize(statement);
        return false;
    }

    if (idMode_ == IdMode::AUTO_INCREMENT) {
        status = sqlite3_bind_blob(statement, 1, vector.data(), static_cast<int>(vector.size() * sizeof(float)), SQLITE_TRANSIENT);
    } else {
        status = sqlite3_bind_int64(statement, 1, id);
        if (status == SQLITE_OK) {
            status = sqlite3_bind_blob(statement, 2, vector.data(), static_cast<int>(vector.size() * sizeof(float)), SQLITE_TRANSIENT);
        }
    }
    if (status == SQLITE_OK) {
        status = sqlite3_step(statement);
    }
    sqlite3_finalize(statement);
    if (status != SQLITE_DONE) {
        INSPIRE_LOGE("Failed to insert vector: %s", sqlite3_errmsg(db_));
        return false;
    }

    allocId = idMode_ == IdMode::AUTO_INCREMENT ? sqlite3_last_insert_rowid(db_) : id;
    return true;
}

std::vector<float> EmbeddingDB::GetVector(int64_t id) const {
    std::vector<float> vector;
    GetVector(id, vector);
    return vector;
}

bool EmbeddingDB::GetVector(int64_t id, std::vector<float> &vector) const {
    std::lock_guard<std::mutex> lock(dbMutex_);
    vector.clear();
    if (!initialized_ || !db_) {
        return false;
    }

    sqlite3_stmt *statement = nullptr;
    const std::string sql = "SELECT embedding FROM " + tableName_ + " WHERE rowid = ?";
    int status = sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement, nullptr);
    if (status == SQLITE_OK) {
        status = sqlite3_bind_int64(statement, 1, id);
    }
    if (status == SQLITE_OK) {
        status = sqlite3_step(statement);
    }
    if (status != SQLITE_ROW) {
        if (status != SQLITE_DONE) {
            INSPIRE_LOGE("Failed to read vector %lld: %s", static_cast<long long>(id), sqlite3_errmsg(db_));
        }
        sqlite3_finalize(statement);
        return false;
    }

    const float *blob_data = static_cast<const float *>(sqlite3_column_blob(statement, 0));
    const size_t blob_size = static_cast<size_t>(sqlite3_column_bytes(statement, 0)) / sizeof(float);
    if (!blob_data || blob_size != vectorDim_) {
        INSPIRE_LOGE("Stored vector %lld has an invalid payload", static_cast<long long>(id));
        sqlite3_finalize(statement);
        return false;
    }
    vector.assign(blob_data, blob_data + blob_size);
    sqlite3_finalize(statement);
    return true;
}

std::vector<int64_t> EmbeddingDB::BatchInsertVectors(const std::vector<VectorData> &vectors) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    std::vector<int64_t> inserted_ids;
    if (!initialized_ || !db_ || !ExecuteSQLUnlocked("BEGIN IMMEDIATE")) {
        return inserted_ids;
    }

    inserted_ids.reserve(vectors.size());
    for (const VectorData &data : vectors) {
        int64_t allocated_id = -1;
        if (!InsertVectorUnlocked(data.id, data.vector, allocated_id)) {
            ExecuteSQLUnlocked("ROLLBACK");
            inserted_ids.clear();
            return inserted_ids;
        }
        inserted_ids.push_back(allocated_id);
    }
    if (!ExecuteSQLUnlocked("COMMIT")) {
        ExecuteSQLUnlocked("ROLLBACK");
        inserted_ids.clear();
    }
    return inserted_ids;
}

std::vector<int64_t> EmbeddingDB::BatchInsertVectors(const std::vector<std::vector<float>> &vectors) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    std::vector<int64_t> inserted_ids;
    if (!initialized_ || !db_ || !ExecuteSQLUnlocked("BEGIN IMMEDIATE")) {
        return inserted_ids;
    }

    inserted_ids.reserve(vectors.size());
    for (const std::vector<float> &vector : vectors) {
        int64_t allocated_id = -1;
        if (!InsertVectorUnlocked(0, vector, allocated_id)) {
            ExecuteSQLUnlocked("ROLLBACK");
            inserted_ids.clear();
            return inserted_ids;
        }
        inserted_ids.push_back(allocated_id);
    }
    if (!ExecuteSQLUnlocked("COMMIT")) {
        ExecuteSQLUnlocked("ROLLBACK");
        inserted_ids.clear();
    }
    return inserted_ids;
}

bool EmbeddingDB::UpdateVector(int64_t id, const std::vector<float> &newVector) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    if (!initialized_ || !db_ || !IsValidVector(newVector)) {
        return false;
    }

    sqlite3_stmt *statement = nullptr;
    const std::string sql = "UPDATE " + tableName_ + " SET embedding = ? WHERE rowid = ?";
    int status = sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement, nullptr);
    if (status == SQLITE_OK) {
        status = sqlite3_bind_blob(statement, 1, newVector.data(), static_cast<int>(newVector.size() * sizeof(float)), SQLITE_TRANSIENT);
    }
    if (status == SQLITE_OK) {
        status = sqlite3_bind_int64(statement, 2, id);
    }
    if (status == SQLITE_OK) {
        status = sqlite3_step(statement);
    }
    sqlite3_finalize(statement);
    if (status != SQLITE_DONE) {
        INSPIRE_LOGE("Failed to update vector %lld: %s", static_cast<long long>(id), sqlite3_errmsg(db_));
        return false;
    }
    return sqlite3_changes(db_) > 0;
}

bool EmbeddingDB::DeleteVector(int64_t id) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    if (!initialized_ || !db_) {
        return false;
    }

    sqlite3_stmt *statement = nullptr;
    const std::string sql = "DELETE FROM " + tableName_ + " WHERE rowid = ?";
    int status = sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement, nullptr);
    if (status == SQLITE_OK) {
        status = sqlite3_bind_int64(statement, 1, id);
    }
    if (status == SQLITE_OK) {
        status = sqlite3_step(statement);
    }
    sqlite3_finalize(statement);
    if (status != SQLITE_DONE) {
        INSPIRE_LOGE("Failed to delete vector %lld: %s", static_cast<long long>(id), sqlite3_errmsg(db_));
        return false;
    }
    return sqlite3_changes(db_) > 0;
}

std::vector<FaceSearchResult> EmbeddingDB::SearchSimilarVectors(const std::vector<float> &queryVector, size_t top_k,
                                                                float keep_similar_threshold, bool return_feature) {
    std::vector<FaceSearchResult> results;
    SearchSimilarVectors(queryVector, results, top_k, keep_similar_threshold, return_feature);
    return results;
}

bool EmbeddingDB::SearchSimilarVectors(const std::vector<float> &queryVector, std::vector<FaceSearchResult> &results, size_t top_k,
                                       float keep_similar_threshold, bool return_feature) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    results.clear();
    if (!initialized_ || !db_ || top_k == 0 || !IsValidVector(queryVector) || !std::isfinite(keep_similar_threshold)) {
        return false;
    }

    const std::string sql = return_feature
                              ? "SELECT rowid, embedding, 1.0 - distance as similarity FROM " + tableName_ +
                                  " WHERE embedding MATCH ? ORDER BY distance LIMIT ?"
                              : "SELECT rowid, 1.0 - distance as similarity FROM " + tableName_ +
                                  " WHERE embedding MATCH ? ORDER BY distance LIMIT ?";
    sqlite3_stmt *statement = nullptr;
    int status = sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement, nullptr);
    if (status == SQLITE_OK) {
        status = sqlite3_bind_blob(statement, 1, queryVector.data(), static_cast<int>(queryVector.size() * sizeof(float)), SQLITE_TRANSIENT);
    }
    if (status == SQLITE_OK) {
        status = sqlite3_bind_int64(statement, 2, static_cast<sqlite3_int64>(top_k));
    }
    if (status != SQLITE_OK) {
        INSPIRE_LOGE("Failed to prepare vector search: %s", sqlite3_errmsg(db_));
        sqlite3_finalize(statement);
        return false;
    }

    while ((status = sqlite3_step(statement)) == SQLITE_ROW) {
        FaceSearchResult result{-1, -1.0, {}};
        result.id = sqlite3_column_int64(statement, 0);
        if (return_feature) {
            const float *blob_data = static_cast<const float *>(sqlite3_column_blob(statement, 1));
            const size_t blob_size = static_cast<size_t>(sqlite3_column_bytes(statement, 1)) / sizeof(float);
            if (!blob_data || blob_size != vectorDim_) {
                INSPIRE_LOGE("Search returned an invalid stored vector payload");
                sqlite3_finalize(statement);
                results.clear();
                return false;
            }
            result.feature.assign(blob_data, blob_data + blob_size);
            result.similarity = sqlite3_column_double(statement, 2);
        } else {
            result.similarity = sqlite3_column_double(statement, 1);
        }
        if (!std::isfinite(result.similarity)) {
            INSPIRE_LOGE("Search returned a non-finite similarity");
            sqlite3_finalize(statement);
            results.clear();
            return false;
        }
        if (result.similarity >= keep_similar_threshold) {
            results.push_back(std::move(result));
        }
    }
    sqlite3_finalize(statement);
    if (status != SQLITE_DONE) {
        INSPIRE_LOGE("Vector search failed: %s", sqlite3_errmsg(db_));
        results.clear();
        return false;
    }
    return true;
}

int64_t EmbeddingDB::GetVectorCount() const {
    std::lock_guard<std::mutex> lock(dbMutex_);
    if (!initialized_ || !db_) {
        return -1;
    }

    sqlite3_stmt *statement = nullptr;
    const std::string sql = "SELECT COUNT(*) FROM " + tableName_;
    int status = sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement, nullptr);
    if (status == SQLITE_OK) {
        status = sqlite3_step(statement);
    }
    if (status != SQLITE_ROW) {
        INSPIRE_LOGE("Failed to count vectors: %s", sqlite3_errmsg(db_));
        sqlite3_finalize(statement);
        return -1;
    }
    const int64_t count = sqlite3_column_int64(statement, 0);
    sqlite3_finalize(statement);
    return count;
}

bool EmbeddingDB::ExecuteSQLUnlocked(const std::string &sql) {
    if (!db_) {
        return false;
    }
    char *error_message = nullptr;
    const int status = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &error_message);
    if (status != SQLITE_OK) {
        INSPIRE_LOGE("SQL error: %s", error_message ? error_message : sqlite3_errmsg(db_));
        sqlite3_free(error_message);
        return false;
    }
    sqlite3_free(error_message);
    return true;
}

bool EmbeddingDB::ValidateSchemaUnlocked() const {
    sqlite3_stmt *statement = nullptr;
    const char *sql = "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?";
    int status = sqlite3_prepare_v2(db_, sql, -1, &statement, nullptr);
    if (status == SQLITE_OK) {
        status = sqlite3_bind_text(statement, 1, tableName_.c_str(), -1, SQLITE_TRANSIENT);
    }
    if (status == SQLITE_OK) {
        status = sqlite3_step(statement);
    }
    if (status != SQLITE_ROW) {
        INSPIRE_LOGE("Failed to read embedding database schema: %s", sqlite3_errmsg(db_));
        sqlite3_finalize(statement);
        return false;
    }

    const unsigned char *schema_text = sqlite3_column_text(statement, 0);
    const std::string schema = schema_text ? reinterpret_cast<const char *>(schema_text) : std::string();
    const std::string expected_dimension = "float[" + std::to_string(vectorDim_) + "]";
    const bool valid = schema.find("vec0") != std::string::npos && schema.find(expected_dimension) != std::string::npos;
    sqlite3_finalize(statement);
    if (!valid) {
        INSPIRE_LOGE("Embedding database schema does not match the configured %zu-dimensional vectors", vectorDim_);
    }
    return valid;
}

bool EmbeddingDB::ShowTable() {
    std::lock_guard<std::mutex> lock(dbMutex_);
    if (!initialized_ || !db_) {
        INSPIRE_LOGE("EmbeddingDB is not initialized");
        return false;
    }

    sqlite3_stmt *statement = nullptr;
    const std::string sql = "SELECT rowid, embedding FROM " + tableName_;
    int status = sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement, nullptr);
    if (status != SQLITE_OK) {
        INSPIRE_LOGE("Failed to prepare table view: %s", sqlite3_errmsg(db_));
        sqlite3_finalize(statement);
        return false;
    }

#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "EmbeddingDB", "=== Table Content ===");
    __android_log_print(ANDROID_LOG_INFO, "EmbeddingDB", "ID | Vector (first 5 elements)");
    __android_log_print(ANDROID_LOG_INFO, "EmbeddingDB", "------------------------");
#else
    std::printf("=== Table Content ===\n");
    std::printf("ID | Vector (first 5 elements)\n");
    std::printf("------------------------\n");
#endif

    while ((status = sqlite3_step(statement)) == SQLITE_ROW) {
        const int64_t id = sqlite3_column_int64(statement, 0);
        const float *vector_data = static_cast<const float *>(sqlite3_column_blob(statement, 1));
        const size_t vector_size = std::min(size_t(5), static_cast<size_t>(sqlite3_column_bytes(statement, 1)) / sizeof(float));
        std::string vector_string;
        for (size_t i = 0; i < vector_size; ++i) {
            vector_string += std::to_string(vector_data[i]);
            if (i + 1 < vector_size) {
                vector_string += ", ";
            }
        }
        vector_string += "...";
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "EmbeddingDB", "%lld | %s", static_cast<long long>(id), vector_string.c_str());
#else
        std::printf("%lld | %s\n", static_cast<long long>(id), vector_string.c_str());
#endif
    }
    sqlite3_finalize(statement);
    if (status != SQLITE_DONE) {
        INSPIRE_LOGE("Failed while reading table: %s", sqlite3_errmsg(db_));
        return false;
    }
    return true;
}

std::vector<int64_t> EmbeddingDB::GetAllIds() {
    std::vector<int64_t> ids;
    GetAllIds(ids);
    return ids;
}

bool EmbeddingDB::GetAllIds(std::vector<int64_t> &ids) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    ids.clear();
    if (!initialized_ || !db_) {
        INSPIRE_LOGE("EmbeddingDB is not initialized");
        return false;
    }

    sqlite3_stmt *statement = nullptr;
    const std::string sql = "SELECT rowid FROM " + tableName_ + " ORDER BY rowid";
    int status = sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement, nullptr);
    if (status != SQLITE_OK) {
        INSPIRE_LOGE("Failed to prepare ID query: %s", sqlite3_errmsg(db_));
        sqlite3_finalize(statement);
        return false;
    }
    while ((status = sqlite3_step(statement)) == SQLITE_ROW) {
        ids.push_back(sqlite3_column_int64(statement, 0));
    }
    sqlite3_finalize(statement);
    if (status != SQLITE_DONE) {
        INSPIRE_LOGE("Failed while reading IDs: %s", sqlite3_errmsg(db_));
        ids.clear();
        return false;
    }
    return true;
}

}  // namespace inspire
