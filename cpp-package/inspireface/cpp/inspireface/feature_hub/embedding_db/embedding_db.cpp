#include "embedding_db.h"
#include "sqlite-vec.h"
#include "isf_check.h"
#include <algorithm>
#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace inspire {

std::unique_ptr<EmbeddingDB> EmbeddingDB::instance_ = nullptr;
std::mutex EmbeddingDB::instanceMutex_;

EmbeddingDB &EmbeddingDB::GetInstance() {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    INSPIREFACE_CHECK_MSG(instance_, "EmbeddingDB not initialized. Call Init() first.");
    return *instance_;
}

void EmbeddingDB::Init(const std::string &dbPath, size_t vectorDim, IdMode idMode) {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    INSPIREFACE_CHECK_MSG(!instance_, "EmbeddingDB already initialized");
    instance_.reset(new EmbeddingDB(dbPath, vectorDim, "cosine", idMode));
}

EmbeddingDB::EmbeddingDB(const std::string &dbPath, size_t vectorDim, const std::string &distanceMetric, IdMode idMode)
: vectorDim_(vectorDim), tableName_("vec_items"), idMode_(idMode) {
    int rc = sqlite3_auto_extension((void (*)())sqlite3_vec_init);
    CheckSQLiteError(rc, nullptr);

    // Open database
    rc = sqlite3_open(dbPath.c_str(), &db_);
    CheckSQLiteError(rc, db_);

    // Create vector table
    std::string createTableSQL = "CREATE VIRTUAL TABLE IF NOT EXISTS " + tableName_ + " USING vec0(embedding float[" + std::to_string(vectorDim_) +
                                 "] distance_metric=" + distanceMetric + ")";

    ExecuteSQL(createTableSQL);
    initialized_ = true;
}

EmbeddingDB::~EmbeddingDB() {
    if (db_) {
        sqlite3_close(db_);
    }
}

bool EmbeddingDB::InsertVector(const std::vector<float> &vector, int64_t &allocId) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    return InsertVector(0, vector, allocId);  // In auto-increment mode, the passed ID is ignored
}

bool EmbeddingDB::InsertVector(int64_t id, const std::vector<float> &vector, int64_t &allocId) {
    CheckVectorDimension(vector);

    sqlite3_stmt *stmt;
    std::string sql;

    if (idMode_ == IdMode::AUTO_INCREMENT) {
        sql = "INSERT INTO " + tableName_ + "(embedding) VALUES (?)";
    } else {
        sql = "INSERT INTO " + tableName_ + "(rowid, embedding) VALUES (?, ?)";
    }

    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    // CheckSQLiteError(rc, db_);
    if (rc != SQLITE_OK) {
        INSPIRE_LOGE("Failed to prepare statement: %s", sqlite3_errmsg(db_));
        sqlite3_finalize(stmt);
        return false;
    }

    if (idMode_ == IdMode::AUTO_INCREMENT) {
        sqlite3_bind_blob(stmt, 1, vector.data(), vector.size() * sizeof(float), SQLITE_STATIC);
    } else {
        sqlite3_bind_int64(stmt, 1, id);
        sqlite3_bind_blob(stmt, 2, vector.data(), vector.size() * sizeof(float), SQLITE_STATIC);
    }

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) {
        INSPIRE_LOGE("Failed to insert vector: %s", sqlite3_errmsg(db_));
        return false;
    }
    // CheckSQLiteError(rc == SQLITE_DONE ? SQLITE_OK : rc, db_);

    allocId = idMode_ == IdMode::AUTO_INCREMENT ? GetLastInsertRowId() : id;
    return true;
}

std::vector<float> EmbeddingDB::GetVector(int64_t id) const {
    std::lock_guard<std::mutex> lock(dbMutex_);

    sqlite3_stmt *stmt;
    std::string sql = "SELECT embedding FROM " + tableName_ + " WHERE rowid = ?";

    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    CheckSQLiteError(rc, db_);

    sqlite3_bind_int64(stmt, 1, id);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        // throw std::runtime_error("Vector with id " + std::to_string(id) + " not found");
        return {};
    }

    const float *blob_data = static_cast<const float *>(sqlite3_column_blob(stmt, 0));
    size_t blob_size = sqlite3_column_bytes(stmt, 0) / sizeof(float);
    std::vector<float> result(blob_data, blob_data + blob_size);

    sqlite3_finalize(stmt);
    return result;
}

std::vector<int64_t> EmbeddingDB::BatchInsertVectors(const std::vector<VectorData> &vectors) {
    ExecuteSQL("BEGIN");
    std::vector<int64_t> insertedIds;
    insertedIds.reserve(vectors.size());

    for (const auto &data : vectors) {
        int64_t id = 0;
        bool ret = InsertVector(data.id, data.vector, id);
        INSPIREFACE_CHECK_MSG(ret, "Failed to insert vector");
        insertedIds.push_back(id);
    }
    ExecuteSQL("COMMIT");

    return insertedIds;
}

std::vector<int64_t> EmbeddingDB::BatchInsertVectors(const std::vector<std::vector<float>> &vectors) {
    ExecuteSQL("BEGIN");
    std::vector<int64_t> insertedIds;
    insertedIds.reserve(vectors.size());

    for (const auto &vector : vectors) {
        int64_t id = 0;
        bool ret = InsertVector(0, vector, id);
        INSPIREFACE_CHECK_MSG(ret, "Failed to insert vector");
        insertedIds.push_back(id);
    }
    ExecuteSQL("COMMIT");

    return insertedIds;
}

int64_t EmbeddingDB::GetLastInsertRowId() const {
    return sqlite3_last_insert_rowid(db_);
}

void EmbeddingDB::UpdateVector(int64_t id, const std::vector<float> &newVector) {
    CheckVectorDimension(newVector);

    sqlite3_stmt *stmt;
    std::string sql = "UPDATE " + tableName_ + " SET embedding = ? WHERE rowid = ?";

    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    CheckSQLiteError(rc, db_);

    sqlite3_bind_blob(stmt, 1, newVector.data(), newVector.size() * sizeof(float), SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, id);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    INSPIREFACE_CHECK_MSG(rc == SQLITE_DONE, "Failed to update vector");
    if (sqlite3_changes(db_) == 0) {
        INSPIRE_LOGF("Vector with id %ld not found", id);
    }
}

void EmbeddingDB::DeleteVector(int64_t id) {
    sqlite3_stmt *stmt;
    std::string sql = "DELETE FROM " + tableName_ + " WHERE rowid = ?";

    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    CheckSQLiteError(rc, db_);

    sqlite3_bind_int64(stmt, 1, id);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    CheckSQLiteError(rc == SQLITE_DONE ? SQLITE_OK : rc, db_);
}

std::vector<FaceSearchResult> EmbeddingDB::SearchSimilarVectors(const std::vector<float> &queryVector, size_t top_k, float keep_similar_threshold,
                                                                bool return_feature) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    CheckVectorDimension(queryVector);

    sqlite3_stmt *stmt;
    std::string sql;
    if (return_feature) {
        sql =
          "SELECT rowid, embedding, 1.0 - distance as similarity "
          "FROM " +
          tableName_ +
          " "
          "WHERE embedding MATCH ? "
          "ORDER BY distance "
          "LIMIT ?";
    } else {
        sql =
          "SELECT rowid, 1.0 - distance as similarity "
          "FROM " +
          tableName_ +
          " "
          "WHERE embedding MATCH ? "
          "ORDER BY distance "
          "LIMIT ?";
    }

    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    CheckSQLiteError(rc, db_);

    sqlite3_bind_blob(stmt, 1, queryVector.data(), queryVector.size() * sizeof(float), SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, top_k);

    std::vector<FaceSearchResult> results;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        FaceSearchResult result;
        result.id = sqlite3_column_int64(stmt, 0);
        if (return_feature) {
            const float *blob_data = static_cast<const float *>(sqlite3_column_blob(stmt, 1));
            size_t blob_size = sqlite3_column_bytes(stmt, 1) / sizeof(float);
            result.feature.assign(blob_data, blob_data + blob_size);
            result.similarity = sqlite3_column_double(stmt, 2);
        } else {
            result.similarity = sqlite3_column_double(stmt, 1);
        }
        results.push_back(result);
    }

    sqlite3_finalize(stmt);
    CheckSQLiteError(rc == SQLITE_DONE ? SQLITE_OK : rc, db_);

    // Filter results whose similarity is below the threshold
    results.erase(std::remove_if(results.begin(), results.end(),
                                 [keep_similar_threshold](const FaceSearchResult &result) { return result.similarity < keep_similar_threshold; }),
                  results.end());

    return results;
}

int64_t EmbeddingDB::GetVectorCount() const {
    std::lock_guard<std::mutex> lock(dbMutex_);
    sqlite3_stmt *stmt;
    std::string sql = "SELECT COUNT(*) FROM " + tableName_;

    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    CheckSQLiteError(rc, db_);

    rc = sqlite3_step(stmt);
    CheckSQLiteError(rc == SQLITE_ROW ? SQLITE_OK : rc, db_);

    int64_t count = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);

    return count;
}

void EmbeddingDB::CheckVectorDimension(const std::vector<float> &vector) const {
    INSPIREFACE_CHECK_MSG(vector.size() == vectorDim_,
                          ("Vector dimension mismatch. Expected: " + std::to_string(vectorDim_) + ", Got: " + std::to_string(vector.size())).c_str());
}

void EmbeddingDB::ExecuteSQL(const std::string &sql) {
    std::lock_guard<std::mutex> lock(dbMutex_);
    char *errMsg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errMsg);

    if (errMsg) {
        std::string error = errMsg;
        sqlite3_free(errMsg);
        INSPIREFACE_CHECK_MSG(false, ("SQL error: " + error).c_str());
    }

    CheckSQLiteError(rc, db_);
}

void EmbeddingDB::CheckSQLiteError(int rc, sqlite3 *db) {
    std::string error = db ? sqlite3_errmsg(db) : "SQLite error";
    INSPIREFACE_CHECK_MSG(rc == SQLITE_OK, error.c_str());
}

void EmbeddingDB::ShowTable() {
    if (!initialized_) {
        INSPIRE_LOGE("EmbeddingDB is not initialized");
        return;
    }
    std::lock_guard<std::mutex> lock(dbMutex_);
    sqlite3_stmt *stmt;
    std::string sql = "SELECT rowid, embedding FROM " + tableName_;

    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    CheckSQLiteError(rc, db_);

    // Print header
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "EmbeddingDB", "=== Table Content ===");
    __android_log_print(ANDROID_LOG_INFO, "EmbeddingDB", "ID | Vector (first 5 elements)");
    __android_log_print(ANDROID_LOG_INFO, "EmbeddingDB", "------------------------");
#else
    printf("=== Table Content ===\n");
    printf("ID | Vector (first 5 elements)\n");
    printf("------------------------\n");
#endif

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t id = sqlite3_column_int64(stmt, 0);
        const float *vector_data = static_cast<const float *>(sqlite3_column_blob(stmt, 1));
        size_t vector_size = std::min(size_t(5), sqlite3_column_bytes(stmt, 1) / sizeof(float));

        std::string vector_str;
        for (size_t i = 0; i < vector_size; ++i) {
            vector_str += std::to_string(vector_data[i]);
            if (i < vector_size - 1)
                vector_str += ", ";
        }
        vector_str += "...";

#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "EmbeddingDB", "%lld | %s", id, vector_str.c_str());
#else
        printf("%lld | %s\n", id, vector_str.c_str());
#endif
    }

    sqlite3_finalize(stmt);
}

std::vector<int64_t> EmbeddingDB::GetAllIds() {
    if (!initialized_) {
        INSPIRE_LOGE("EmbeddingDB is not initialized");
        return {};
    }
    std::lock_guard<std::mutex> lock(dbMutex_);
    std::vector<int64_t> ids;

    sqlite3_stmt *stmt;
    std::string sql = "SELECT rowid FROM " + tableName_;

    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    CheckSQLiteError(rc, db_);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ids.push_back(sqlite3_column_int64(stmt, 0));
    }

    sqlite3_finalize(stmt);
    return ids;
}

}  // namespace inspire