//
// Created by Tunm-Air13 on 2023/10/11.
//

#include <iostream>
#include <iomanip>  // for std::setw
#include "sqlite_faces_manage.h"
#include "herror.h"

namespace inspire {


SQLiteFaceManage::SQLiteFaceManage() {

}

SQLiteFaceManage::~SQLiteFaceManage() {
    CloseDatabase();
    // Optionally, you can add logging here if needed:
    // LOG_INFO("SQLiteFaceManage object destroyed and database connection closed.");
}

struct SQLiteDeleter {
    void operator()(sqlite3* ptr) const {
        sqlite3_close(ptr);
    }
};


int32_t SQLiteFaceManage::OpenDatabase(const std::string &dbPath) {
    sqlite3* rawDb;
    if (sqlite3_open(dbPath.c_str(), &rawDb) != SQLITE_OK) {
        // Handle error
        return HERR_FT_HUB_OPEN_ERROR;
    }

    m_db_ = std::shared_ptr<sqlite3>(rawDb, SQLiteDeleter());

    // Check if the table exists
    const char* checkTableSQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='FaceFeatures';";
    sqlite3_stmt* stmt = nullptr;

    if (sqlite3_prepare_v2(m_db_.get(), checkTableSQL, -1, &stmt, nullptr) != SQLITE_OK) {
        INSPIRE_LOGE("Error checking for table existence: %s", sqlite3_errmsg(m_db_.get()));
        return HERR_FT_HUB_CHECK_TABLE_ERROR;  // Assuming you have this error code
    }

    int result = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    // If table doesn't exist, create it
    if (result != SQLITE_ROW) {
        return CreateTable();
    }

    return HSUCCEED;
}


int32_t SQLiteFaceManage::CloseDatabase() {
    if (!m_db_) {
//        LOGE("Attempted to close an already closed or uninitialized database.");
        return HERR_FT_HUB_NOT_OPENED;
    }

    // Reset the shared_ptr. This will decrease its reference count.
    // If this is the last reference, the database will be closed due to the custom deleter.
    m_db_.reset();

    // Optionally, log that the database was successfully closed
//    LOGD("Database successfully closed.");

    return HSUCCEED;
}

int32_t SQLiteFaceManage::CreateTable() {
    if (!m_db_) {
        INSPIRE_LOGE("Database is not opened. Please open the database first.");
        return HERR_FT_HUB_NOT_OPENED;  // Example error code for unopened database
    }

    const char* createTableSQL = R"(
    CREATE TABLE IF NOT EXISTS FaceFeatures (
        customId INTEGER PRIMARY KEY,
        tag TEXT,
        feature BLOB
    )
    )";

    char* errMsg = nullptr;
    int result = sqlite3_exec(m_db_.get(), createTableSQL, 0, 0, &errMsg);

    if (result != SQLITE_OK) {
        INSPIRE_LOGE("Error creating table: %s" , errMsg);
        sqlite3_free(errMsg);
        return result;
    }

//    LOGD("Table successfully created or already exists.");
    return SQLITE_OK;  // or SUCCESS_CODE, based on your error code system
}

int32_t SQLiteFaceManage::InsertFeature(const FaceFeatureInfo& info) {
    if (!m_db_) {
        INSPIRE_LOGE("Database is not opened. Please open the database first.");
        return HERR_FT_HUB_NOT_OPENED;  // Example error code for unopened database
    }

    const char* insertSQL = "INSERT INTO FaceFeatures (customId, tag, feature) VALUES (?, ?, ?)";
    sqlite3_stmt* stmt = nullptr;

    int result = sqlite3_prepare_v2(m_db_.get(), insertSQL, -1, &stmt, nullptr);
    if (result != SQLITE_OK) {
        INSPIRE_LOGE("Error preparing the SQL statement: %s", sqlite3_errmsg(m_db_.get()));
        return result;
    }

    // Binding values
    sqlite3_bind_int(stmt, 1, info.customId);
    sqlite3_bind_text(stmt, 2, info.tag.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_blob(stmt, 3, info.feature.data(), info.feature.size() * sizeof(float), SQLITE_STATIC);

    result = sqlite3_step(stmt);
    if (result != SQLITE_DONE) {
        INSPIRE_LOGE("Error inserting new feature: %s" , sqlite3_errmsg(m_db_.get()));
        sqlite3_finalize(stmt);
        return HERR_FT_HUB_INSERT_FAILURE;
    }

    // Clean up the statement
    sqlite3_finalize(stmt);

//    LOGD("Feature successfully inserted.");
    return SQLITE_OK;  // or SUCCESS_CODE, based on your error code system
}

int32_t SQLiteFaceManage::GetFeature(int32_t customId, FaceFeatureInfo& outInfo) {
    if (!m_db_) {
        INSPIRE_LOGE("Database is not opened. Please open the database first.");
        return HERR_FT_HUB_NOT_OPENED;
    }

    const char* selectSQL = "SELECT customId, tag, feature FROM FaceFeatures WHERE customId = ?";
    sqlite3_stmt* stmt = nullptr;

    int result = sqlite3_prepare_v2(m_db_.get(), selectSQL, -1, &stmt, nullptr);
    if (result != SQLITE_OK) {
        INSPIRE_LOGE("Error preparing the SQL statement: %s", sqlite3_errmsg(m_db_.get()));
        return HERR_FT_HUB_PREPARING_FAILURE;
    }

    // Bind the customId to the prepared statement
    sqlite3_bind_int(stmt, 1, customId);

    result = sqlite3_step(stmt);
    if (result == SQLITE_ROW) {
        outInfo.customId = sqlite3_column_int(stmt, 0);
        outInfo.tag = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        const void* blobData = sqlite3_column_blob(stmt, 2);
        int blobSize = sqlite3_column_bytes(stmt, 2) / sizeof(float);
        const float* begin = static_cast<const float*>(blobData);
        outInfo.feature = std::vector<float>(begin, begin + blobSize);

    } else if (result == SQLITE_DONE) {
        INSPIRE_LOGE("No feature found with customId: %d", customId);
        sqlite3_finalize(stmt);
        return HERR_FT_HUB_NO_RECORD_FOUND;  // Assuming you have an error code for record not found
    } else {
        INSPIRE_LOGE("Error executing the SQL statement: %s", sqlite3_errmsg(m_db_.get()));
        sqlite3_finalize(stmt);
        return HERR_FT_HUB_EXECUTING_FAILURE;
    }

    // Clean up the statement
    sqlite3_finalize(stmt);

    INSPIRE_LOGD("Feature successfully retrieved.");
    return HSUCCEED;
}

int32_t SQLiteFaceManage::DeleteFeature(int32_t customId) {
    if (!m_db_) {
        INSPIRE_LOGE("Database is not opened. Please open the database first.");
        return HERR_FT_HUB_NOT_OPENED;
    }

    const char* deleteSQL = "DELETE FROM FaceFeatures WHERE customId = ?";
    sqlite3_stmt* stmt = nullptr;

    int result = sqlite3_prepare_v2(m_db_.get(), deleteSQL, -1, &stmt, nullptr);
    if (result != SQLITE_OK) {
        INSPIRE_LOGE("Error preparing the SQL statement: %s", sqlite3_errmsg(m_db_.get()));
        return HERR_FT_HUB_PREPARING_FAILURE;
    }

    // Bind the customId to the prepared statement
    sqlite3_bind_int(stmt, 1, customId);

    result = sqlite3_step(stmt);
    if (result != SQLITE_DONE) {
        INSPIRE_LOGE("Error deleting feature with customId: %d, Error: %s", customId, sqlite3_errmsg(m_db_.get()));
        sqlite3_finalize(stmt);
        return HERR_FT_HUB_EXECUTING_FAILURE;
    }

    int changes = sqlite3_changes(m_db_.get());
    if (changes == 0) {
        INSPIRE_LOGE("No feature found with customId: %d. Nothing was deleted.", customId);
        sqlite3_finalize(stmt);
        return HERR_FT_HUB_NO_RECORD_FOUND;  // Assuming you have an error code for record not found
    }

    // Clean up the statement
    sqlite3_finalize(stmt);

//    LOGD("Feature with customId: %d successfully deleted.", customId);
    return HSUCCEED;
}

int32_t SQLiteFaceManage::UpdateFeature(const FaceFeatureInfo& info) {
    if (!m_db_) {
        INSPIRE_LOGE("Database is not opened. Please open the database first.");
        return HERR_FT_HUB_NOT_OPENED;
    }

    const char* updateSQL = "UPDATE FaceFeatures SET tag = ?, feature = ? WHERE customId = ?";
    sqlite3_stmt* stmt = nullptr;

    int result = sqlite3_prepare_v2(m_db_.get(), updateSQL, -1, &stmt, nullptr);
    if (result != SQLITE_OK) {
        INSPIRE_LOGE("Error preparing the SQL statement: %s", sqlite3_errmsg(m_db_.get()));
        return HERR_FT_HUB_PREPARING_FAILURE;
    }

    // Binding values
    sqlite3_bind_text(stmt, 1, info.tag.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_blob(stmt, 2, info.feature.data(), info.feature.size() * sizeof(float), SQLITE_STATIC);
    sqlite3_bind_int(stmt, 3, info.customId);

    result = sqlite3_step(stmt);
    if (result != SQLITE_DONE) {
        INSPIRE_LOGE("Error updating feature with customId: %d, Error: %s", info.customId, sqlite3_errmsg(m_db_.get()));
        sqlite3_finalize(stmt);
        return result;
    }

    int changes = sqlite3_changes(m_db_.get());
    if (changes == 0) {
        INSPIRE_LOGE("No feature found with customId: %d. Nothing was updated.", info.customId);
        sqlite3_finalize(stmt);
        return HERR_FT_HUB_NO_RECORD_FOUND;  // Assuming you have an error code for record not found
    }

    // Clean up the statement
    sqlite3_finalize(stmt);

//    LOGD("Feature with customId: %d successfully updated.", info.customId);
    return HSUCCEED;
}

int32_t SQLiteFaceManage::ViewTotal() {
    if (!m_db_) {
        INSPIRE_LOGE("Database is not opened. Please open the database first.");
        return HERR_FT_HUB_NOT_OPENED;
    }

    const char* selectSQL = "SELECT customId, tag FROM FaceFeatures";
    sqlite3_stmt* stmt = nullptr;

    int result = sqlite3_prepare_v2(m_db_.get(), selectSQL, -1, &stmt, nullptr);
    if (result != SQLITE_OK) {
        INSPIRE_LOGE("Error preparing the SQL statement: %s", sqlite3_errmsg(m_db_.get()));
        return result;
    }

    // Print table header
    std::cout << "+----------+-----------------------+\n";
    std::cout << "| customId | tag                   |\n";
    std::cout << "+----------+-----------------------+\n";

    while ((result = sqlite3_step(stmt)) == SQLITE_ROW) {
        int32_t customId = sqlite3_column_int(stmt, 0);
        const unsigned char* tag = sqlite3_column_text(stmt, 1);

        std::cout << "| " << std::setw(8) << customId << " | " << std::setw(21) << tag << " |\n";
    }
    std::cout << "+----------+-----------------------+\n";

    if (result != SQLITE_DONE) {
        INSPIRE_LOGE("Error executing the SQL statement: %s", sqlite3_errmsg(m_db_.get()));
        sqlite3_finalize(stmt);
        return HERR_FT_HUB_PREPARING_FAILURE;
    }

    // Clean up the statement
    sqlite3_finalize(stmt);

    INSPIRE_LOGD("Successfully displayed all records.");
    return HSUCCEED;
}

int32_t SQLiteFaceManage::GetTotalFeatures(std::vector<FaceFeatureInfo>& infoList) {
    if (!m_db_) {
        INSPIRE_LOGE("Database is not opened. Please open the database first.");
        return HERR_FT_HUB_NOT_OPENED;
    }

    const char* selectSQL = "SELECT customId, tag, feature FROM FaceFeatures";
    sqlite3_stmt* stmt = nullptr;

    int result = sqlite3_prepare_v2(m_db_.get(), selectSQL, -1, &stmt, nullptr);
    if (result != SQLITE_OK) {
        INSPIRE_LOGE("Error preparing the SQL statement: %s", sqlite3_errmsg(m_db_.get()));
        return HERR_FT_HUB_PREPARING_FAILURE;
    }

    while ((result = sqlite3_step(stmt)) == SQLITE_ROW) {
        FaceFeatureInfo featureInfo;

        featureInfo.customId = sqlite3_column_int(stmt, 0);
        featureInfo.tag = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        const void* blobData = sqlite3_column_blob(stmt, 2);
        int blobSize = sqlite3_column_bytes(stmt, 2) / sizeof(float);
        const float* begin = static_cast<const float*>(blobData);
        featureInfo.feature = std::vector<float>(begin, begin + blobSize);

        infoList.push_back(featureInfo);
    }

    if (result != SQLITE_DONE) {
        INSPIRE_LOGE("Error executing the SQL statement: %s", sqlite3_errmsg(m_db_.get()));
        sqlite3_finalize(stmt);
        return HERR_FT_HUB_EXECUTING_FAILURE;
    }

    // Clean up the statement
    sqlite3_finalize(stmt);

//    LOGD("Successfully retrieved all features.");
    return HSUCCEED;
}


}   // namespace hyper
