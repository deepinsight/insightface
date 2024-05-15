//
// Created by Tunm-Air13 on 2023/10/11.
//

#pragma once
#ifndef HYPERFACEREPO_SQLITEFACEMANAGE_H
#define HYPERFACEREPO_SQLITEFACEMANAGE_H

#include "data_type.h"
#include "log.h"
#include "sqlite3.h"  // Include the SQLite3 header
#include <vector>
#include <string>
#include "memory"

namespace inspire {

/**
 * @struct FaceFeatureInfo
 * @brief Structure to represent information about a facial feature.
 */
typedef struct {
    int32_t customId;              ///< Custom identifier for the feature.
    std::string tag;              ///< Tag associated with the feature.
    std::vector<float> feature;   ///< Vector of floats representing the feature.
} FaceFeatureInfo;

/**
 * @class SQLiteFaceManage
 * @brief Class for managing facial features using SQLite database.
 *
 * This class provides methods to open, close, create tables, insert, retrieve, delete, and update
 * facial features in an SQLite database. It also allows viewing the total number of features in the database.
 */
class INSPIRE_API SQLiteFaceManage {
public:
    /**
     * @brief Constructor for SQLiteFaceManage class.
     */
    SQLiteFaceManage();

    /**
     * @brief Destructor for SQLiteFaceManage class.
     */
    ~SQLiteFaceManage();

    /**
     * @brief Opens an SQLite database at the specified path.
     *
     * @param dbPath Path to the SQLite database file.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t OpenDatabase(const std::string& dbPath);

    /**
     * @brief Closes the currently open SQLite database.
     *
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t CloseDatabase();

    /**
     * @brief Creates an SQLite table for storing facial features if it doesn't exist.
     *
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t CreateTable();

    /**
     * @brief Inserts a facial feature into the SQLite database.
     *
     * @param info Information about the facial feature to be inserted.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InsertFeature(const FaceFeatureInfo& info);

    /**
     * @brief Retrieves information about a facial feature from the SQLite database by custom ID.
     *
     * @param customId Custom identifier of the facial feature to retrieve.
     * @param outInfo Output parameter to store the retrieved feature information.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t GetFeature(int32_t customId, FaceFeatureInfo& outInfo);

    /**
     * @brief Deletes a facial feature from the SQLite database by custom ID.
     *
     * @param customId Custom identifier of the facial feature to delete.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t DeleteFeature(int32_t customId);

    /**
     * @brief Updates a facial feature in the SQLite database.
     *
     * @param info Updated information about the facial feature.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t UpdateFeature(const FaceFeatureInfo& info);

    /**
     * @brief Retrieves information about all facial features stored in the SQLite database.
     *
     * @param infoList Output parameter to store the list of facial feature information.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t GetTotalFeatures(std::vector<FaceFeatureInfo>& infoList);

    /**
     * @brief Displays the total number of facial features stored in the SQLite database.
     *
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t ViewTotal();

private:
    std::shared_ptr<sqlite3> m_db_;  ///< Pointer to the SQLite database.

};

}   // namespace inspire

#endif //HYPERFACEREPO_SQLITEFACEMANAGE_H
