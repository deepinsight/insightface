#include <inspireface.h>
#include <unistd.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        HFLogPrint(HF_LOG_ERROR, "Usage: %s <pack_path>", argv[0]);
        return -1;
    }
    
    const char* packPath = argv[1];
    HResult ret;
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);
        return ret;
    }
    const char* DBFilePath = "feature.db";
    
    // remove old db file
    if (access(DBFilePath, F_OK) == 0) {
        if (remove(DBFilePath) != 0) {
            HFLogPrint(HF_LOG_ERROR, "Failed to remove old db file: %s", DBFilePath);
            return -1;
        }
        HFLogPrint(HF_LOG_INFO, "Remove old db file: %s", DBFilePath);
    }

    HFFeatureHubConfiguration featureHubConfiguration;
    featureHubConfiguration.primaryKeyMode = HF_PK_AUTO_INCREMENT; 
    featureHubConfiguration.enablePersistence = 1;                 
    featureHubConfiguration.persistenceDbPath = DBFilePath;      
    featureHubConfiguration.searchMode = HF_SEARCH_MODE_EAGER;     
    featureHubConfiguration.searchThreshold = 0.48f;             

    ret = HFFeatureHubDataEnable(featureHubConfiguration);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Enable FeatureHub failed: %d\n", ret);
        return ret;
    }
    if (access(DBFilePath, F_OK) != 0) {
        HFLogPrint(HF_LOG_ERROR, "DB file not found: %s", DBFilePath);
        return -1;
    }
    HFLogPrint(HF_LOG_INFO, "DB file found: %s", DBFilePath);

    // ....

    HFTerminateInspireFace();
    return 0;
}