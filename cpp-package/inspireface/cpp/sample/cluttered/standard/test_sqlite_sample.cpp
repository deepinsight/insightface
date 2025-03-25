/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include <iostream>
#include "inspireface/feature_hub/persistence/sqlite_faces_manage.h"

using namespace inspire;

int main() {
    SQLiteFaceManage db;

    db.OpenDatabase("t.db");

    db.ViewTotal();

    return 0;
}