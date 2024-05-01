//
// Created by Tunm-Air13 on 2023/10/11.
//

#include <iostream>
#include "inspireface/feature_hub/persistence/sqlite_faces_manage.h"

using namespace inspire;

int main() {
    SQLiteFaceManage db;

    db.OpenDatabase("t.db");

    db.ViewTotal();

    return 0;
}