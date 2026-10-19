#include <inspireface/inspireface.hpp>

int main() {
    auto launch = inspire::Launch::GetInstance();
    return launch->SetGlobalCPUEnginePowerMode(inspire::Launch::CPU_ENGINE_POWER_NORMAL);
}
