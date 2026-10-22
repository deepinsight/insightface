#include <inspireface/inspireface.hpp>
int main() {
    inspire::Session session;
    const auto version = inspire::GetComponentVersion(inspire::ComponentType::MNN);
    return version.IsEnabled() && !inspire::GetComponentVersionsString().empty() ? 0 : 1;
}
