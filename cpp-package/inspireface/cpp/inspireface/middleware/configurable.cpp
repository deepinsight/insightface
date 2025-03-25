/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "configurable.h"

namespace inspire {

std::string Configurable::toString(int indent) const {
    if (indent != 0)
        return m_configuration.dump(indent);
    else
        return m_configuration.dump();
}

std::vector<std::string> Configurable::getNameList() const {
    std::vector<std::string> keys;
    for (const auto &element : m_configuration.items()) {
        keys.push_back(element.key());
    }
    return keys;
}

Configurable &Configurable::operator=(const Configurable &other) {
    if (this != &other) {                         // Check the self-assignment
        m_configuration = other.m_configuration;  // Deep copy using the assignment operator of nlohmann::json
    }
    return *this;
}

}  // namespace inspire