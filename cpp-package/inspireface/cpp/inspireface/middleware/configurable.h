//
// Created by tunm on 2023/5/5.
//

#pragma once
#ifndef HYPERAI_PARAMETER_H
#define HYPERAI_PARAMETER_H
#include "nlohmann/json.hpp"
#include <iostream>

using nlohmann::json;

#ifndef INSPIRE_API
#define INSPIRE_API
#endif

namespace inspire {

/**
 * @class Configurable
 * @brief Class for managing parameters as JSON data.
 *
 * This class provides methods to set, get, and load parameters as JSON data.
 */
class INSPIRE_API Configurable {
public:
    Configurable() = default;;

    Configurable(const Configurable& p) : m_configuration(p.m_configuration) {}

    virtual ~Configurable() {}

    Configurable& operator=(const Configurable& other);

    /**
     * @brief Get a list of parameter names.
     * @return std::vector<std::string> A list of parameter names.
     */
    std::vector<std::string> getNameList() const;

    /**
     * @brief Convert parameters to a formatted JSON string.
     * @param indent The indentation level for formatting.
     * @return std::string The formatted JSON string representing the parameters.
     */
    std::string toString(int indent = 4) const;

    /**
     * @brief Check if a parameter exists by name.
     * @param name The name of the parameter to check.
     * @return bool True if the parameter exists, false otherwise.
     */
    bool has(const std::string& name) const noexcept {
        return m_configuration.contains(name);
    }

    /**
     * @brief Set a parameter with a specific name and value.
     * @param name The name of the parameter.
     * @param value The value to set for the parameter.
     */
    template <typename ValueType>
    void set(const std::string& name, const ValueType& value) {
        m_configuration[name] = value;
    }

    /**
     * @brief Set a parameter with a specific name and value.
     * @param name The name of the parameter.
     * @param value The value to set for the parameter.
     */
    template <typename ValueType>
    ValueType get(const std::string& name) const {
        if (!has(name)) {
            throw std::out_of_range("out_of_range in Parameter::get : " + name);
        }
        return m_configuration.at(name).get<ValueType>();
    }

    /**
     * @brief Load parameters from a JSON object.
     * @param j The JSON object containing parameters.
     */
    void load(const nlohmann::json& j) {
        for (const auto& item : j.items()) {
            const auto& key = item.key();
            const auto& value = item.value();

            if (value.is_boolean()) {
                set<bool>(key, value.get<bool>());
            } else if (value.is_number_integer()) {
                set<int>(key, value.get<int>());
            } else if (value.is_number_float()) {
                set<float>(key, value.get<float>());
            } else if (value.is_string()) {
                set<std::string>(key, value.get<std::string>());
            } else if (value.is_array()) {
                if (!value.empty()) {
                    if (value[0].is_number_integer()) {
                        set<std::vector<int>>(key, value.get<std::vector<int>>());
                    } else if (value[0].is_number_float()) {
                        set<std::vector<float>>(key, value.get<std::vector<float>>());
                    } // add more types as needed
                    // ...
                }
            }
            // Add handling for other types as needed
        }
    }


private:
    json m_configuration;      ///< JSON object to store parameters.
};

#define CONFIGURABLE_SUPPORT                                                                                 \
protected:                                                                                                \
    inspire::Configurable m_configuration;                                                                            \
                                                                                                          \
public:                                                                                                   \
    const inspire::Configurable& getConfiguration() const {                                                        \
        return m_configuration;                                                                                  \
    }                                                                                                     \
                                                                                                          \
    void setConfiguration(const inspire::Configurable& param) {                                                    \
        m_configuration = param;                                                                                 \
    }                                                                                                     \
                                                                                                          \
    bool hasData(const std::string& name) const noexcept {                                                   \
        return m_configuration.has(name);                                                                       \
    }                                                                                                     \
                                                                                                          \
    template <typename ValueType>                                                                         \
    void setData(const std::string& name, const ValueType& value) {                                           \
        m_configuration.set<ValueType>(name, value);                                                             \
    }                                                                                                     \
                                                                                                          \
    template <typename ValueType>                                                                         \
    ValueType getData(const std::string& name) const {                                                        \
        return m_configuration.get<ValueType>(name);                                                             \
    }                                                                                                     \
                                                                                                          \
    template <typename ValueType>                                                                         \
    void pushData(const inspire::Configurable& param, const std::string& name,                              \
                          const ValueType& default_value) {                                               \
        if (param.has(name)) {                                                                           \
            setData<ValueType>(name, param.get<ValueType>(name));                                        \
        } else {                                                                                          \
            setData<ValueType>(name, default_value);                                                     \
        }                                                                                                 \
    }                                                                                                     \
    void loadData(const nlohmann::json& j) {                                                                  \
        m_configuration.load(j);                                                                                 \
    }                                                                                                        \
    std::string toStr(int indent = 4) {                                                                      \
        return m_configuration.toString(indent);                                                          \
    }

} // namespace hyper
#endif //HYPERAI_PARAMETER_H
