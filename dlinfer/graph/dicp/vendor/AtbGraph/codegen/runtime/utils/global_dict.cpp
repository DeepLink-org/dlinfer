#include "utils/global_dict.h"

#include <acl/acl.h>

#include <cstdlib>
#include <mutex>

#include "utils/config.h"
#include "utils/log.h"
#include "utils/tensor_utils.h"

namespace dicp {

GlobalDict::GlobalDict() = default;

void GlobalDict::Register(const std::string& key) {
    current_key_ = key;
    data_.try_emplace(key);
}

std::unordered_map<std::string, int>& GlobalDict::GetData() {
    if (current_key_.empty() || !data_.count(current_key_)) {
        throw std::runtime_error("Invalid GlobalDict access");
    }
    return data_.at(current_key_);
}

GlobalDict& GetGlobalDict_() {
    static GlobalDict global_dict;
    return global_dict;
}

void GlobalDict::Set(const std::string& key) {
    if (current_key_.empty() || !data_.count(current_key_)) {
        throw std::runtime_error("Invalid GlobalDict access");
    }
    current_key_ = key;
}

void RegisterToGlobalDict(const std::string& key) {
    auto& global_dict = GetGlobalDict_();
    global_dict.Register(key);
}

void SetGlobalDict(const std::string& key) {
    auto& global_dict = GetGlobalDict_();
    global_dict.Set(key);
}

std::unordered_map<std::string, int>& GetGlobalDictData() { return GetGlobalDict_().GetData(); }

}  // namespace dicp
