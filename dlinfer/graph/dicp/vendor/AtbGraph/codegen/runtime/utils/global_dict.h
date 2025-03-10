#pragma once

#include <atb/types.h>
#include <torch/torch.h>

#include <unordered_map>
#include <vector>

namespace dicp {

class GlobalDict {
public:
    GlobalDict();
    ~GlobalDict(){};
    void Register(const std::string& key);
    void Set(const std::string& key);
    std::unordered_map<std::string, int>& GetData();

private:
    std::string current_key_;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> data_;
};

void RegisterToGlobalDict(const std::string& key);
void SetGlobalDict(const std::string& key);
GlobalDict& GetGlobalDict_();

std::unordered_map<std::string, int>& GetGlobalDictData();

}  // namespace dicp
