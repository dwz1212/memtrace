#include "cache.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cassert>

// Cache 类的构造函数
Cache::Cache(size_t size, size_t block_size, size_t mapping_pol, ReplacementPolicy replace_pol)
    : cache_size(size), block_size(block_size), mapping_pol(mapping_pol), replace_pol(replace_pol) {
    num_sets = cache_size / (block_size * mapping_pol);
    tag_shift = static_cast<int>(log2(num_sets));
    set_shift = static_cast<int>(log2(block_size));
}

// 读取地址并检查是否命中缓存
bool Cache::read(uint64_t address) {
    uint64_t tag = getTag(address);
    uint64_t set_num = getSet(address);
    auto& cache_set = cache[set_num];

    auto it = cache_set.find(tag);
    if (it != cache_set.end()) {
        // 缓存命中
        updateUse(it->second, cache_set);
        return true;
    } else {
        // 缓存未命中
        return false;
    }
}

// 将地址加载到缓存中
void Cache::load(uint64_t address) {
    uint64_t tag = getTag(address);
    uint64_t set_num = getSet(address);
    auto& cache_set = cache[set_num];

    if (cache_set.size() >= mapping_pol) {
        // 需要驱逐一个缓存行
        evictLine(cache_set);
    }

    cache_set[tag] = {0};  // 加载块
}

// 驱逐缓存行
void Cache::evictLine(std::unordered_map<uint64_t, CacheLine>& cache_set) {
    uint64_t victim_tag;
    if (replace_pol == LRU || replace_pol == LFU || replace_pol == FIFO) {
        victim_tag = std::min_element(cache_set.begin(), cache_set.end(),
                                      [](const auto& a, const auto& b) { return a.second.use < b.second.use; })->first;
    } else if (replace_pol == RAND) {
        auto it = cache_set.begin();
        std::advance(it, std::rand() % cache_set.size());
        victim_tag = it->first;
    }
    cache_set.erase(victim_tag);
}

// 更新缓存行的使用信息
void Cache::updateUse(CacheLine& line, std::unordered_map<uint64_t, CacheLine>& cache_set) {
    if (replace_pol == LRU || replace_pol == FIFO) {
        for (auto& [tag, other_line] : cache_set) {
            if (other_line.use < line.use) {
                other_line.use += 1;
            }
        }
        line.use = 0;
    } else if (replace_pol == LFU) {
        line.use += 1;
    }
}

// 获取地址的标签部分
uint64_t Cache::getTag(uint64_t address) const {
    return address >> (set_shift + tag_shift);
}

// 获取地址的集合部分
uint64_t Cache::getSet(uint64_t address) const {
    uint64_t set_mask = num_sets - 1;
    return (address >> set_shift) & set_mask;
}

// 将未命中的地址记录到日志文件
void logMissAddress(const std::string& file_path, uint64_t address) {
    std::ofstream file(file_path, std::ios::app);
    file << "0x" << std::hex << std::setw(8) << std::setfill('0') << address << "\n";
}
