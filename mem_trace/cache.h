#ifndef CACHE_H
#define CACHE_H
#include <cstddef>   // For size_t
#include <string>    // For std::string
#include <unordered_map>
#include <cstdint>

#include <unordered_map>
#include <cstdint>

class Cache {
public:
    enum ReplacementPolicy { LRU, LFU, FIFO, RAND };

    Cache(size_t size, size_t block_size, size_t mapping_pol, ReplacementPolicy replace_pol);

    bool read(uint64_t address);
    void load(uint64_t address);

private:
    struct CacheLine {
        int use;
    };

    size_t cache_size;
    size_t block_size;
    size_t mapping_pol;
    size_t num_sets;
    int tag_shift;
    int set_shift;
    ReplacementPolicy replace_pol;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, CacheLine>> cache;

    void evictLine(std::unordered_map<uint64_t, CacheLine>& cache_set);
    void updateUse(CacheLine& line, std::unordered_map<uint64_t, CacheLine>& cache_set);
    uint64_t getTag(uint64_t address) const;
    uint64_t getSet(uint64_t address) const;
};

void logMissAddress(const std::string& file_path, uint64_t address);

#endif // CACHE_H
