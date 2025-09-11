#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <iterator>
#include <sstream>
#include <fstream>

#include "robin_hood.h"


// future optimizations: switch from int to smaller + unsigned b/c < 50k


// with python it takes 180s
const int num_merges = 100;

std::vector<uint16_t> to_bytes(const std::string& s) {
    std::vector<uint16_t> bytes;
    bytes.reserve(s.size());
    std::transform(s.begin(), s.end(), std::back_inserter(bytes), [](char c){
        return static_cast<uint16_t>(std::byte(c));
    });
    return bytes;
}

struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        return h1 ^ (h2 << 1); 
    }
};

constexpr uint16_t nsize = num_merges + 256 + 1;
void byte_pair_enc(std::vector<std::vector<uint16_t>>& bytes) {
    const uint16_t N = (int)bytes.size();

    robin_hood::unordered_map<uint16_t, std::vector<uint16_t>> decoder_dict;
    robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, uint16_t, PairHash> encoder_dict;

    robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, long long, PairHash> freqs;
    const uint16_t start_idx = 256;
    for (uint16_t it=0; it<num_merges; it++) {
        const uint16_t new_idx = it + start_idx;

        for (auto byte : bytes) 
            for (uint16_t i=0, sz=(int)byte.size(); i<sz-1; i++) 
                freqs[std::pair<uint16_t, uint16_t>{byte[i], byte[i+1]}]++;
        
        std::pair<long long, std::pair<uint16_t, uint16_t>> most_freq = {0, {0, 0}};
        for (auto &[k, v]: freqs)
            most_freq = std::max(most_freq, {v, k});

        if (most_freq.first <= 1) break;
        std::pair<uint16_t, uint16_t> mf_vals = most_freq.second;

        std::vector<uint16_t> nkey;
        nkey.insert(nkey.end(), decoder_dict[mf_vals.first].begin(),
                    decoder_dict[mf_vals.first].end());
        nkey.insert(nkey.end(), decoder_dict[mf_vals.second].begin(),
                    decoder_dict[mf_vals.second].end());
        decoder_dict[new_idx] = nkey;
        encoder_dict[mf_vals] = new_idx;

        for (auto& enc_x : bytes) {
            std::vector<uint16_t> nenc_x;
            nenc_x.reserve(enc_x.size());
            
            for (uint16_t i=0, sz=(int)enc_x.size(); i<sz-1; i++) {
                if (enc_x[i] == mf_vals.first && enc_x[i+1] == mf_vals.second) {
                    nenc_x.push_back(new_idx);
                    i++;
                } else {
                    nenc_x.push_back(enc_x[i]);
                }
            }
            enc_x = nenc_x;
        }
        freqs.clear();
    }

}

void byte_pair_enc_test(std::vector<std::vector<uint16_t>>& bytes) {
    const uint16_t N = (int)bytes.size();

    robin_hood::unordered_map<uint16_t, std::vector<uint16_t>> decoder_dict;
    robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, uint16_t, PairHash> encoder_dict;

    robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, long long, PairHash> freqs;
    const uint16_t start_idx = 256;
    for (uint16_t it=0; it<num_merges; it++) {
        const uint16_t new_idx = it + start_idx;

        for (auto byte : bytes) 
            for (uint16_t i=0, sz=(int)byte.size(); i<sz-1; i++) 
                freqs[std::pair<uint16_t, uint16_t>{byte[i], byte[i+1]}]++;
        
        std::pair<long long, std::pair<uint16_t, uint16_t>> most_freq = {0, {0, 0}};
        for (auto &[k, v]: freqs)
            most_freq = std::max(most_freq, {v, k});

        if (most_freq.first <= 1) break;
        std::pair<uint16_t, uint16_t> mf_vals = most_freq.second;

        std::vector<uint16_t> nkey;
        nkey.insert(nkey.end(), decoder_dict[mf_vals.first].begin(),
                    decoder_dict[mf_vals.first].end());
        nkey.insert(nkey.end(), decoder_dict[mf_vals.second].begin(),
                    decoder_dict[mf_vals.second].end());
        decoder_dict[new_idx] = nkey;
        encoder_dict[mf_vals] = new_idx;

        for (auto& enc_x : bytes) {
            std::vector<uint16_t> nenc_x;
            nenc_x.reserve(enc_x.size());
            
            for (uint16_t i=0, sz=(int)enc_x.size(); i<sz-1; i++) {
                if (enc_x[i] == mf_vals.first && enc_x[i+1] == mf_vals.second) {
                    nenc_x.push_back(new_idx);
                    i++;
                } else {
                    nenc_x.push_back(enc_x[i]);
                }
            }
            enc_x = nenc_x;
        }
        freqs.clear();
    }
}

#include <benchmark/benchmark.h>

static void BM_BPE(benchmark::State& state) {
    std::ifstream t("data/input.txt");
    std::stringstream buffer;
    buffer << t.rdbuf();

    std::vector<std::string> in{buffer.str()};

    std::vector<std::vector<uint16_t>> bytes;
    bytes.reserve((int)in.size());
    std::transform(in.begin(), in.end(), std::back_inserter(bytes),
                    [](std::string &s) { return (to_bytes(s)); });
    for (auto _ : state) {
        byte_pair_enc(bytes);
        benchmark::DoNotOptimize(bytes);
    }
}

static void BM_BPE_test(benchmark::State& state) {
    std::ifstream t("data/input.txt");
    std::stringstream buffer;
    buffer << t.rdbuf();

    std::vector<std::string> in{buffer.str()};

    std::vector<std::vector<uint16_t>> bytes;
    bytes.reserve((int)in.size());
    std::transform(in.begin(), in.end(), std::back_inserter(bytes),
                    [](std::string &s) { return (to_bytes(s)); });
    for (auto _ : state) {
        byte_pair_enc_test(bytes);
        benchmark::DoNotOptimize(bytes);
    }
}

BENCHMARK(BM_BPE);
BENCHMARK(BM_BPE_test);
BENCHMARK_MAIN();


/*
https://en.cppreference.com/w/cpp/regex.html

cuda parallize along the axes (regex word splits), use grid-stride trick
instead of moving vectors every single time, we have another vector offsets. when looking for pairs use offset array, and when replacing we increment offset
figure out how to use dp and tries to optimize the code

compile with O3 - done, 10x speedup

test this: https://codeforces.com/blog/entry/60737
*/
