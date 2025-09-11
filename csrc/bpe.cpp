#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <iterator>
#include <sstream>
#include <fstream>


// future optimizations: switch from int to smaller + unsigned b/c < 50k

// with python it takes 180s
const int num_merges = 100;

std::vector<int> to_bytes(const std::string& s) {
    std::vector<int> bytes;
    bytes.reserve(s.size());
      
    std::transform(s.begin(), s.end(), std::back_inserter(bytes), [](char c){
        return static_cast<int>(std::byte(c));
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

void byte_pair_enc(std::vector<std::vector<int>>& bytes) {
    const int N = (int)bytes.size();

    std::unordered_map<int, std::vector<int>> decoder_dict;
    std::unordered_map<std::pair<int, int>, int, PairHash> encoder_dict;

    std::unordered_map<std::pair<int, int>, int, PairHash> freqs;
    const int start_idx = 256;
    for (int it=0; it<num_merges; it++) {
        const int new_idx = it + start_idx;

        for (auto byte : bytes) 
            for (int i=0, sz=(int)byte.size(); i<sz-1; i++) 
                freqs[std::pair<int, int>{byte[i], byte[i+1]}]++;
        
        std::pair<int, std::pair<int, int>> most_freq = {0, {0, 0}};
        for (auto &[k, v]: freqs)
            most_freq = std::max(most_freq, {v, k});

        if (most_freq.first <= 1) break;
        std::pair<int, int> mf_vals = most_freq.second;

        std::vector<int> nkey;
        nkey.insert(nkey.end(), decoder_dict[mf_vals.first].begin(),
                    decoder_dict[mf_vals.first].end());
        nkey.insert(nkey.end(), decoder_dict[mf_vals.second].begin(),
                    decoder_dict[mf_vals.second].end());
        decoder_dict[new_idx] = nkey;
        encoder_dict[mf_vals] = new_idx;

        for (auto& enc_x : bytes) {
            std::vector<int> nenc_x;
            nenc_x.reserve(enc_x.size());
            
            for (int i=0, sz=(int)enc_x.size(); i<sz-1; i++) {
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

void byte_pair_enc_1(std::vector<std::vector<int>>& bytes) {
    const int N = (int)bytes.size();
    std::vector<std::vector<int>> offsets(N);
    for (int i=0; i<N; i++)
        offsets[i] = std::vector<int>((int)bytes[i].size(), 1);

    std::unordered_map<int, std::vector<int>> decoder_dict;
    std::unordered_map<std::pair<int, int>, int, PairHash> encoder_dict;

    std::unordered_map<std::pair<int, int>, int, PairHash> freqs;
    const int start_idx = 256;
    for (int it=0; it<num_merges; it++) {
        const int new_idx = it + start_idx;

        for (int b=0; b<N; b++) 
            for (int i=0, sz=(int)bytes[b].size(); i<sz-1; i++) 
                freqs[std::pair<int, int>{bytes[b][i], bytes[b][i+offsets[b][i]]}]++;
        
        std::pair<int, std::pair<int, int>> most_freq = {0, {0, 0}};
        for (auto &[k, v]: freqs)
            most_freq = std::max(most_freq, {v, k});

        if (most_freq.first <= 1) break;
        std::pair<int, int> mf_vals = most_freq.second;

        std::vector<int> nkey;
        nkey.insert(nkey.end(), decoder_dict[mf_vals.first].begin(),
                    decoder_dict[mf_vals.first].end());
        nkey.insert(nkey.end(), decoder_dict[mf_vals.second].begin(),
                    decoder_dict[mf_vals.second].end());
        decoder_dict[new_idx] = nkey;
        encoder_dict[mf_vals] = new_idx;

        for (int b=0; b<N; b++) {
            auto& enc_x = bytes[b];
            for (int i=0, sz=(int)enc_x.size(); i<sz-1; i++) {
                if (enc_x[i] == mf_vals.first && enc_x[i+offsets[b][i]] == mf_vals.second) {
                    enc_x[i] = new_idx;
                    offsets[b][i]++;
                } 
            }
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

    std::vector<std::vector<int>> bytes;
    bytes.reserve((int)in.size());
    std::transform(in.begin(), in.end(), std::back_inserter(bytes),
                    [](std::string &s) { return to_bytes(s); });
    for (auto _ : state) {
        byte_pair_enc(bytes);
        benchmark::DoNotOptimize(bytes);
    }
}

static void BM_BPE_1(benchmark::State& state) {
    std::ifstream t("data/input.txt");
    std::stringstream buffer;
    buffer << t.rdbuf();

    std::vector<std::string> in{buffer.str()};

    std::vector<std::vector<int>> bytes;
    bytes.reserve((int)in.size());
    std::transform(in.begin(), in.end(), std::back_inserter(bytes),
                    [](std::string &s) { return to_bytes(s); });
    for (auto _ : state) {
        byte_pair_enc_1(bytes);
        benchmark::DoNotOptimize(bytes);
    }
}

BENCHMARK(BM_BPE);
BENCHMARK(BM_BPE_1);
BENCHMARK_MAIN();


/*
https://en.cppreference.com/w/cpp/regex.html

cuda parallize along the axes (regex word splits), use grid-stride trick
instead of moving vectors every single time, we have another vector offsets. when looking for pairs use offset array, and when replacing we increment offset
figure out how to use dp and tries to optimize the code

compile with O3 - done, 10x speedup

test this: https://codeforces.com/blog/entry/60737
*/
