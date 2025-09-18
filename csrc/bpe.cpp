#include <__atomic/aliases.h>
#include <_types/_uint16_t.h>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>
#include <streambuf>
#include <cstdio>

#include <benchmark/benchmark.h>
#include "robin_hood.h"

#include <unicode/regex.h>
#include <boost/heap/d_ary_heap.hpp>

#include "ref.hpp"

template <class T>
using Heap = boost::heap::d_ary_heap<
    T,
    boost::heap::arity<4>,
    boost::heap::mutable_<true>,
    boost::heap::compare<std::less<T>>
>;




struct ScopedTimer {
  double &acc;
  std::chrono::high_resolution_clock::time_point start;
  explicit ScopedTimer(double &accumulator)
      : acc(accumulator), start(std::chrono::high_resolution_clock::now()) {}
  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    acc += std::chrono::duration<double>(end - start).count();
  }
};


// with python it takes 180s
const std::string regex_pat = R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)";
const int num_merges = 100;

std::vector<uint16_t> to_bytes(const std::string &s) {
  std::vector<uint16_t> bytes;
  bytes.reserve(s.size());
  std::transform(s.begin(), s.end(), std::back_inserter(bytes),
                 [](char c) { return static_cast<uint16_t>(std::byte(c)); });
  return bytes;
}

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};




constexpr uint16_t nsize = num_merges + 256 + 1;
void byte_pair_enc(
    std::vector<std::vector<uint16_t>> &bytes,
    Heap<std::pair<long long, std::pair<std::uint16_t, std::uint16_t>>> &pq,
    robin_hood::unordered_map<std::pair<uint16_t, uint16_t>,
                              decltype(pq.push({})), PairHash> &keys) {
  const uint16_t N = (int)bytes.size();

  const uint16_t start_idx = 256;
  std::vector<long long> pos;
  robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, long long, PairHash> diffs;

  for (uint16_t it = 0; it < num_merges; it++) {
    const uint16_t new_idx = it + start_idx;

    const std::pair<long long, std::pair<uint16_t, uint16_t>>& most_freq = pq.top();

    if (most_freq.first <= 1)
      break;
    std::pair<uint16_t, uint16_t> mf_vals = most_freq.second;

    for (int b = 0; b < N; b++) {
      auto& enc_x = bytes[b];
      std::vector<uint16_t> nenc_x;
      nenc_x.reserve(enc_x.size());

      for (uint16_t i = 0, sz = (int)enc_x.size(); i < sz - 1; i++) {
        if (enc_x[i] == mf_vals.first && enc_x[i + 1] == mf_vals.second) {
          nenc_x.push_back(new_idx);
          pos.push_back((int)nenc_x.size() - 1);
          if (i - 1 >= 0) {
            diffs[{enc_x[i-1], new_idx}]++;
            diffs[{enc_x[i-1], mf_vals.first}]--;
          }
          i++;
        } else {
          nenc_x.push_back(enc_x[i]);
        }
      }
      enc_x = nenc_x;

      for (auto i : pos) {
        if (i+1 < (int)enc_x.size()) {
          if (enc_x[i+1] != new_idx)
            diffs[{mf_vals.second, enc_x[i + 1]}]--;
          else 
            diffs[{mf_vals.second, mf_vals.first}]--;
          diffs[{new_idx, enc_x[i + 1]}]++;
        }
      }
      pos.clear();
    }

    for (auto [k, v] : diffs) {
      if (keys.contains(k)) {
        auto handle = keys[k];
        pq.update(handle, {(*handle).first + v, k});
      } else {
        keys[k] = pq.push({v, k});
      }
    }
    diffs.clear();
  }
}

void byte_pair_enc_old(std::vector<std::vector<uint16_t>> &bytes, robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, long long, PairHash> &freqs) {
  const uint16_t N = (int)bytes.size();

  const uint16_t start_idx = 256;
  std::vector<long long> pos;

  for (uint16_t it = 0; it < num_merges; it++) {
    const uint16_t new_idx = it + start_idx;

    // std::priority_queue<std::pair<long long, std::pair<uint16_t, uint16_t>>> pq;
    std::pair<long long, std::pair<uint16_t, uint16_t>> most_freq; // idea: use a priority queue for this
    most_freq = {0, {0, 0}};
    for (auto &[k, v] : freqs)
      most_freq = std::max(most_freq, {v, k});

    if (most_freq.first <= 1)
      break;
    std::pair<uint16_t, uint16_t> mf_vals = most_freq.second;

    for (int b = 0; b < N; b++) {
      auto& enc_x = bytes[b];
      std::vector<uint16_t> nenc_x;
      nenc_x.reserve(enc_x.size());

      for (uint16_t i = 0, sz = (int)enc_x.size(); i < sz - 1; i++) {
        if (enc_x[i] == mf_vals.first && enc_x[i + 1] == mf_vals.second) {
          nenc_x.push_back(new_idx);
          pos.push_back((int)nenc_x.size() - 1);
          if (i - 1 >= 0) {
            freqs[{enc_x[i-1], new_idx}]++;
            freqs[{enc_x[i-1], mf_vals.first}]--;
          }
          i++;
        } else {
          nenc_x.push_back(enc_x[i]);
        }
      }
      enc_x = nenc_x;

      for (auto i : pos) {
        if (i+1 < (int)enc_x.size()) {
          if (enc_x[i+1] != new_idx)
            freqs[{mf_vals.second, enc_x[i + 1]}]--;
          else 
            freqs[{mf_vals.second, mf_vals.first}]--;
          freqs[{new_idx, enc_x[i + 1]}]++;
        }
      }
      pos.clear();
    }
    freqs.erase(mf_vals); 
  }
}



std::vector<std::string> regex_split(const std::vector<std::string>& in) {
  std::vector<std::string> matches;

  for (auto &s : in) {
    UErrorCode status = U_ZERO_ERROR;
    icu_77::UnicodeString us = icu_77::UnicodeString::fromUTF8(s);
    icu_77::RegexMatcher m(icu_77::UnicodeString::fromUTF8(regex_pat), us, 0,
                           status);

    while (m.find(status) && U_SUCCESS(status)) {
      icu_77::UnicodeString tok = m.group(status);
      std::string utf8;
      tok.toUTF8String(utf8);
      matches.emplace_back(std::move(utf8));
    }
  }
  return matches;
}

struct OutputSilencer {
  std::streambuf *old_out{nullptr}, *old_err{nullptr};
  OutputSilencer() {
    static struct : std::streambuf { int overflow(int c) override { return c; } } nullbuf;
    old_out = std::cout.rdbuf(&nullbuf);
    old_err = std::cerr.rdbuf(&nullbuf);
  }
  ~OutputSilencer() {
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    fflush(stdout); fflush(stderr);
  }
};

static void BM_BPE(benchmark::State &state) {
  std::ifstream t("data/input.txt");
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::vector<std::string> in{buffer.str()};
  std::vector<std::string> matches = regex_split(in);

  for (auto _ : state) {
    // state.PauseTiming();

    robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, long long, PairHash> freqs;
    std::vector<std::vector<uint16_t>> bytes;
    bytes.reserve((int)matches.size());
    std::transform(matches.begin(), matches.end(), std::back_inserter(bytes),
                  [](std::string &s) { return (to_bytes(s)); });

    for (auto byte : bytes)
      for (uint16_t i = 0, sz = (int)byte.size(); i < sz - 1; i++)
        freqs[{byte[i], byte[i + 1]}]++;

    Heap<std::pair<long long, std::pair<std::uint16_t, std::uint16_t>>> pq;
    robin_hood::unordered_map<std::pair<uint16_t, uint16_t>,
                              decltype(pq.push({})), PairHash>
        keys;
    for (auto [k, v] : freqs) {
      keys[k] = pq.push({v, k});
    }
    // state.ResumeTiming();

    byte_pair_enc(bytes, pq, keys);
    // byte_pair_enc_old(bytes, freqs);
    benchmark::DoNotOptimize(bytes);
  }
}

static void BM_BPE_OLD(benchmark::State &state) {
  std::ifstream t("data/input.txt");
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::vector<std::string> in{buffer.str()};
  std::vector<std::string> matches = regex_split(in);

  for (auto _ : state) {
    state.PauseTiming();

    robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, long long, PairHash> freqs;
    std::vector<std::vector<uint16_t>> bytes;
    bytes.reserve((int)matches.size());
    std::transform(matches.begin(), matches.end(), std::back_inserter(bytes),
                  [](std::string &s) { return (to_bytes(s)); });

    for (auto byte : bytes)
      for (uint16_t i = 0, sz = (int)byte.size(); i < sz - 1; i++)
        freqs[{byte[i], byte[i + 1]}]++;
    state.ResumeTiming();

    byte_pair_enc_old(bytes, freqs);
    benchmark::DoNotOptimize(bytes);
  }
}

static void BM_REF(benchmark::State &state){
  for(auto _ : state){
    OutputSilencer mute; 
    fastBPE::learnbpe(100, "data/input.txt", "");
  }
}

BENCHMARK(BM_BPE);
BENCHMARK(BM_BPE_OLD);
BENCHMARK(BM_REF);
BENCHMARK_MAIN();


/*
todo: need to split the function into smaller parts? 

cuda parallize along the axes (regex word splits), use grid-stride trick
instead of moving vectors every single time, we have another vector offsets.
when looking for pairs use offset array, and when replacing we increment offset
figure out how to use dp and tries to optimize the code

*/