#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "robin_hood.h"

#include <unicode/regex.h>

// -----------------------------------------------------------------------------
// Global timing helpers (used by encoder functions and benchmarks)
// -----------------------------------------------------------------------------

static double g_seg1 = 0.0; // counting pair frequencies
static double g_seg2 = 0.0; // finding most-frequent pair
static double g_seg3 = 0.0; // applying the merge
static double g_seg4 = 0.0; // applying the merge

// RAII helper to accumulate durations easily
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

// future optimizations: switch from int to smaller + unsigned b/c < 50k

// with python it takes 180s
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
void byte_pair_enc(std::vector<std::vector<uint16_t>> &bytes) {
  const uint16_t N = (int)bytes.size();

  // reset segment timers for this invocation
  g_seg1 = g_seg2 = g_seg3 = g_seg4 = 0.0;

  robin_hood::unordered_map<uint16_t, std::vector<uint16_t>> decoder_dict;
  robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, uint16_t, PairHash>
      encoder_dict;

  robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, long long, PairHash>
      freqs;
  const uint16_t start_idx = 256;
  for (uint16_t it = 0; it < num_merges; it++) {
    const uint16_t new_idx = it + start_idx;

    {
      ScopedTimer _t(g_seg1);
      for (auto byte : bytes)
        for (uint16_t i = 0, sz = (int)byte.size(); i < sz - 1; i++)
          freqs[{byte[i], byte[i + 1]}]++;
    }

    std::pair<long long, std::pair<uint16_t, uint16_t>> most_freq;
    {
      ScopedTimer _t(g_seg2);
      most_freq = {0, {0, 0}};
      for (auto &[k, v] : freqs)
        most_freq = std::max(most_freq, {v, k});
    }

    if (most_freq.first <= 1)
      break;
    std::pair<uint16_t, uint16_t> mf_vals = most_freq.second;

    std::vector<uint16_t> nkey;
    {
      ScopedTimer _t(g_seg3);
      nkey.insert(nkey.end(), decoder_dict[mf_vals.first].begin(),
                  decoder_dict[mf_vals.first].end());
      nkey.insert(nkey.end(), decoder_dict[mf_vals.second].begin(),
                  decoder_dict[mf_vals.second].end());
      decoder_dict[new_idx] = nkey;
      encoder_dict[mf_vals] = new_idx;
    }

    {
      ScopedTimer _t(g_seg4);
      for (auto &enc_x : bytes) {
        std::vector<uint16_t> nenc_x;
        nenc_x.reserve(enc_x.size());

        for (uint16_t i = 0, sz = (int)enc_x.size(); i < sz - 1; i++) {
          if (enc_x[i] == mf_vals.first && enc_x[i + 1] == mf_vals.second) {
            nenc_x.push_back(new_idx);
            i++;
          } else {
            nenc_x.push_back(enc_x[i]);
          }
        }
        enc_x = nenc_x;
      }
    }
    freqs.clear();
  }
}

void byte_pair_enc_test(std::vector<std::vector<uint16_t>> &bytes) {
  const uint16_t N = (int)bytes.size();

  g_seg1 = g_seg2 = g_seg3 = g_seg4 = 0.0;

  robin_hood::unordered_map<uint16_t, std::vector<uint16_t>> decoder_dict;
  robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, uint16_t, PairHash>
      encoder_dict;

  robin_hood::unordered_map<std::pair<uint16_t, uint16_t>, long long, PairHash>
      freqs;
  const uint16_t start_idx = 256;
  for (uint16_t it = 0; it < num_merges; it++) {
    const uint16_t new_idx = it + start_idx;

    {
      ScopedTimer _t(g_seg1);
#pragma omp parallel for simd
      for (auto byte : bytes) {
#pragma omp parallel for simd
        for (uint16_t i = 0, sz = (int)byte.size(); i < sz - 1; i++) {
#pragma omp atomic update
          freqs[{byte[i], byte[i + 1]}]++;
        }
      }
    }

    std::pair<long long, std::pair<uint16_t, uint16_t>> most_freq;
    {
      ScopedTimer _t(g_seg2);
      most_freq = {0, {0, 0}};
      for (auto &[k, v] : freqs)
        most_freq = std::max(most_freq, {v, k});
    }

    if (most_freq.first <= 1)
      break;
    std::pair<uint16_t, uint16_t> mf_vals = most_freq.second;

    std::vector<uint16_t> nkey;
    {
      ScopedTimer _t(g_seg3);
      nkey.insert(nkey.end(), decoder_dict[mf_vals.first].begin(),
                  decoder_dict[mf_vals.first].end());
      nkey.insert(nkey.end(), decoder_dict[mf_vals.second].begin(),
                  decoder_dict[mf_vals.second].end());
      decoder_dict[new_idx] = nkey;
      encoder_dict[mf_vals] = new_idx;
    }

    {
      ScopedTimer _t(g_seg4);
#pragma omp parallel for simd
      for (auto &enc_x : bytes) {
        std::vector<uint16_t> nenc_x;
        nenc_x.reserve(enc_x.size());

        for (uint16_t i = 0, sz = (int)enc_x.size(); i < sz - 1; i++) {
          if (enc_x[i] == mf_vals.first && enc_x[i + 1] == mf_vals.second) {
            nenc_x.push_back(new_idx);
            i++;
          } else {
            nenc_x.push_back(enc_x[i]);
          }
        }
        enc_x = nenc_x;
      }
    }
    freqs.clear();
  }
}

#include <benchmark/benchmark.h>

static void BM_BPE(benchmark::State &state) {
  std::ifstream t("data/input.txt");
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::vector<std::string> matches{buffer.str()};
//   std::vector<std::string> matches;

//   for (auto &s : in) {
//     UErrorCode status = U_ZERO_ERROR;
//     icu_77::UnicodeString us = icu_77::UnicodeString::fromUTF8(s);
//     icu_77::RegexMatcher m(
//         R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)",
//         us, 0, status);

//     while (m.find(status) && U_SUCCESS(status)) {
//       icu_77::UnicodeString tok = m.group(status);
//       std::string utf8;
//       tok.toUTF8String(utf8);
//       matches.emplace_back(std::move(utf8));
//     }
//   }

  double total_part1 = 0.0, total_part2 = 0.0, total_part3 = 0.0, total_part4 = 0.0;
  for (auto _ : state) {
    state.PauseTiming();
    std::vector<std::vector<uint16_t>> bytes;
    bytes.reserve((int)matches.size());
    std::transform(matches.begin(), matches.end(), std::back_inserter(bytes),
                  [](std::string &s) { return (to_bytes(s)); });
    state.ResumeTiming();

    byte_pair_enc(bytes);

    // accumulate segment times produced by this invocation
    total_part1 += g_seg1;
    total_part2 += g_seg2;
    total_part3 += g_seg3;
    total_part4 += g_seg4;
    benchmark::DoNotOptimize(bytes);
  }

  state.counters["part1_sec"] = total_part1;
  state.counters["part2_sec"] = total_part2;
  state.counters["part3_sec"] = total_part3;
  state.counters["part4_sec"] = total_part4;
}

static void BM_BPE_test(benchmark::State &state) {
  std::ifstream t("data/input.txt");
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::vector<std::string> in{buffer.str()};
  std::vector<std::string> matches;

  for (auto &s : in) {
    UErrorCode status = U_ZERO_ERROR;
    icu_77::UnicodeString us = icu_77::UnicodeString::fromUTF8(s);
    icu_77::RegexMatcher m(
        R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)",
        us, 0, status);

    while (m.find(status) && U_SUCCESS(status)) {
      icu_77::UnicodeString tok = m.group(status);
      std::string utf8;
      tok.toUTF8String(utf8);
      matches.emplace_back(std::move(utf8));
    }
  }

  double total_part1 = 0.0, total_part2 = 0.0, total_part3 = 0.0, total_part4 = 0.0;
  for (auto _ : state) {
    state.PauseTiming();
    std::vector<std::vector<uint16_t>> bytes;
    bytes.reserve((int)matches.size());
    std::transform(matches.begin(), matches.end(), std::back_inserter(bytes),
                  [](std::string &s) { return (to_bytes(s)); });
    state.ResumeTiming();

    byte_pair_enc_test(bytes);

    total_part1 += g_seg1;
    total_part2 += g_seg2;
    total_part3 += g_seg3;
    total_part4 += g_seg4;
    benchmark::DoNotOptimize(bytes);
  }

  state.counters["part1_sec"] = total_part1;
  state.counters["part2_sec"] = total_part2;
  state.counters["part3_sec"] = total_part3;
  state.counters["part4_sec"] = total_part4;
}

BENCHMARK(BM_BPE);
BENCHMARK(BM_BPE_test);
BENCHMARK_MAIN();

/*
todo: need to split the function into smaller parts? 

cuda parallize along the axes (regex word splits), use grid-stride trick
instead of moving vectors every single time, we have another vector offsets.
when looking for pairs use offset array, and when replacing we increment offset
figure out how to use dp and tries to optimize the code

*/