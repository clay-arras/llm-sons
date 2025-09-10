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

void byte_pair_enc(std::vector<std::string>& xs) {
    // ignoring regex for now
    // omp parallel for
    // NEED TO ADD PROPER BENCHES ASAP!!!
    int N = (int)xs.size();
    std::vector<std::vector<int>> bytes;
    bytes.reserve(N);
    std::transform(xs.begin(), xs.end(), std::back_inserter(bytes),
                   [](std::string &s) { return to_bytes(s); });

    std::cout << (int)bytes[0].size() << std::endl;

    std::unordered_map<int, std::vector<int>> decoder_dict;
    std::unordered_map<std::pair<int, int>, int, PairHash> encoder_dict;

    std::unordered_map<std::pair<int, int>, int, PairHash> freqs;
    const int start_idx = 256;
    for (int it=0; it<num_merges; it++) {
        int new_idx = it + start_idx;

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

    std::cout << (int)bytes[0].size() << std::endl;
}

int main() {
    std::ifstream t("data/input.txt");
    std::stringstream buffer;
    buffer << t.rdbuf();

    std::vector<std::string> in{buffer.str()};
    byte_pair_enc(in);
}
/*
https://en.cppreference.com/w/cpp/regex.html
*/