from collections import defaultdict
import pickle


NUM_MERGES = 500
SAVE_FREQ = 100


def byte_pair_enc(xs, num_merges): 
    enc_xs = [list(x.encode("utf-8")) for x in xs]

    decoder_dict = {i: [i] for i in range(256)}
    encoder_dict = {}
    start_idx = 256
    for it in tqdm(range(num_merges)): 
        new_idx = it + start_idx
        freqs = defaultdict(lambda: 0)
        for enc_x in enc_xs:
            for f,s in zip(enc_x, enc_x[1:]):
                freqs[(f, s)] += 1
        
        most_freq = max(freqs, key=freqs.get)
        if freqs[most_freq] <= 1:
            break
        
        decoder_dict[new_idx] = decoder_dict[most_freq[0]] + decoder_dict[most_freq[1]]
        encoder_dict[most_freq] = new_idx

        for xs_idx in range(len(enc_xs)):
            enc_x = enc_xs[xs_idx]
            nenc_x = []
            i = 0
            while i < len(enc_x):
                if tuple(enc_x[i:i+2]) == most_freq:
                    nenc_x.append(new_idx)
                    i += 1
                else: 
                    nenc_x.append(enc_x[i])
                i += 1
            enc_xs[xs_idx] = nenc_x

        if it % SAVE_FREQ == 0: 
            with open(f'ckpt/bpe/enc_{it}.pkl', 'wb') as f:
                pickle.dump(encoder_dict, f)
            with open(f'ckpt/bpe/dec_{it}.pkl', 'wb') as f:
                pickle.dump(decoder_dict, f)

    return encoder_dict, decoder_dict


def main():
    encoder_dict, decoder_dict = byte_pair_enc(xs, NUM_MERGES)
    
    with open(f'ckpt/bpe/enc_final.pkl', 'wb') as f:
        pickle.dump(encoder_dict, f)
    with open(f'ckpt/bpe/dec_final.pkl', 'wb') as f:
        pickle.dump(decoder_dict, f)


if __name__ == "__main__":
    main()