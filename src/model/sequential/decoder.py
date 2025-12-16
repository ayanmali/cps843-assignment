import torch
import torch.nn as nn

def ctc_greedy_decode(logits, char_to_index, index_to_char):
    
    log_probs = nn.functional.log_softmax(logits, dim=2)
    
    log_probs = log_probs.permute(1, 0, 2)
    B, _, _ = log_probs.size()

    best_paths = torch.argmax(log_probs, dim=2) 
    
    blank_index = len(char_to_index) - 1

    decoded_texts = []
    
    for i in range(B):
        raw_sequence = best_paths[i].tolist()
        
        text = []
        last_char_index = -1
        
        for char_index in raw_sequence:
            if char_index == blank_index:
                last_char_index = char_index
                continue
            
            if char_index == last_char_index:
                continue
            
            text.append(index_to_char[char_index])
            last_char_index = char_index
            
        decoded_texts.append("".join(text))

    return decoded_texts

if __name__ == "__main__":
    char_list = list("abcdefghijklmnopqrstuvwxyz")
    char_list.append('_')
    char_to_index = {c: i for i, c in enumerate(char_list)}
    index_to_char = {i: c for c, i in char_to_index.items()}

    W, B, C = 15, 1, 27 
    dummy_logits = torch.randn(W, B, C) 

    h_idx, e_idx, l_idx, o_idx, w_idx, r_idx, d_idx, blank_idx = (
        char_to_index['h'], char_to_index['e'], char_to_index['l'], char_to_index['o'], 
        char_to_index['w'], char_to_index['r'], char_to_index['d'], char_to_index['_']
    )

    simulated_indices = [h_idx, h_idx, e_idx, blank_idx, l_idx, l_idx, o_idx, o_idx, 
                        blank_idx, w_idx, r_idx, r_idx, l_idx, d_idx, blank_idx]

    for t, idx in enumerate(simulated_indices):
        dummy_logits[t, 0, idx] = 10.0

    decoded_strings = ctc_greedy_decode(dummy_logits, char_to_index, index_to_char)

    print(f"Decoded Text: {decoded_strings[0]}")