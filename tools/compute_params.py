def compute(enc_seq_length, noise_rate, span_len):
    return int(enc_seq_length * (noise_rate + noise_rate / span_len) / (1 - noise_rate + noise_rate / span_len)) + 100

enc_seq_length = 1024
noise_rate = 0.15
span_len = 6

print(compute(enc_seq_length, noise_rate, span_len))