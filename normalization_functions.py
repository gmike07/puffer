MILLION = 1000000.0
PKT_BYTES = 1500.0
DELETED_COLS = ['file_index', 'chunk_index', 'max_pacing_rate', 'reordering',
                'advmss', 'pmtu', 'fackets', 'snd_mss', 'probes']
TO_DELETE_COLS = [x for x in DELETED_COLS if x != 'chunk_index']

def normalize_chunks(df_chunks):
    df_chunks.drop(TO_DELETE_COLS, 1 if len(df_chunks.shape) > 1 else 0, inplace=True)
    df_chunks['ca_state'] /= 4.0
    df_chunks['total_retrans'] /= 100
    df_chunks['busy_time'] /= 1000 * MILLION

    time_normalization = ['rtt', 'sndbuf_limited', 'rwnd_limited',
                          'delivery_rate', 'min_rtt', 'rcv_rtt',
                          'rttvar', 'timestamp', 'rto', 'ato']
    for col in time_normalization:
        df_chunks[col] /= 10 * MILLION

    data_normalization = ['data_segs_out', 'data_segs_in', 'segs_out',
                          'segs_in',
                          'rcv_mss', 'snd_cwnd', 'rcv_space',
                          'last_ack_recv', 'last_data_recv', 'rcv_ssthresh',
                          'last_data_sent']
    for col in data_normalization:
        df_chunks[col] /= 16 * PKT_BYTES

    bytes_normalization = ['bytes_acked', 'bytes_received', 'notsent_bytes',
                           'snd_ssthresh']
    for col in bytes_normalization:
        df_chunks[col] /= (PKT_BYTES * 1024)

    df_chunks['bytes_acked'] /= 16.0
    df_chunks['snd_ssthresh'] /= 1000.0
    df_chunks['rcv_ssthresh'] /= 1000.0

    metrics_normalization = ['retrans', 'lost', 'sacked', 'unacked']
    for col in metrics_normalization:
        df_chunks[col] /= 1024.0
    return df_chunks


def normalize_answers(df_answers):
    answer_norm = ['media_chunk_size']
    for col in answer_norm:
        df_answers[col] /= 100.0
    # df_answers['cum_rebuffer'] /= 1000.0
    return df_answers
