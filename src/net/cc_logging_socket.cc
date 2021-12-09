#include "socket.hh"


void TCPSocket::add_chunk(ChunkInfo info)
{
    if(not info.is_video) 
    {
        return;
    }
    if(not curr_chunk.is_video)
    {
        curr_chunk = info;
    }
    else
    {
        prev_chunk = curr_chunk;
        curr_chunk = info;
    }
    is_new_chunk_scoring = true;
    is_new_chunk_model = true;
}

std::string TCPSocket::socket_double_to_string(const double input, const int precision) const
{
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision) << input;
    return stream.str();
}

double TCPSocket::quality_chunk(const ChunkInfo& chunk, const std::string& score_type) const
{
    if(score_type == "ssim")
    {
        return ssim_db_cc(chunk.ssim);
    }
    if(score_type == "bit_rate")
    {
        return chunk.media_chunk_size / (1000 * 1000 * chunk.trans_time);
    }
    return 0.0;
}

double TCPSocket::quality_chunk(const ChunkInfo& chunk) const
{
    return quality_chunk(chunk, scoring_type);
}

double TCPSocket::calc_rebuffer(const ChunkInfo& curr_chunk) const
{
    return std::max(0.0, curr_chunk.trans_time / 1000.0 - curr_chunk.video_buffer);
}

double TCPSocket::score_chunks(const ChunkInfo& prev_chunk, const ChunkInfo& curr_chunk) const
{
    double curr_quality = quality_chunk(curr_chunk);
    double prev_quality = quality_chunk(prev_chunk);
    double rebuffer_time = calc_rebuffer(curr_chunk);
    return curr_quality - 
    quality_change_qoef * abs(curr_quality - prev_quality) -
    buffer_length_coef * rebuffer_time; 
}

double TCPSocket::score_chunks() const
{
    return score_chunks(prev_chunk, curr_chunk);
}

void TCPSocket::add_cc(std::string cc)
{
    supported_ccs.push_back(cc);
}

std::vector<double> TCPSocket::get_qoe_vector() const
{
    double curr_quality_ssim = quality_chunk(curr_chunk, "ssim"), prev_quality_ssim = 0;
    double change_quality_ssim = 0, curr_quality_bit = quality_chunk(curr_chunk, "bit_rate");
    double prev_quality_bit = 0, change_quality_bit = 0;
    double rebuffer_time = calc_rebuffer(curr_chunk);
    
    if(prev_chunk.is_video)
    {
        prev_quality_ssim = quality_chunk(prev_chunk, "ssim");
        change_quality_ssim = std::abs(curr_quality_ssim - prev_quality_ssim);
        prev_quality_bit = quality_chunk(prev_chunk, "bit_rate");
        change_quality_bit = std::abs(curr_quality_bit - prev_quality_bit);
    }
    // media / 100 / 100000.0
    // time / 1000
    return {curr_chunk.ssim, 
            curr_chunk.video_buffer / 20, 
            curr_chunk.cum_rebuffer / 10.0,
            curr_chunk.media_chunk_size / 100000.0 / 100.0, 
            curr_chunk.trans_time / 1000.0,
            curr_quality_ssim - change_quality_ssim * quality_change_qoef - buffer_length_coef * rebuffer_time,
            curr_quality_ssim / MAX_SSIM,
            change_quality_ssim * quality_change_qoef / MAX_SSIM,
            buffer_length_coef * rebuffer_time, 
            curr_quality_bit,
            quality_change_qoef * change_quality_bit};
}

std::string TCPSocket::generate_chunk_statistics() const
{
    if(not curr_chunk.is_video)
    {
        return "audio,";
    }
    std::string stats = "video,";
    for(double stat: get_qoe_vector())
    {
        stats += socket_double_to_string(stat, 8) + ",";
    }
    return stats.substr(0, stats.size() - 1);
}


std::vector<uint64_t> TCPSocket::get_tcp_full_vector() const
{
    tcp_info info = get_tcp_full_info();
    return 
        {info.tcpi_sndbuf_limited, info.tcpi_rwnd_limited, info.tcpi_busy_time, info.tcpi_delivery_rate,
        info.tcpi_data_segs_out, info.tcpi_data_segs_in, info.tcpi_min_rtt, info.tcpi_notsent_bytes,
        info.tcpi_segs_in, info.tcpi_segs_out, info.tcpi_bytes_received, info.tcpi_bytes_acked, 
        info.tcpi_max_pacing_rate, info.tcpi_total_retrans, info.tcpi_rcv_space, info.tcpi_rcv_rtt,
        info.tcpi_reordering, info.tcpi_advmss, info.tcpi_snd_cwnd, info.tcpi_snd_ssthresh, info.tcpi_rttvar,
        info.tcpi_rtt, info.tcpi_rcv_ssthresh, info.tcpi_pmtu, info.tcpi_last_ack_recv, info.tcpi_last_data_recv,
        info.tcpi_last_data_sent, info.tcpi_fackets, info.tcpi_retrans, info.tcpi_lost, info.tcpi_sacked, 
        info.tcpi_unacked, info.tcpi_rcv_mss, info.tcpi_snd_mss, info.tcpi_ato, info.tcpi_rto, 
        info.tcpi_backoff, info.tcpi_probes, info.tcpi_ca_state};
}

std::vector<double> TCPSocket::get_tcp_full_normalized_vector(uint64_t delta_time) const
{
    std::vector<uint64_t> info = get_tcp_full_vector();
    info.push_back(delta_time);
    return get_tcp_full_normalized_vector(info);
}

std::vector<double> TCPSocket::get_tcp_full_normalized_vector(const std::vector<uint64_t>& vec) const
{
    return 
    {
        vec[0] / (10 * million), vec[1] / (10 * million), 
        vec[2] / (1000 * million), vec[3] / (10 * million),
        vec[4] / (16 * pkt_bytes), vec[5] / (16 * pkt_bytes), 
        vec[6] / (10 * million), vec[7] / (1024 *  pkt_bytes),
        vec[8] / (16 * pkt_bytes), vec[9] / (16 * pkt_bytes), 
        vec[10] / (1024 *  pkt_bytes), vec[11] / (16 * 1024 *  pkt_bytes), 
        vec[12] / 100.0,vec[13] / (16 * pkt_bytes), 
        vec[14] / (10 * million), vec[15] / (16 * pkt_bytes), 
        vec[16] / (1000 * 1024 * pkt_bytes), vec[17] / (10 * million),
        vec[18] / (10 * million), vec[19] / (1000 * 16 * pkt_bytes), 
        vec[20] / (16 * pkt_bytes), vec[21] / (16 * pkt_bytes),
        vec[22] / (16 * pkt_bytes), vec[23] / 1024.0, 
        vec[24] / 1024.0, vec[25] / 1024.0, 
        vec[26] / 1024.0, vec[27] / (16 * pkt_bytes), 
        vec[28] / (10 * million), vec[29] / (10 * million), 
        vec[30] / 1.0, vec[31] / 4.0,
        vec[32] / (10 * million)
    };
}

double TCPSocket::ssim_db_cc(const double ssim)
{
    if (ssim != 1) {
        return std::max(MIN_SSIM, std::min(MAX_SSIM, -10 * log10(1 - ssim)));
    } else {
        return MAX_SSIM;
    }
}

const ChunkInfo& TCPSocket::get_current_chunk() const {return curr_chunk;}

std::vector<std::string>& TCPSocket::get_supported_cc() {return supported_ccs;}

bool TCPSocket::is_valid_score_type() const
{
    return std::find(scoring_types.begin(), scoring_types.end(), scoring_type) != scoring_types.end();
}

