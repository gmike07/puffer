#ifndef STSZ_BOX_HH
#define STSZ_BOX_HH

#include "box.hh"

namespace MP4 {

class StszBox : public FullBox
{
public:
  StszBox(const uint64_t size, const std::string & type);

  /* accessors */
  uint32_t sample_size() { return sample_size_; }
  uint32_t sample_count () { return sample_count_; }

  void parse_data(MP4File & mp4, const uint64_t data_size);

  void print_structure(const unsigned int indent = 0);

private:
  uint32_t sample_size_;
  uint32_t sample_count_;

  std::vector<uint32_t> entries_;
};

}
#endif /* STSZ_BOX_HH */