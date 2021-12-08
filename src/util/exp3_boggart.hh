#ifndef BOGGART_HH
#define BOGGART_HH

#include "exp3.hh"

#include <string>
#include <vector>
#include <fstream>

class Exp3Boggart : public Exp3
{
public:
  Exp3Boggart(std::string model_path, double lr);
  std::tuple<std::size_t, std::size_t> predict(std::size_t max_conservative,
                                               std::size_t max_risk,
                                               std::vector<std::size_t> available_actions_idx);

private:
  std::size_t calc_inner_index(std::size_t max_conservative, std::size_t max_risk);
};

#endif /* BOGGART_HH */
