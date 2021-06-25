#ifndef EXP3_HH
#define EXP3_HH

#include <string>
#include <vector>

#include "context.hh"

class Exp3 {
public:
    Exp3();
    Exp3(std::string model_path);
    std::size_t predict(std::vector<double> input, std::size_t curr_buffer_, std::size_t last_format_);
    void reload_model();
private:
    std::string model_path_;
    std::vector<Context> contexts_;
    double dist(std::vector<double> v1, std::vector<double> v2);
};

#endif /* EXP3_HH */
