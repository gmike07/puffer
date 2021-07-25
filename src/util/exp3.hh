#ifndef EXP3_HH
#define EXP3_HH

#include <string>
#include <vector>

#include "context.hh"

class Exp3 {
public:
    Exp3();
    Exp3(std::string model_path);
    Exp3(std::string model_path, std::string norm_path, double learning_rate, double delta);
    std::tuple<std::size_t, std::size_t> predict(std::vector<double> input, std::size_t curr_buffer_, std::size_t last_format_);
    void reload_model();
    std::size_t version_;
private:
    std::string model_path_;
    std::vector<Context> contexts_;
    std::vector<double> mean_;
    std::vector<double> std_;
    double dist(std::vector<double> v1, std::vector<double> v2);
    std::vector<double> read_file(std::string filename);
    void normalize_inplace(std::vector<double>& input);
    double learning_rate_;
    double delta_;
};

#endif /* EXP3_HH */
