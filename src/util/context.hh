#ifndef CONTEXT_HH
#define CONTEXT_HH

#include <string>
#include <vector>
#include <random>

class Context {
public:
    Context(std::string model_path, double learning_rate);
    std::size_t predict(std::vector<double> input);
    std::vector<double> cluster_;
    std::string model_path_;
private:
    std::vector<double> read_file(std::string filename);
    std::vector<double> weights_;
    double gamma_;
    std::default_random_engine generator_ {};
    double learning_rate_;
};

#endif /* CONTEXT_HH */
