#ifndef CONTEXT_HH
#define CONTEXT_HH

#include <string>
#include <vector>

class Context {
public:
    Context(std::string model_path);
    std::size_t predict(std::vector<double> input);
    std::vector<double> cluster_;
private:
    std::vector<double> weights_;
    double gamma_;
};

#endif /* CONTEXT_HH */
