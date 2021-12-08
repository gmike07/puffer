#ifndef CONTEXT_HH
#define CONTEXT_HH

#include <string>
#include <vector>
#include <random>

class Context
{
public:
    Context() {}
    Context(std::vector<double> weights, double gamma, double lr);
    std::size_t predict();

private:
    std::vector<double> weights_;
    double gamma_;
    std::default_random_engine generator_{};
    double lr_;

    static constexpr double MIN_PROBABILITY_WEIGHT = 0.01;
    static constexpr double MIN_PROBABILITY = 1e-100;

    double min_probability_weight_{MIN_PROBABILITY_WEIGHT};
    double min_probability_{MIN_PROBABILITY};
};

#endif /* CONTEXT_HH */
