#ifndef EXP3_KMEANS_HH
#define EXP3_KMEANS_HH

#include <string>
#include <vector>
#include <map>
#include <fstream>

#include "exp3.hh"

class Exp3Kmeans : public Exp3
{
public:
    Exp3Kmeans(std::string model_path, std::string kmeans_path, double delta, double lr);
    std::tuple<std::size_t, std::size_t> predict(std::vector<double> input, std::size_t curr_buffer_, std::size_t last_format_, std::vector<std::size_t> available_actions_idx);

private:
    std::map<int, std::vector<double>> clusters_;
    std::vector<double> mean_;
    std::vector<double> std_;

    double delta_;
    void load_clusters(fs::path kmeans_path);
    double dist(std::vector<double> v1, std::vector<double> v2);
    std::vector<double> read_file(fs::path filename);
    void normalize_inplace(std::vector<double> &input);
};

#endif /* EXP3_KMEANS_HH */
