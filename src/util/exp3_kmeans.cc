#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <fstream>

#include "filesystem.hh"
#include "exp3_kmeans.hh"
#include "json.hpp"

using json = nlohmann::json;

Exp3Kmeans::Exp3Kmeans(std::string model_path,
                       std::string kmeans_path,
                       double delta,
                       double lr) : Exp3(model_path, lr), delta_(delta)
{
    load_clusters(fs::path(kmeans_path) / "clusters.json");
    mean_ = read_file(fs::path(kmeans_path) / "mean.txt");
    std_ = read_file(fs::path(kmeans_path) / "std.txt");
}

std::vector<double> Exp3Kmeans::read_file(fs::path filename)
{
    std::ifstream file(filename);
    std::vector<double> result;
    double a;

    while (file >> a)
    {
        result.push_back(a);
    }

    return result;
}

void Exp3Kmeans::normalize_inplace(std::vector<double> &input)
{
    assert(input.size() == mean_.size());
    assert(input.size() == std_.size());

    for (std::size_t i = 0; i < input.size(); i++)
    {
        if (i < input.size() - 2)
        {
            input[i] *= 1 - delta_;
        }
        else
        {
            input[i] *= delta_;
        }

        input[i] = (input[i] - mean_[i]) / std_[i];
    }
}

std::tuple<std::size_t, std::size_t> Exp3Kmeans::predict(
    std::vector<double> input,
    std::size_t curr_buffer_,
    std::size_t last_format_,
    std::vector<std::size_t> available_actions_idx)
{
    input.push_back(curr_buffer_);
    input.push_back(last_format_);

    normalize_inplace(input);

    // std::cout.precision(17);
    // std::cout << "datapoint: ";
    // for (auto &i: input)
    //   std::cout << i << "\t";
    // std::cout << "###############" << std::endl;

    // find min cluster
    const auto &min_cluster = *std::min_element(
        clusters_.begin(),
        clusters_.end(),
        [&](std::pair<int, std::vector<double>> elem1, std::pair<int, std::vector<double>> elem2)
        {
            // std::cout << "dist of " << elem1.first << "-" << elem2.first << " : " << dist(input, elem1.second) << ", " << dist(input, elem2.second) << std::endl;
            return dist(input, elem1.second) < dist(input, elem2.second);
        });

    std::size_t arm = contexts_[min_cluster.first].predict();

    if (arm > available_actions_idx.back())
    {
        arm = available_actions_idx.back();
    }
    else if (arm < available_actions_idx.front())
    {
        arm = available_actions_idx.front();
    }

    return std::make_tuple(arm, min_cluster.first);
}

double Exp3Kmeans::dist(std::vector<double> v1, std::vector<double> v2)
{
    assert(v1.size() == v2.size());
    double distance = 0;

    for (std::size_t i = 0; i < v1.size(); i++)
    {
        // if (std::pow(v1[i] - v2[i], 2) > 1e4) {
        //   distance += 1e10;
        //   std::cout << "input:" << v1[i] << "cluster: " << v2[i] << std::endl;
        //   continue;
        // }

        distance += std::pow(v1[i] - v2[i], 2);
    }

    return std::sqrt(distance);
}

void Exp3Kmeans::load_clusters(fs::path clusters_path)
{
    std::ifstream ifs(clusters_path);
    json json_data = json::parse(ifs);

    for (auto &el : json_data.items())
    {
        auto cluster = el.value().get<std::vector<double>>();
        clusters_[stoi(el.key())] = cluster;
    }
}