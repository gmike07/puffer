#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <fstream>

#include "filesystem.hh"
#include "exp3.hh"
#include "json.hpp"

using json = nlohmann::json;

Exp3::Exp3(std::string model_path) : model_path_(model_path)
{
  reload_model();
}

Exp3::Exp3(std::string model_path, double lr) : model_path_(model_path), lr_(lr)
{
  reload_model();
}

bool Exp3::should_reload()
{
  std::vector<fs::path> dirs;
  for (const auto &entry : fs::directory_iterator(model_path_))
  {
    dirs.push_back(entry.path());
  }

  std::sort(dirs.begin(), dirs.end(), [](const fs::path f1, const fs::path f2)
            {
              std::string s1 = f1.c_str();
              std::string s2 = f2.c_str();
              int n1 = stoi(s1.substr(s1.find_last_of('/') + 1, s1.length()));
              int n2 = stoi(s2.substr(s2.find_last_of('/') + 1, s2.length()));
              return n1 < n2;
            });

  std::string exp3_relevant_dir = dirs.front().c_str();

  std::size_t version = stoi(exp3_relevant_dir.substr(exp3_relevant_dir.find_last_of('/') + 1, exp3_relevant_dir.length()));

  return version + 1 != version_;
}

void Exp3::reload_model()
{
  num_of_arms_ = 0;
  contexts_.clear();

  std::vector<fs::path> dirs;
  for (const auto &entry : fs::directory_iterator(model_path_))
  {
    dirs.push_back(entry.path());
  }

  std::sort(dirs.begin(), dirs.end(), [](const fs::path f1, const fs::path f2)
            {
              std::string s1 = f1.c_str();
              std::string s2 = f2.c_str();
              int n1 = stoi(s1.substr(s1.find_last_of('/') + 1, s1.length()));
              int n2 = stoi(s2.substr(s2.find_last_of('/') + 1, s2.length()));
              return n1 < n2;
            });

  std::string exp3_relevant_dir = dirs.front().c_str();

  std::size_t version = stoi(exp3_relevant_dir.substr(exp3_relevant_dir.find_last_of('/') + 1, exp3_relevant_dir.length()));
  version_ = version + 1;

  std::ifstream ifs_weights(dirs.front() / "weights.json");
  json json_weights = json::parse(ifs_weights);

  std::ifstream ifs_gamma(dirs.front() / "gamma.json");
  json json_gamma = json::parse(ifs_gamma);

  for (auto &el : json_weights.items())
  {
    auto weights = el.value().get<std::vector<double>>();
    auto gamma = json_gamma[el.key()].get<double>();

    if (num_of_arms_ == 0)
    {
      num_of_arms_ = weights.size();
    }

    assert(weights.size() == num_of_arms_);
    contexts_[stoi(el.key())] = Context(weights, gamma, lr_);
  }

  std::cout << "weights dir: " << dirs.front() << " version:" << version_ << std::endl;
}
