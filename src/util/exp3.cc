#include <algorithm>
#include <assert.h>
#include <cmath>

#include "filesystem.hh"
#include "exp3.hh"

Exp3::Exp3(){}

Exp3::Exp3(std::string model_path) : model_path_(model_path)
{
  reload_model();
}

std::size_t Exp3::predict(std::vector<double> input)
{
  /* find cluster index */
  Context &min_context = contexts_[0];
  for (const auto &context : contexts_)
  {
    if (dist(input, context.cluster_) < dist(input, min_context.cluster_))
    {
      min_context = context;
    }
  }

  return min_context.predict(input);
}

double Exp3::dist(std::vector<double> v1, std::vector<double> v2)
{
  double distance = 0;
  assert(v1.size() == v2.size());

  for (std::size_t i = 0; i < v1.size(); i++) {
    distance += std::pow(v1[i] - v2[i], 2);
  }

  return std::sqrt(distance);
}

void Exp3::reload_model()
{
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
         int n1 = stoi(s1.substr(0, s1.find_last_of('.')));
         int n2 = stoi(s2.substr(0, s2.find_last_of('.')));
         return n1 < n2;
       });

  std::string exp3_relevant_dir = dirs.back().c_str();

  for (const auto &entry : fs::directory_iterator(exp3_relevant_dir))
  {
    contexts_.push_back(Context(model_path_ / entry.path()));
  }
}