#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include "filesystem.hh"
#include "exp3.hh"

Exp3::Exp3() {}

Exp3::Exp3(std::string model_path) : model_path_(model_path)
{
  reload_model();
}

Exp3::Exp3(std::string model_path, std::string norm_path) : model_path_(model_path)
{
  reload_model();

  /* load mean & std */
  mean_ = read_file(norm_path + "mean.txt");
  std_ = read_file(norm_path + "std.txt");
}

std::vector<double> Exp3::read_file(std::string filename)
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

void Exp3::normalize_inplace(std::vector<double> &input)
{
  assert(input.size() == mean_.size());
  assert(input.size() == std_.size());

  for (std::size_t i = 0; i < input.size(); i++)
  {
    if (i < input.size() - 2)
    {
      input[i] *= 1 - DELTA;
    }
    else
    {
      input[i] *= DELTA;
    }

    input[i] = (input[i] - mean_[i]) / std_[i];
  }
}

std::size_t Exp3::predict(std::vector<double> input, std::size_t curr_buffer_, std::size_t last_format_)
{
  input.push_back(curr_buffer_);
  input.push_back(last_format_);

  normalize_inplace(input);

  /* find cluster index */
  Context &min_context = contexts_.back();
  for (const auto &context : contexts_)
  {
    // std::cout << "dist:" << dist(input, context.cluster_) << std::endl;

    if (dist(input, context.cluster_) < dist(input, min_context.cluster_))
    {
      min_context = context;
    }
  }

  std::cout << "min cluster:" << &min_context << std::endl;

  return min_context.predict(input);
}

double Exp3::dist(std::vector<double> v1, std::vector<double> v2)
{
  assert(v1.size() == v2.size());
  double distance = 0;

  for (std::size_t i = 0; i < v1.size(); i++) //todo: change to v1.size()
  {
    if (std::pow(v1[i] - v2[i], 2) > 1e10) {
      distance += 1e10;
      // std::cout << "input:" << v1[i] << "cluster: " << v2[i] << std::endl;
      continue;
    }
  
    distance += std::pow(v1[i] - v2[i], 2);
  }

  return std::sqrt(distance);
}

std::size_t Exp3::reload_model()
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

  std::size_t version = stoi(exp3_relevant_dir.substr(exp3_relevant_dir.find_last_of('/') + 1, exp3_relevant_dir.length()));
  version++;

  // std::cout << "weights dir: " << exp3_relevant_dir << " version:" << version << std::endl;

  for (const auto &entry : fs::directory_iterator(exp3_relevant_dir))
  {
    contexts_.push_back(Context(entry.path()));
  }

  return version;
}