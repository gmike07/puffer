#include "context.hh"

#include <exception>
#include <iostream>
#include <fstream>
#include <math.h>

Context::Context(std::string model_path)
{
  cluster_ = read_file(model_path + "/" + "cluster.txt");
  weights_ = read_file(model_path + "/" + "weights.txt");
  gamma_ = read_file(model_path + "/" + "gamma.txt").back();
}

std::vector<double> Context::read_file(std::string filename)
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

std::size_t Context::predict(std::vector<double> input)
{
  std::vector<double> probs;
  for (double weight : weights_)
  {
    probs.push_back(exp(gamma_ * weight));
  }

  std::discrete_distribution<int> distribution(probs.begin(), probs.end());

  int arm = distribution(generator_);

  // std::cout << "arm: " << arm << std::endl;

  return arm;
}