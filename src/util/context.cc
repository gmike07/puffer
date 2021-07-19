#include "context.hh"

#include <exception>
#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>

Context::Context(std::string model_path, double learning_rate) : learning_rate_(learning_rate)
{
  cluster_ = read_file(model_path + "/" + "cluster.txt");
  weights_ = read_file(model_path + "/" + "weights.txt");
  gamma_ = read_file(model_path + "/" + "gamma.txt").back();

  // std::cout << std::setprecision(10) << std::scientific << "check: " << d << std::endl;
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

  // std::cout << "weights: ";
  for (double weight : weights_)
  {
    // std::cout << "," << weight;
    probs.push_back((1 - learning_rate_) * exp(gamma_ * weight) + learning_rate_ * 1 / weights_.size());
  }
  // std::cout << std::endl;   

  std::discrete_distribution<int> distribution(probs.begin(), probs.end());

  int arm = distribution(generator_);

  std::cout << "probs: ";
  for (double elem: probs) {
    std::cout << "," << elem;
  }                          

  std::cout << std::endl;   
  
  // std::cout << "arm: " << arm << std::endl;

  return arm;
}