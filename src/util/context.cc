#include "context.hh"
#include "filesystem.hh"

#include <exception>
#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>

Context::Context(std::vector<double> weights, double gamma, double lr)
    : weights_(weights), gamma_(gamma), lr_(lr)
{
}

std::size_t Context::predict()
{
  std::vector<double> probs;

  double sum_of_weights = 0;
  for (double weight : weights_)
  {
    sum_of_weights += weight;
  }

  for (double weight : weights_)
  {
    probs.push_back(weight / sum_of_weights);
  }

  std::discrete_distribution<int> distribution(probs.begin(), probs.end());
  int arm = distribution(generator_);

  // std::cout << "probs: ";
  // for (double elem : probs)
  // {
  //   std::cout << "," << elem;
  // }
  // std::cout << std::endl;

  return arm;
}
