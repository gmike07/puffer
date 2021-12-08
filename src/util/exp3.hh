#ifndef EXP3_HH
#define EXP3_HH

#include <string>
#include <vector>
#include <map>

#include "context.hh"

class Exp3
{
public:
    Exp3(std::string model_path, double lr);
    Exp3(std::string model_path);
    virtual void reload_model();
    std::size_t version_;
    bool should_reload();
protected:
    std::string model_path_;
    std::map<int, Context> contexts_;
    std::size_t num_of_arms_;
    double lr_;
};

#endif /* EXP3_HH */
