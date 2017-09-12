#ifndef FACTORY_H_
#define FACTORY_H_    value

#include <string>
#include <iostream>
#include "Activation.h"

namespace dlop {
namespace Factory {
Activation* ActivationFactory(const std::string& type) {
    if (type == "ReLu") return new Relu();

    if ((type == "default") || (type == "Linear")) return new Linear();

    // if (type == "Sigmoid")

    // if (type == "")

    else return NULL;
}
}
}
#endif  // FACTORY_H_
