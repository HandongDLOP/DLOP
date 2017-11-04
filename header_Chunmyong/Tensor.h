#ifndef Tensor_H_
#define Tensor_H_

#include <iostream>
#include <string>
#include "Shape.h"

class Tensor {
private:
    Shape dimension;
    char type;  // 추후 수정 예정
    void *data;

public:
    Tensor();
    virtual ~Tensor();

    Shape GetShape();



    //Initialization(const std::string &type = "default");
};

#endif  // Tensor_H_
