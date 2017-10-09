#ifndef Ark_H_
#define Ark_H_

#include <iostream>
#include <string>
#include "Shape.h"

class Ark {
private:
    Shape dimension;
    char type;  // 추후 수정 예정
    void *data;

public:
    Ark();
    virtual ~Ark();

    Shape GetShape();

    //Initialization(const std::string &type = "default");
};

#endif  // Ark_H_
