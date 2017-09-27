#ifndef Manna_H_
#define Manna_H_

#include <iostream>
#include <string>
#include "Shape.h"

class Manna {
private:
    Shape dimension;
    char type;  // 추후 수정 예정
    void *data;

public:
    Manna();
    virtual ~Manna();

    Shape GetShape();

    //Initialization(const std::string &type = "default");
};

#endif  // Manna_H_
