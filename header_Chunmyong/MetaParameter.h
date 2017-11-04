#ifndef META_PARAMETER_H_
#define META_PARAMETER_H_    value

#include <string>

#include "Shape.h"
#include "Tensor.h"

class MetaParameter {
public:
    // 정의
    MetaParameter();
    virtual ~MetaParameter();
};

class ConvParam : public MetaParameter {
private:
    Tensor *filter;
    Shape *stride;
    void *padding;
    std::string m_name     = NULL;
    void *data_format      = NULL;
    void *use_cudnn_on_gpu = NULL;

public:
    ConvParam() {}

    virtual ~ConvParam() {}
};

#endif  // META_PARAMETER_H_
