#ifndef META_PARAMETER_H_
#define META_PARAMETER_H_    value

#include <String>

#include "Shape.h"
#include "Ark.h"

class MetaParameter {
public:
    // 정의
    MetaParameter();
    virtual ~MetaParameter();
};

class ConvParam : public MetaParameter {
private:
    Ark *filter;
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
