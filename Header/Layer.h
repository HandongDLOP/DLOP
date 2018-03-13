#ifndef __LAYER__
#define __LAYER__    value

#include "Optimizer_utils.h"

template<typename DTYPE> class Layer {
private:
    Container<Operator<DTYPE> *> *m_aaOperator;
    Container<Tensorholder<DTYPE> *> *m_aaParameter;
    Container<Layer<DTYPE> *> *m_aaLayer;

    int m_numOfOperator;
    int m_numOfParameter;
    int m_numOfLayer;

    std::string m_name;

public:
    Layer(std::string pName = "No Name");
    virtual ~Layer();

    int  Alloc();
    void Delete();

    // =======

    Operator<DTYPE>    * AddLayer(Layer<DTYPE> *pLayer);
    Operator<DTYPE>    * AddOperator(Operator<DTYPE> *pOperator);
    Tensorholder<DTYPE>* AddParameter(Tensorholder<DTYPE> *pParameter);

    // =======

    Container<Operator<DTYPE> *>    * GetOperatorContainer();
    Container<Tensorholder<DTYPE> *>* GetParameterContainer();
    int                               GetNumOfOperator();
    int                               GetNumOfParameter();
    std::string                       GetName();

    Operator<DTYPE>                 * PopOperator();
    Tensorholder<DTYPE>             * PopParameter();
};

#endif  // ifndef __LAYER__
