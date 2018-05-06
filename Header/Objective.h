#ifndef Objective_H_
#define Objective_H_

#include "Operator_utils.h"

template<typename DTYPE> class Objective {
private:
    Tensor<DTYPE> *m_aResult;
    Tensor<DTYPE> *m_aGradient;

    Operator<DTYPE> *m_pInputOperator;
    Tensor<DTYPE> *m_pInputTensor;

    Operator<DTYPE> *m_pLabel;

    std::string m_name;

    Device m_Device;

    int m_numOfThread;

public:
    Objective(std::string pName = "NO NAME");
    Objective(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName = "NO NAME");

    virtual ~Objective();

    virtual int            Alloc(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel);
    virtual void           Delete();

    void                   SetResult(Tensor<DTYPE> *pTensor);
    void                   SetGradient(Tensor<DTYPE> *pTensor);

    Tensor<DTYPE>        * GetResult() const;
    Tensor<DTYPE>        * GetGradient() const;
    Operator<DTYPE>      * GetOperator() const;
    Tensor<DTYPE>        * GetTensor() const;
    Operator<DTYPE>      * GetLabel() const;
    std::string            GetName() const;

    // For Propagate
    virtual Tensor<DTYPE>* ForwardPropagate();
    virtual Tensor<DTYPE>* ForwardPropagate(int pTime, int pThreadNum); //

    // For BackPropagate
    virtual Tensor<DTYPE>* BackPropagate();
    virtual Tensor<DTYPE>* BackPropagate(int pTime, int pThreadNum); //

    DTYPE                & operator[](unsigned int index);

    virtual void SetDeviceCPU(); //
    virtual void SetDeviceCPU(int pNumOfThread); //
#ifdef __CUDNN__
    virtual void SetDeviceGPU(); //

#endif  // if __CUDNN__

    virtual Device GetDevice() {
        return m_Device;
    }

    int GetNumOfThread() {
        return m_numOfThread;
    }

    // reset value
    int ResetResult();
    int ResetGradient();
};

#endif  // Objective_H_
