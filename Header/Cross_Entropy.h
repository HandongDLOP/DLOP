#ifndef CROSS_ENTROPY_H_
#define CROSS_ENTROPY_H_    value

#include <iostream>
// #include <string>
#include <math.h>

// #include "Tensor.h"
#include "Operator.h"

class Cross_Entropy : public Operator {
private:
    // Tensor *m_aSoftmax_Result = NULL;
    double m_epsilon   = 0.0;        // for backprop
    double m_min_error = 1e-10;
    double m_max_error = 1e+10;

public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Cross_Entropy::Alloc())
    Cross_Entropy(Operator *pInput1, Operator *pInput2, int epsilon = 1e-30) : Operator(pInput1, pInput2) {
        std::cout << "Cross_Entropy::Cross_Entropy(Operator *, Operator *, int)" << '\n';
        Alloc(pInput1, pInput2, epsilon);
    }

    Cross_Entropy(Operator *pInput1, Operator *pInput2, std::string pName) : Operator(pInput1, pInput2, pName) {
        std::cout << "Cross_Entropy::Cross_Entropy(Operator *, Operator *, std::string)" << '\n';
        Alloc(pInput1, pInput2, 1e-30);
    }

    Cross_Entropy(Operator *pInput1, Operator *pInput2, int epsilon, std::string pName) : Operator(pInput1, pInput2, pName) {
        std::cout << "Cross_Entropy::Cross_Entropy(Operator *, Operator *, int, std::string)" << '\n';
        Alloc(pInput1, pInput2, epsilon);
    }

    virtual ~Cross_Entropy() {
        std::cout << "Cross_Entropy::~Cross_Entropy()" << '\n';
    }

    virtual bool Alloc(Operator *pInput1, Operator *pInput2, int epsilon = 1e-30) {
        std::cout << "Cross_Entropy::Alloc(Operator *, Operator *, int)" << '\n';
        // if pInput1 and pInput2의 shape가 다르면 abort

        int *shape     = GetInputOperator()[0]->GetOutput()->GetShape();
        Tensor *output = new Tensor(shape[0], shape[1], 1, 1, 1);
        SetOutput(output);

        // m_aSoftmax_Result = new Tensor(shape);

        m_epsilon = epsilon;

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int *shape             = GetInputOperator()[0]->GetOutput()->GetShape();
        double *****input_data = GetInputOperator()[0]->GetOutput()->GetData();
        double *****label_data = GetInputOperator()[1]->GetOutput()->GetData();

        GetOutput()->Reset();
        double *****output = GetOutput()->GetData();
        // double *****softmax_result = GetSoftmaxResult()->GetData();

        int Time    = shape[0];
        int Batch   = shape[1];
        int Channel = shape[2];
        int Row     = shape[3];
        int Col     = shape[4];

        int num_of_output = Channel * Row * Col;

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            output[ti][ba][0][0][0] += cross_entropy(label_data[ti][ba][ch][ro][co],
                                                                     input_data[ti][ba][ch][ro][co],
                                                                     num_of_output);
                        }
                    }
                }
            }
        }

        // GetInputOperator()[0]->GetOutput()->PrintData();
        // GetInputOperator()[1]->GetOutput()->PrintData();
        // GetOutput()->PrintData();

        return true;
    }

    virtual bool ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int *shape = GetInputOperator()[0]->GetOutput()->GetShape();

        double *****input_data = GetInputOperator()[0]->GetOutput()->GetData();
        double *****label_data = GetInputOperator()[1]->GetOutput()->GetData();

        GetInputOperator()[0]->GetDelta()->Reset();
        double *****delta_input_data = GetInputOperator()[0]->GetDelta()->GetData();

        int Time    = shape[0];
        int Batch   = shape[1];
        int Channel = shape[2];
        int Row     = shape[3];
        int Col     = shape[4];

        int num_of_output = Channel * Row * Col;

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            delta_input_data[ti][ba][ch][ro][co] = cross_entropy_derivative(label_data[ti][ba][ch][ro][co], input_data[ti][ba][ch][ro][co], num_of_output);
                            // std::cout << "target_prediction : " << softmax_result[ti][ba][ch][ro][co] << '\n';
                        }
                    }
                }
            }
        }

        // std::cout << "softmax" << '\n';
        // GetSoftmaxResult()->PrintData();
        // std::cout << "answer" << '\n';
        // GetInputOperator()[1]->GetOutput()->PrintData();
        // std::cout << "del" << '\n';
        // GetInputOperator()[0]->GetDelta()->PrintData();

        return true;
    }

    double cross_entropy(double label, double prediction, int num_of_output) {
        // std::cout << "label : " << label <<  '\n';
        // std::cout << "prediction : " << prediction <<  '\n';
        // std::cout << "num_of_output : " << num_of_output <<  '\n';
        // std::cout << "-label *log(prediction) / num_of_output : " << -label *log(prediction) / num_of_output <<  '\n';

        double error_ = -label *log(prediction + m_epsilon) / num_of_output;

        // if (std::isnan(error_) || (error_ < m_max_error)) {
        // error_ = m_min_error;
        // }
        //
        // if (std::isinf(error_) || (error_ > m_max_error)) {
        // error_ = m_max_error;
        // }

        if (std::isnan(error_)) {
            error_ = m_min_error;
        }

        if (std::isinf(error_)) {
            error_ = m_max_error;
        }


        return error_;
    }

    double cross_entropy_derivative(double label_data, double input_data, int num_of_output) {
        double delta_ = 0.0;

        delta_ = -label_data / ((input_data * num_of_output) + m_epsilon);

        // if (std::isnan(delta_) || (delta_ < m_max_error)) {
        // delta_ = m_min_error;
        // }
        //
        // if (std::isinf(delta_) || (delta_ > m_max_error)) {
        // delta_ = m_max_error;
        // }

        if (std::isnan(delta_)) {
            delta_ = m_min_error;
        }

        if (std::isinf(delta_)) {
            delta_ = m_max_error;
        }

        return delta_;
    }
};

#endif  // CROSS_ENTROPY_H_
