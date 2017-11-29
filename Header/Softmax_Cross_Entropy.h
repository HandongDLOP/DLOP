#ifndef SOFTMAX_CROSS_ENTROPY_H_
#define SOFTMAX_CROSS_ENTROPY_H_    value

#include <iostream>
// #include <string>
#include <math.h>

// #include "Tensor.h"
#include "Operator.h"

class Softmax_Cross_Entropy : public Operator {
private:
    Tensor *m_aSoftmax_Result = NULL;
    double m_epsilon          = 0.0; // for backprop
    double m_min_error        = 1e-40;
    double m_max_error        = 1e+40;

public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Softmax_Cross_Entropy::Alloc())
    Softmax_Cross_Entropy(Operator *pInput1, Operator *pInput2, double epsilon = 1e-20) : Operator(pInput1, pInput2) {
        std::cout << "Softmax_Cross_Entropy::Softmax_Cross_Entropy(Operator *, Operator *, int)" << '\n';
        Alloc(pInput1, pInput2, epsilon);
    }

    Softmax_Cross_Entropy(Operator *pInput1, Operator *pInput2, std::string pName) : Operator(pInput1, pInput2, pName) {
        std::cout << "Softmax_Cross_Entropy::Softmax_Cross_Entropy(Operator *, Operator *, std::string)" << '\n';
        Alloc(pInput1, pInput2, 1e-20);
    }

    Softmax_Cross_Entropy(Operator *pInput1, Operator *pInput2, double epsilon, std::string pName) : Operator(pInput1, pInput2, pName) {
        std::cout << "Softmax_Cross_Entropy::Softmax_Cross_Entropy(Operator *, Operator *, int, std::string)" << '\n';
        Alloc(pInput1, pInput2, epsilon);
    }

    virtual ~Softmax_Cross_Entropy() {
        std::cout << "Softmax_Cross_Entropy::~Softmax_Cross_Entropy()" << '\n';

        delete m_aSoftmax_Result;
    }

    virtual bool Alloc(Operator *pInput1, Operator *pInput2, double epsilon = 1e-20) {
        std::cout << "Softmax_Cross_Entropy::Alloc(Operator *, Operator *, int)" << '\n';
        // if pInput1 and pInput2의 shape가 다르면 abort

        int *shape     = GetInputOperator()[0]->GetOutput()->GetShape();
        Tensor *output = new Tensor(shape[0], shape[1], 1, 1, 1);
        SetOutput(output);

        m_aSoftmax_Result = new Tensor(shape);

        m_epsilon = epsilon;

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int *shape             = GetInputOperator()[0]->GetOutput()->GetShape();
        double *****input_data = GetInputOperator()[0]->GetOutput()->GetData();
        double *****label_data = GetInputOperator()[1]->GetOutput()->GetData();

        GetOutput()->Reset();
        double *****output         = GetOutput()->GetData();
        double *****softmax_result = GetSoftmaxResult()->GetData();

        int Time    = shape[0];
        int Batch   = shape[1];
        int Channel = shape[2];
        int Row     = shape[3];
        int Col     = shape[4];

        double sum[Time][Batch] = { 0.0 };
        double max[Time][Batch] = { 0.0 };
        int    num_of_output    = Channel * Row * Col;

        double temp = 0.0;

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                max[ti][ba] = Max(input_data[ti][ba], Channel, Row, Col);

                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            // std::cout << (exp(input_data[ti][ba][ch][ro][co] - max[ti][ba]) + m_epsilon) << '\n';
                            temp += (exp(input_data[ti][ba][ch][ro][co] - max[ti][ba]) + m_epsilon);
                        }
                    }
                }
                // 부동소수점 문제가 있는 듯 함 - 중요
                sum[ti][ba] = temp;
                temp = 0.0;
            }
        }

        // for (int ti = 0; ti < Time; ti++) {
        //     for (int ba = 0; ba < Batch; ba++) {
        //         sum[ti][ba];
        //         // std::cout << sum[ti][ba] << ' ';
        //     }
        // }


        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            // std::cout << (exp(input_data[ti][ba][ch][ro][co] - max[ti][ba]) + m_epsilon) << ", " << sum[ti][ba] << '\n';
                            softmax_result[ti][ba][ch][ro][co] = (exp(input_data[ti][ba][ch][ro][co] - max[ti][ba]) + m_epsilon) / sum[ti][ba];
                            output[ti][ba][0][0][0]           += cross_entropy(label_data[ti][ba][ch][ro][co],
                                                                               softmax_result[ti][ba][ch][ro][co],
                                                                               num_of_output);
                        }
                    }
                }
            }
        }

        // GetSoftmaxResult()->PrintData();

        // GetInputOperator()[0]->GetOutput()->PrintData();
        // GetInputOperator()[1]->GetOutput()->PrintData();
        // GetOutput()->PrintData();

        return true;
    }

    virtual bool ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int *shape = GetInputOperator()[0]->GetOutput()->GetShape();

        double *****label_data     = GetInputOperator()[1]->GetOutput()->GetData();
        double *****softmax_result = GetSoftmaxResult()->GetData();

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
                            delta_input_data[ti][ba][ch][ro][co] = softmax_cross_entropy_derivative(label_data[ti][ba][ch][ro][co],
                                                                                                    softmax_result[ti][ba][ch][ro][co],
                                                                                                    num_of_output);
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

    double Max(double ***data, int Channel, int Row, int Col) {
        // initialize
        double max = data[0][0][0];

        for (int ch = 0; ch < Channel; ch++) {
            for (int ro = 0; ro < Row; ro++) {
                for (int co = 0; co < Col; co++) {
                    if (data[ch][ro][co] > max) max = data[ch][ro][co];
                }
            }
        }

        // std::cout << max << '\n';

        return max;
    }

    double cross_entropy(double label, double prediction, int num_of_output) {
        double error_ = -label *log(prediction + m_epsilon) / num_of_output;

        // if (std::isnan(error_) || (error_ < m_max_error)) {
        // error_ = m_min_error;
        // }
        //
        // if (std::isinf(error_) || (error_ > m_max_error)) {
        // error_ = m_max_error;
        // }

        // if (std::isnan(error_)) {
        // error_ = m_min_error;
        // }
        //
        // if (std::isinf(error_)) {
        // error_ = m_max_error;
        // }

        return error_;
    }

    double softmax_cross_entropy_derivative(double label_data, double prediction, int num_of_output) {
        double delta_ = -label_data * (1 - prediction) / num_of_output;

        // if (std::isnan(delta_)) {
        // delta_ = m_min_error;
        // }
        //
        // if (std::isinf(delta_)) {
        // delta_ = m_max_error;
        // }

        return delta_;
    }

    Tensor* GetSoftmaxResult() {
        return m_aSoftmax_Result;
    }
};

#endif  // SOFTMAX_CROSS_ENTROPY_H_
