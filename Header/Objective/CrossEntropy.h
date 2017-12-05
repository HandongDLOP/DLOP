// #ifndef CROSSENTROPY_H_
// #define CROSSENTROPY_H_    value
//
// #include "..//Operator.h"
//
// template<typename DTYPE>
// class CrossEntropy : public Operator<DTYPE>{
//public:
//    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;
// private:
//     // Tensor<DTYPE> *m_aSoftmax_Result = NULL;
//     DTYPE m_epsilon   = 0.0;        // for backprop
//     DTYPE m_min_error = 1e-10;
//     DTYPE m_max_error = 1e+10;
//
// public:
//     // Constructor의 작업 순서는 다음과 같다.
//     // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
//     // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (CrossEntropy::Alloc())
//     CrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int epsilon = 1e-30) : Operator<DTYPE>(pInput0, pInput1) {
//         std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
//         this->Alloc(pInput0, pInput1, epsilon);
//     }
//
//     CrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
//         std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
//         this->Alloc(pInput0, pInput1, 1e-30);
//     }
//
//     CrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int epsilon, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
//         std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int, std::string)" << '\n';
//         this->Alloc(pInput0, pInput1, epsilon);
//     }
//
//     ~CrossEntropy() {
//         std::cout << "CrossEntropy::~CrossEntropy()" << '\n';
//     }
//
//     virtual int Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int epsilon = 1e-30) {
//         std::cout << "CrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
//         // if pInput0 and pInput1의 shape가 다르면 abort
//
//         int *shape            = pInput0->GetOutput()->GetShape();
//         Tensor<DTYPE> *output = new Tensor<DTYPE>(shape[0], shape[1], 1, 1, 1);
//         this->SetOutput(output);
//
//         // m_aSoftmax_Result = new Tensor(shape);
//
//         m_epsilon = epsilon;
//
//         return 1;
//     }
//
//     template <typename TENSOR_DTYPE> int ComputeForwardPropagate() {
//         // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';
//
//         int *shape            = this->GetInputOperator()[0]->GetOutput()->GetShape();
//         DTYPE *****input_data = this->GetInputOperator()[0]->GetOutput()->GetData();
//         DTYPE *****label_data = this->GetInputOperator()[1]->GetOutput()->GetData();
//
//         this->GetOutput()->Reset();
//         DTYPE *****output = this->GetOutput()->GetData();
//         // DTYPE *****softmax_result = GetSoftmaxResult()->GetData();
//
//         int Time    = shape[0];
//         int Batch   = shape[1];
//         int Channel = shape[2];
//         int Row     = shape[3];
//         int Col     = shape[4];
//
//         int num_of_output = Channel * Row * Col;
//
//         for (int ti = 0; ti < Time; ti++) {
//             for (int ba = 0; ba < Batch; ba++) {
//                 for (int ch = 0; ch < Channel; ch++) {
//                     for (int ro = 0; ro < Row; ro++) {
//                         for (int co = 0; co < Col; co++) {
//                             output[ti][ba][0][0][0] += cross_entropy(label_data[ti][ba][ch][ro][co],
//                                                                      input_data[ti][ba][ch][ro][co],
//                                                                      num_of_output);
//                         }
//                     }
//                 }
//             }
//         }
//
//         // GetInputOperator()[0]->GetOutput()->PrintData();
//         // GetInputOperator()[1]->GetOutput()->PrintData();
//         // GetOutput()->PrintData();
//
//         return 1;
//     }
//
//     virtual int ComputeBackPropagate() {
//         // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';
//
//         int *shape = this->GetInputOperator()[0]->GetOutput()->GetShape();
//
//         DTYPE *****input_data = this->GetInputOperator()[0]->GetOutput()->GetData();
//         DTYPE *****label_data = this->GetInputOperator()[1]->GetOutput()->GetData();
//
//         this->GetInputOperator()[0]->GetDelta()->Reset();
//         DTYPE *****delta_input_data = this->GetInputOperator()[0]->GetDelta()->GetData();
//
//         int Time    = shape[0];
//         int Batch   = shape[1];
//         int Channel = shape[2];
//         int Row     = shape[3];
//         int Col     = shape[4];
//
//         int num_of_output = Channel * Row * Col;
//
//         for (int ti = 0; ti < Time; ti++) {
//             for (int ba = 0; ba < Batch; ba++) {
//                 for (int ch = 0; ch < Channel; ch++) {
//                     for (int ro = 0; ro < Row; ro++) {
//                         for (int co = 0; co < Col; co++) {
//                             delta_input_data[ti][ba][ch][ro][co] = cross_entropy_derivative(label_data[ti][ba][ch][ro][co], input_data[ti][ba][ch][ro][co], num_of_output);
//                             // std::cout << "target_prediction : " << softmax_result[ti][ba][ch][ro][co] << '\n';
//                         }
//                     }
//                 }
//             }
//         }
//
//         // std::cout << "softmax" << '\n';
//         // GetSoftmaxResult()->PrintData();
//         // std::cout << "answer" << '\n';
//         // GetInputOperator()[1]->GetOutput()->PrintData();
//         // std::cout << "del" << '\n';
//         // GetInputOperator()[0]->GetDelta()->PrintData();
//
//         return 1;
//     }
//
//     DTYPE cross_entropy(DTYPE label, DTYPE prediction, int num_of_output) {
//         // std::cout << "label : " << label <<  '\n';
//         // std::cout << "prediction : " << prediction <<  '\n';
//         // std::cout << "num_of_output : " << num_of_output <<  '\n';
//         // std::cout << "-label *log(prediction) / num_of_output : " << -label *log(prediction) / num_of_output <<  '\n';
//
//         DTYPE error_ = -label *log(prediction + m_epsilon) / num_of_output;
//
//         // if (std::isnan(error_) || (error_ < m_max_error)) {
//         // error_ = m_min_error;
//         // }
//         //
//         // if (std::isinf(error_) || (error_ > m_max_error)) {
//         // error_ = m_max_error;
//         // }
//         //
//         // if (std::isnan(error_)) {
//         // error_ = m_min_error;
//         // }
//         //
//         // if (std::isinf(error_)) {
//         // error_ = m_max_error;
//         // }
//
//
//         return error_;
//     }
//
//     DTYPE cross_entropy_derivative(DTYPE label_data, DTYPE input_data, int num_of_output) {
//         DTYPE delta_ = 0.0;
//
//         // delta_ = -label_data / ((input_data * num_of_output) + m_epsilon);
//         delta_ = -label_data / (input_data + m_epsilon) / num_of_output;
//
//         // if (std::isnan(delta_) || (delta_ < m_max_error)) {
//         // delta_ = m_min_error;
//         // }
//         //
//         // if (std::isinf(delta_) || (delta_ > m_max_error)) {
//         // delta_ = m_max_error;
//         // }
//         //
//         // if (std::isnan(delta_)) {
//         // delta_ = m_min_error;
//         // }
//         //
//         // if (std::isinf(delta_)) {
//         // delta_ = m_max_error;
//         // }
//
//         return delta_;
//     }
// };
//
// #endif  // CROSSENTROPY_H_
