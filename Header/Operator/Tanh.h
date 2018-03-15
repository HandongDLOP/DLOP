#ifndef TANH_H_
#define TANH_H_   value

#include "..//Operator.h"
#include <stdio.h>

template<typename DTYPE>
class Tanh :public Operator<DTYPE>{
public:
  FILE *fp;

  Tanh(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName){
    std::cout<<"Tanh::Tanh(Operator *, string)" << '\n';
    this->Alloc(pInput);
  }

  ~Tanh(){
    std::cout<< "Tanh::~Tanh()" << '\n';
  //  fclose(fp);
  }

  int Alloc(Operator<DTYPE> *pInput){
    std::cout<< "Tanh::Alloc(Operator *)" << '\n';
    // shape
    Shape *shapeOfResult = new Shape(pInput->GetResult()->GetShape());
    // create tanh result
    this->SetResult(new Tensor<DTYPE>(shapeOfResult));

    Shape *shapeOfDelta = new Shape(pInput->GetResult()->GetShape());
    this->SetDelta(new Tensor<DTYPE>(shapeOfDelta));

  //  fp = fopen("tanh Test", "w");
  //  if(fp == NULL){
  //    printf("FILEOPEN ERROR\n");
  //    return 0;
  //  }

    return TRUE;
  }

  int ComputeForwardPropagate(){
    printf("ComputeForwardPropagate()\n");
    Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();
    int capacity          = input->GetData()->GetCapacity();

    for (int i = 0; i < capacity; i++){
      printf("(*input)[%d] : %f\n", i, (*input)[i]);
    }

    for (int i = 0; i < capacity; i++){
      (*result)[i] = this->TANH((*input)[i]);
      printf("(*result)[%d] : %f\n", i, (*result)[i]);
    //  fprintf(fp, "%lf", (*result)[i]);
      //fp << (*result)[i] << ";" <<endl;
    }

    return TRUE;
  }

  int ComputeBackPropagate(){
    std::cout << "ComputeBackwardPropagate()\n" << '\n';
    Tensor<DTYPE> *result      = this->GetResult();
    Tensor<DTYPE> *this_delta  = this->GetDelta();
    Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
    int capacity               = result->GetData()->GetCapacity();

    for (int i = 0; i < capacity; i++){
      printf("%d\n", i);
      printf("(*result)[%d] : %f", i, (*result)[i]);
      // std::cout << result << '\n';
      (*input_delta)[i] = (1 - ((*result)[i] * (*result)[i])) * (*this_delta)[i];
    }

    return TRUE;
  }

  inline DTYPE TANH(DTYPE data){
    printf("TANH(DTYPE)\n");
    //DTYPE sinh = ((DTYPE)exp(data) - (DTYPE)exp(-data)) / 2;
    //DTYPE cosh = ((DTYPE)exp(data) + (DTYPE)exp(-data)) / 2;
    //return sinh/cosh;
    return ((DTYPE)exp(data) - (DTYPE)exp(-data)) / ((DTYPE)exp(data) + (DTYPE)exp(-data));
  }
};

#endif // TANH_H_

//int main(int argc, char const *argv[]){
  //Tensor<int> *temp = new Tensor<int>(1, 100, 1, 28, 28);

//  Tensor<int> *temp = Tensor<int>::Constant(1, 1, 1, 1, 1, 3);
//}
