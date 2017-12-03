#include "Template.h"

template class Temp<int>;
template class Temp<double>;

template<typename T>
Temp<T>::Temp (){
    std::cout << "Temp<T>::Temp" << '\n';
}
template<typename T>
Temp<T>::~Temp (){
    std::cout << "Temp<T>::~Temp" << '\n';
}

template<typename T>
void Temp<T>::Print(){
    std::cout << "Print" << '\n';
}
