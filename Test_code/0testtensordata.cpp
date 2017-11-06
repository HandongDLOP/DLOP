#include <iostream>

template <class T>

class Tensor {
private:
    T * data = NULL;
public:
    Tensor (T p_data){
        data = new T(p_data);
    }
    virtual ~Tensor (){}

    T * GetData_address(){
        return data;
    }

    void Printdata(){
        std::cout << data[0] << '\n';
    }
};

int main(int argc, char const *argv[]) {
    Tensor<int> test1(1);

    // 실제로는 데이터 타입을 먼저 확인한다음에 진행한다.

    test1.Printdata();

    int * data = test1.GetData_address();

    *data = 3;

    test1.Printdata();

    return 0;
}
