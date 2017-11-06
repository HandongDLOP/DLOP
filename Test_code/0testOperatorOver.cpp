#include <iostream>

class Test {
private:
    /* data */
    int data = 0;

public:
    Test(int i) {
        data = i;
    }

    Test(const Test &rhs){
        data = rhs.data;

    }

    virtual ~Test() {}

    operator int() {
        return data;
    }


    // +
    Test operator+(const Test &rhs){
        Test result(0);

        result.data = this->data + rhs.data;

        // 이런식으로 완전히 새로운 객체를 반환하는 방법이 존재한다
        // TF에서 relu 안에 값을 넣는 방법을 사용할 수 있었던 부분에 대해서 조금 더 고민해보면 답이 나올 것 같다

        return result;
    }

    // =
    Test& operator=(const Test &rhs){
        data = rhs.data;

        return *this;
    }

    void Print(){
        std::cout << "data : " << data << '\n';
    }
};


int main(int argc, char const *argv[]) {
    Test a(1);

    Test b(2);

    Test c = a + b;

    Test * d = new Test(a + c);

    c.Print();

    d->Print();

    return 0;
}
