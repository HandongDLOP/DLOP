#include <iostream>
#include <typeinfo>

#define type int

template<typename T = int>
class Super {

private:
    T mtemp;

public:

    Super(T ptemp) {
        mtemp = ptemp;

        std::cout << (typeid(mtemp).name() == typeid(float).name()) << '\n';
    }

    static Super<T> * CreateSuper(T ptemp){
        return new Super<T>(ptemp);
    }

    virtual ~Super() {}
};

int main(int argc, char const *argv[]) {
    Super<type> *temp1 = new Super<type>(1);

    Super<float> *temp2 = new Super<float>(1.f);

    Super<float> *temp3 = Super<float>::CreateSuper(1.f);

    // foo * temp3 = new foo();

    delete temp1;

    delete temp2;

    return 0;
}
