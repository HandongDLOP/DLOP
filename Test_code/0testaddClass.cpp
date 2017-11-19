#include <iostream>
#include <typeinfo>

class A {
private:
    int m_add = 0;

public:
    A(int p_add) {
        m_add = p_add;
    }

    virtual ~A() {}

    void SetData(int a, int b){
        m_add = a + b;
    }

    int GetData(){
        return m_add;
    }

    virtual void Print(){
        std::cout << "A" << '\n';
    }

    A* operator+ (A *rhs){

        A * result = new A(0);

        result->SetData(this->GetData(), rhs->GetData());

        return result;
    }

};

class B : public A {
private:
    /* data */

public:
    B(int p_add) : A(p_add) {}

    virtual ~B() {}

    // B* operator+ (A *rhs){
    //     return new B(this->GetData() + rhs->GetData());
    // }

    virtual void Print(){
        std::cout << "B" << '\n';
    }

};

class C : public A{
private:
    /* data */

public:
    C (int p_add) : A(p_add){

    }
    virtual ~C (){}

    virtual void Print(){
        std::cout << "C" << '\n';
    }
};

int main(int argc, char const *argv[]) {

    A *t1 = new B(1);

    A *t2 = new C(2);

    std::cout << typeid(dynamic_cast<B*>(*t1 + t2)).name() << '\n';

    return 0;
}
