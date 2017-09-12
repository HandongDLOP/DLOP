#include <iostream>
#include <string>

class Activation {
private:
    int count;

public:
    Activation() {}

    virtual ~Activation() {}

    virtual void Do() {
        std::cout << "Original!" << '\n';
    }
};

class ReLu : public Activation {
private:
    /* data */

public:
    ReLu() {}

    virtual ~ReLu() {}

    void Do() {
        std::cout << "ReLu!" << '\n';
        //std::cout << this->count << '\n';

    }
};

Activation* SelectActivation(const std::string& type) {
    if (type == "ReLu") return new ReLu();

    if (type == "origin" || type == "default") return new Activation();

    else return NULL;
}

class Test {
private:
    /* data */
    Activation *m_Activation;

public:
    Test(const std::string& type = "default") {
        m_Activation = SelectActivation(type);
    }

    virtual ~Test() {}

    void Do() {
        m_Activation->Do();
    }
};


int main(int argc, char const *argv[]) {

    Test *test  = new Test("ReLu");
    Test *test2 = new Test();

    test2->Do();
    test->Do();

    std::cout << "Done!" << '\n';

    return 0;
}
