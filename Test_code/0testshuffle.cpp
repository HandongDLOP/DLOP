// random_shuffle example
#include <iostream>     // std::cout
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

// random generator function:
int myrandom (int i) { return std::rand()%i;}

int main () {
  std::srand ( unsigned ( std::time(0) ) );

  int list[60000] = {0};

  for(int i = 0; i < 60000; i++){
      list[i] = i;
  }

  std::vector<int> myvector(list, list + 60000);

  // set some values:
  // for (int i=1; i<10; ++i) myvector.push_back(i); // 1 2 3 4 5 6 7 8 9

  // using built-in random generator:
  std::random_shuffle ( myvector.begin(), myvector.end() );

  // using myrandom:
  std::random_shuffle ( myvector.begin(), myvector.end(), myrandom);

  // print out content:
  std::cout << "myvector contains:";

  for(int i = 0; i < 60000; i++){
      std::cout << ' ' << myvector[i];
  }

  // for (std::vector<int>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
  //   std::cout << ' ' << *it;

  std::cout << '\n';

  return 0;
}
