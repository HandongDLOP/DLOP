#ifndef OBJECTIVE_FUNCTION_H_
#define OBJECTIVE_FUNCTION_H_    value

#include "Ark.h"

class Objective {
private:
/* data */

public:
Objective() {}

virtual ~Objective() {}

void ComputeDeltaBar(Ark * pDesiredOutput);
};

#endif  // OBJECTIVE_FUNCTION_H_
