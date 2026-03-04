#ifndef INTERFACEDEFINES_H
#define INTERFACEDEFINES_H
#include "cuda_runtime.h"
// #define LIBPOISSONVERSION "0.0" not a good practice, let CMAKE handle this
#ifndef DOUBLE_PRECISION
#define SINGLE_PRECISION
#endif
namespace libpoisson {
#if defined SINGLE_PRECISION
using real = float;
#else
using real = double;
#endif

} // namespace libpoisson
#endif
