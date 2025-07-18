#pragma once

#include <atb/atb_infer.h>
#include <atb/operation.h>
#include <atb/types.h>
namespace dicp {

#define CREATE_OPERATION(param, operation)                              \
    do {                                                                \
        atb::Status atbStatus = atb::CreateOperation(param, operation); \
        if (atbStatus != atb::NO_ERROR) {                               \
            return atbStatus;                                           \
        }                                                               \
    } while (0)

#define CREATE_OPERATION_NO_RETURN(param, operation)                    \
    do {                                                                \
        atb::Status atbStatus = atb::CreateOperation(param, operation); \
        if (atbStatus != atb::NO_ERROR) {                               \
        }                                                               \
    } while (0)
}  // namespace dicp
