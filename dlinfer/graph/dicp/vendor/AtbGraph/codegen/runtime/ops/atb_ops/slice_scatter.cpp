// #include <cstddef>
// #include "atb_ops.h"
// #include "utils/common.h"

// namespace dicp {

// atb::Operation* SliceScatterOperationCreate(const nlohmann::json& paramJson) {
//     std::string opName;
//     int64_t dim, start, end, step, rank;
//     atb::SVector<int64_t> starts, ends, strides;
//     atb::infer::SetValueParam param;
//     if (paramJson.contains("name")) {
//         opName = paramJson["name"].get<std::string>();
//     }
//     if (paramJson.contains("rank")) {
//         rank = paramJson["rank"].get<std::int64_t>();
//     }
//     if (paramJson.contains("dim")) {
//         dim = paramJson["dim"].get<std::int64_t>();
//     }
//     if (paramJson.contains("start")) {
//         start = paramJson["start"].get<std::int64_t>();
//         starts.resize(rank);
//         for (size_t i = 0; i < rank; ++i) {
//             starts[i] = i == dim ? start : 0;
//         }
//     }
//     if (paramJson.contains("end")) {
//         end = paramJson["end"].get<std::int64_t>();
//         ends.resize(rank);
//         for (size_t i = 0; i < rank; ++i) {
//             ends[i] = i == dim ? end : -1;
//             // ends[i] = 8;
//         }
//     }
//     if (paramJson.contains("step")) {
//         step = paramJson["step"].get<std::int64_t>();
//         strides.resize(rank);
//         for (size_t i = 0; i < rank; ++i) {
//             strides[i] = i == dim ? step : 1;
//         }
//     }
//     DICP_LOG(INFO) << "SliceScatterOperation name: " << opName << ", starts:" << svectorToString(starts) << ", ends:" << svectorToString(ends) << ",
//     strides:" << svectorToString(strides); atb::Operation* op = nullptr; CREATE_OPERATION_NO_RETURN(param, &op); return op;
// }

// REGISTER_ATB_OPERATION("SliceScatterOperation", SliceScatterOperationCreate);

// }  // namespace dicp
