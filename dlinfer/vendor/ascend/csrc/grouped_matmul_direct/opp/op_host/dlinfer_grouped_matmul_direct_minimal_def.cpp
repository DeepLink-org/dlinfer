#include "register/op_def_registry.h"

namespace ops {
class DlinferGroupedMatmulDirect : public OpDef {
public:
    explicit DlinferGroupedMatmulDirect(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> floatTypes = {
            ge::DT_FLOAT16, ge::DT_BF16};
        const std::initializer_list<ge::Format> ndFormats = {
            ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("x").ParamType(DYNAMIC).DataType(floatTypes).Format(ndFormats);
        this->Input("weight").ParamType(DYNAMIC).DataType(floatTypes).Format(ndFormats);
        this->Input("bias").ParamType(DYNAMIC).DataType(floatTypes).Format(ndFormats);
        this->Input("scale").ParamType(DYNAMIC)
            .DataType({ge::DT_UINT64, ge::DT_UINT64}).Format(ndFormats);
        this->Input("offset").ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format(ndFormats);
        this->Input("antiquant_scale").ParamType(DYNAMIC).DataType(floatTypes).Format(ndFormats);
        this->Input("antiquant_offset").ParamType(DYNAMIC).DataType(floatTypes).Format(ndFormats);
        this->Input("group_list").ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64}).Format(ndFormats);
        this->Input("per_token_scale").ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format(ndFormats);
        this->Output("y").ParamType(DYNAMIC).DataType(floatTypes).Format(ndFormats);

        this->Attr("split_item").AttrType(OPTIONAL).Int(0);
        this->Attr("dtype").AttrType(OPTIONAL).Int(0);
        this->Attr("transpose_weight").AttrType(OPTIONAL).Bool(false);
        this->Attr("transpose_x").AttrType(OPTIONAL).Bool(false);
        this->Attr("group_type").AttrType(OPTIONAL).Int(-1);
        this->Attr("group_list_type").AttrType(OPTIONAL).Int(0);
        this->Attr("act_type").AttrType(OPTIONAL).Int(0);
        this->Attr("tuning_config").AttrType(OPTIONAL).ListInt({0});

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910_93", config);
    }
};

OP_ADD(DlinferGroupedMatmulDirect);
}  // namespace ops
