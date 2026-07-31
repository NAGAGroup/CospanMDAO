#include "naga/cospan/core/component.hpp"

namespace naga::cospan::core {

void Component::assign_objective(Context& ctx, std::string local_name) {
    ctx.assign_objective(name() + "/" + local_name);
}

void Component::assign_design_var(Context& ctx, std::string local_name) {
    ctx.assign_design_var(name() + "/" + local_name);
}

const std::vector<IOMetadata>& Component::input_metadata() const {
    return declared_inputs_;
}

const std::vector<IOMetadata>& Component::output_metadata() const {
    return declared_outputs_;
}

} // namespace naga::cospan::core
