#include "naga/core/system.hpp"
#include "naga/core/component.hpp"

namespace naga::core {

void System::add_subsystem(Context& ctx,
                           std::string child_name,
                           std::shared_ptr<Composable> component,
                           bool register_flow) {
    subsystems_.push_back({child_name, component, register_flow});
    component->set_name(name().empty() ? child_name : name() + "/" + child_name);
    component->declare(ctx);
    if (register_flow) {
        if (Component* comp = dynamic_cast<Component*>(component.get())) {
            ctx.register_component(comp, comp->name(),
                                   comp->input_metadata(),
                                   comp->output_metadata());
        }
        // Plain Systems have no I/O metadata and were already discarded by
        // register_component's empty-I/O guard; skipping them here is equivalent.
    }
}

void System::connect(Context& ctx, std::string source_local,
                     std::string target_local) {
    ctx.connect(name().empty() ? source_local : name() + "/" + source_local,
                name().empty() ? target_local : name() + "/" + target_local);
}

void System::disconnect(Context& ctx, std::string source_local,
                        std::string target_local) {
    ctx.disconnect(name().empty() ? source_local : name() + "/" + source_local,
                   name().empty() ? target_local : name() + "/" + target_local);
}

void System::assign_objective(Context& ctx, std::string local_name) {
    ctx.assign_objective(name().empty() ? local_name : name() + "/" + local_name);
}

void System::assign_design_var(Context& ctx, std::string local_name) {
    ctx.assign_design_var(name().empty() ? local_name : name() + "/" + local_name);
}

void System::add_entry_point(Context& ctx, std::string child_name,
                             std::shared_ptr<Composable> component) {
    add_subsystem(ctx, std::move(child_name), component, /*register_flow=*/true);
    if (!ctx.component_has_io(component->name())) {
        throw std::runtime_error(
            "add_entry_point: component has no I/O metadata — entry points must be "
            "Components, not plain Systems");
    }
    ctx.mark_entry_point(component->name());
}

} // namespace naga::core
