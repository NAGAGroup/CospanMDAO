#pragma once

// =============================================================================
// naga/core/detail/composable.ipp — Composable method definitions
// =============================================================================
//
// Included after Context is fully defined. Provides out-of-line definitions
// for template methods that call Context member functions.

#include <naga/core/composable.hpp>
#include <naga/core/context.hpp>

namespace naga::core {

template <int Dim>
inline Input<Dim> Composable::get_input(Context& ctx, std::string_view local_name) {
    return ctx.get_input<Dim>(name() + "/" + std::string(local_name));
}

template <int Dim>
inline Output<Dim> Composable::get_output(Context& ctx, std::string_view local_name) {
    return ctx.get_output<Dim>(name() + "/" + std::string(local_name));
}

template <int Dim>
inline IOInitializer<Dim> Composable::get_input_initializer(Context& ctx, const std::string& local_name) {
    std::string fqn = name().empty() ? local_name : name() + "/" + local_name;
    return ctx.get_initializer<Dim>(fqn);
}

template <int Dim>
inline IOInitializer<Dim> Composable::get_output_initializer(Context& ctx, const std::string& local_name) {
    std::string fqn = name().empty() ? local_name : name() + "/" + local_name;
    return ctx.get_initializer<Dim>(fqn);
}

} // namespace naga::core
