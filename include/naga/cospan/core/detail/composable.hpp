#pragma once

// =============================================================================
// naga/cospan/core/detail/composable.hpp — Composable declaration (V2 Framework)
// =============================================================================
//
// Declaration-only header. Method definitions that require Context's full
// definition live in detail/composable.ipp.
//
// Composable provides: declare() (pure virtual), the name protocol
// (set_name/name), I/O-metadata virtuals (input_metadata/output_metadata),
// and I/O-access helper template declarations. The execution lifecycle
// (forward/backward/init/reset) lives on Component, not Composable — Systems
// derive from Composable and are not executable.

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <sycl/sycl.hpp>

#include <naga/cospan/core/io.hpp>

namespace naga::cospan::core {

// Forward declaration — Context is defined in context.hpp
class Context;

struct IOMetadata {
    std::string name;
    size_t      size;
};

class Composable {
public:
    virtual ~Composable() = default;

    virtual void declare(Context& ctx) = 0;

    virtual const std::vector<IOMetadata>& input_metadata() const {
        static const std::vector<IOMetadata> empty;
        return empty;
    }
    virtual const std::vector<IOMetadata>& output_metadata() const {
        static const std::vector<IOMetadata> empty;
        return empty;
    }

    void set_name(std::string name) { name_ = std::move(name); }
    const std::string& name() const { return name_; }

    // I/O access helpers — declarations only; definitions in composable.ipp
    template <int Dim>
    Input<Dim> get_input(Context& ctx, std::string_view local_name);

    template <int Dim>
    Output<Dim> get_output(Context& ctx, std::string_view local_name);

    template <int Dim>
    IOInitializer<Dim> get_input_initializer(Context& ctx, const std::string& local_name);

    template <int Dim>
    IOInitializer<Dim> get_output_initializer(Context& ctx, const std::string& local_name);

protected:
    std::string name_;
};

} // namespace naga::cospan::core
