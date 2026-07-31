#pragma once

// =============================================================================
// naga/core/executor.hpp — Executor: drives the declare/finalize/forward/backward lifecycle
// =============================================================================
//
// Executor is the user-facing entry point for working with the V2 framework.
// Users construct an Executor with a root system and a slabkit context, then
// call setup() to declare and finalize, then forward()/backward() to execute.
//
// The Executor owns the Context and manages its lifecycle. Users never
// interact with Context directly.
//
// Usage:
//   auto executor = Executor(my_system, sk_ctx);
//   executor.setup();       // declares all I/O, finalizes context
//   executor.init();         // initializes learned params and inputs
//   // optimization iteration:
//   executor.run_with_backward();  // forward + backward (accumulates grad)
//   executor.reset();              // reset runtime state between iterations

#include "naga/core/system.hpp"
#include "naga/core/context.hpp"

#include <memory>
#include <span>
#include <string>

namespace naga::core {

class Executor {
public:
    // Factory method — creates and returns shared_ptr to Executor
    // Constructor is private; use this method instead
    static std::shared_ptr<Executor> create(std::shared_ptr<System> system,
                                            std::shared_ptr<Context> ctx);

    // Virtual destructor for polymorphic deletion
    virtual ~Executor() = default;

    // Primary execution API — run() for forward-only execution
    virtual void run();

    // Primary execution API — run_with_backward() for forward+backward execution
    virtual void run_with_backward();

    // setup — calls declare() on the root system, then finalizes the context.
    // After setup, all I/O is declared, memory is allocated, connections are
    // aliased, and the execution order is computed.
    // Virtual so subclasses can inject steps between declare and finalize.
    virtual void setup();

    // forward — execute one forward tick in topological order.
    void forward();

    // backward — execute one backward tick in reverse topological order.
    void backward();

    // init — called once after setup to initialize learned params and inputs.
    void init();

    // reset — called per iteration to reset non-learned inputs only.
    void reset();

    // Gradient lifecycle control — for optimizer use.
    void zero_non_dv_grads()    { ctx_->zero_non_dv_grads(); }
    void zero_dv_grads()        { ctx_->zero_dv_grads(); }
    void seed_objective_grad()  { ctx_->seed_objective_grad(); }

    // Serialization — delegates to Context.
    void save(const std::string& path) { ctx_->save(path); }
    void load(const std::string& path) { ctx_->load(path); }

    // Context access — for advanced usage (reading/writing I/O values,
    // zeroing gradients, etc.)
    Context& ctx() { return *ctx_; }
    const Context& ctx() const { return *ctx_; }

    // System access
    System& system() { return *system_; }
    const System& system() const { return *system_; }

protected:
    Executor(std::shared_ptr<System> system,
             std::shared_ptr<Context> ctx)
        : system_(std::move(system))
        , ctx_(std::move(ctx))
    {}

    std::shared_ptr<System> system_;
    std::shared_ptr<Context> ctx_;
};

} // namespace naga::core
