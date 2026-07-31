#include "naga/cospan/core/executor.hpp"

namespace naga::cospan::core {

std::shared_ptr<Executor> Executor::create(std::shared_ptr<System> system,
                                            std::shared_ptr<Context> ctx)
{
    return std::shared_ptr<Executor>(
        new Executor(std::move(system), std::move(ctx)));
}

void Executor::setup() {
    system_->declare(*ctx_);
    ctx_->finalize();
}

void Executor::run() {
    forward();
}

void Executor::run_with_backward() {
    zero_dv_grads();
    zero_non_dv_grads();
    seed_objective_grad();
    forward();
    backward();
}

void Executor::forward() {
    ctx_->forward();
}

void Executor::backward() {
    ctx_->backward();
}

void Executor::init() {
    ctx_->init_components();
    ctx_->reset_components();
}

void Executor::reset() {
    ctx_->reset_components();
}

} // namespace naga::cospan::core
