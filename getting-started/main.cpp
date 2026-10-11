#include <iostream>
#include <vector>

#include <naga/cospan/cospan.hpp>

namespace cs = naga::cospan;
using cs::DType;

// The component: y = a * x + b (here a = 2, b = 1), one work-item per element.
class Axpb : public cs::Component<Axpb> {
public:
    struct Inputs  { cs::Input<DType::f32, 1> x; };
    struct Outputs { cs::Output<DType::f32, 1> y; };
    static void compute_impl(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = 2.0f * in.x[i] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    void setup() override {
        declare_input(inputs.x, "x", cs::unspecified);
        declare_output(outputs.y, "y", cs::unspecified);
    }
    void pre_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }
    std::unique_ptr<cs::LaunchDescription> compute_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()));
    }
};

// The model's root: a system holding the one component.
class App : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<Axpb>("axpb")); }
};

int main() {
    cs::Model model = cs::create_model<App>();
    model.setup();

    model.update_input("axpb.x", cs::init::from_host(std::vector<float>{1, 2, 3, 4}));
    model.run();

    auto y = model.view_output<DType::f32, 1>("axpb.y");
    y.sync();
    for (std::size_t i = 0; i < y.shape()[0]; ++i) std::cout << y[i] << ' ';
    std::cout << '\n';
}
