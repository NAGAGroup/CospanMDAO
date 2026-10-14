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
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = 2.0f * in.x[i] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", cs::unspecified);
        declare_output(outputs.y, "y", cs::unspecified);
    }
    void prepare_forward() override { set_shape(outputs.y, shape_of(inputs.x)[0]); }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(inputs.x.shape()));
    }
};

int main() {
    cs::Model model = cs::create_model<cs::System>();
    model.add_subsystem(cs::subsystem_desc<Axpb>("axpb"));
    model.setup();

    model.update_input("axpb.x", cs::init::from_host(std::vector<float>{1, 2, 3, 4}));
    model.run();

    auto y = model.view_output<DType::f32, 1>("axpb.y");
    y.sync();
    for (std::size_t i = 0; i < y.shape()[0]; ++i) std::cout << y[i] << ' ';
    std::cout << '\n';
}
