#include "cases.hpp"
int main() {
    sycl::queue q{sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    return spike::run_all(q);
}
