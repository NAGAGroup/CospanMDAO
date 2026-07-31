add_executable(sellar sellar/sellar.cpp)
target_link_libraries(sellar PRIVATE naga::cospan)
add_sycl_to_target(TARGET sellar SOURCES sellar/sellar.cpp)
