
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include "../../libs/kompute/single_include/kompute/Kompute.hpp"
#pragma GCC diagnostic pop

#include "spdlog/spdlog.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// Forward declaring our helper function to read compiled shader
static std::vector<uint32_t> readShader(const std::string shader_path);

int main() {
    kp::Manager mgr;

    auto tensorInA = mgr.tensor({ 2.0, 4.0, 6.0 });
    auto tensorInB = mgr.tensor({ 0.0, 1.0, 2.0 });
    auto tensorOut = mgr.tensor({ 0.0, 0.0, 0.0 });

    std::vector<std::shared_ptr<kp::Tensor>> params = { tensorInA, tensorInB, tensorOut };

    std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, readShader("shader.comp.spv"));

    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpTensorSyncLocal>(params)
        ->eval();

    // prints "Output {  0  4  12  }"
    std::cout << "Output: {  ";
    for (const float& elem : tensorOut->vector()) {
        std::cout << elem << "  ";
    }
    std::cout << "}" << std::endl;
}

static std::vector<uint32_t> readShader(const std::string shader_path) {
    std::ifstream fileStream(shader_path, std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
    return { (uint32_t*)buffer.data(), (uint32_t*)(buffer.data() + buffer.size()) };
}
