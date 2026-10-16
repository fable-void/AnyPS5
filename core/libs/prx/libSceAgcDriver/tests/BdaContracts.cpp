#include "BdaShader.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include <array>

namespace {

template<typename TAction>
void reject(TAction action, const char* expected) {
    try { action(); }
    catch (const std::runtime_error& error) {
        AgcDriver::Graphics::Require(std::string(error.what()).find(expected) != std::string::npos, std::string("unexpected contract error: ") + error.what());
        return;
    }
    throw std::runtime_error(std::string("missing contract rejection: ") + expected);
}

}

void RunBdaContractTests() {
    using namespace ShaderRecompiler;
    const std::array<std::uint32_t, 3> capabilities{spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess};
    const std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    SpirvTargetOptions target{0x00401000u, 0x00010300u, 32, BdaAbi::Version, capabilities, extensions};
    IrProgram program;
    program.Resources().stage = IrShaderStage::Compute;
    program.Info().usesDma = true;
    ValidateBdaTarget(program, target);
    target.bdaAbiVersion = 0;
    reject([&] { ValidateBdaTarget(program, target); }, "ABI version");
    target.bdaAbiVersion = BdaAbi::Version;
    target.supportedCapabilities = {};
    reject([&] { ValidateBdaTarget(program, target); }, "capability");
    target.supportedCapabilities = capabilities;
    target.supportedExtensions = {};
    reject([&] { ValidateBdaTarget(program, target); }, "extension");
    target.supportedExtensions = extensions;
    auto& block = program.CreateBlock();
    block.AppendInstruction(&program.CreateValue(IrOpcode::Barrier, IrType::Void));
    program.BlockOrder().push_back(&block);
    ValidateBdaTarget(program, target);
    AgcDriver::Graphics::Require(!BdaInvocationsMayStop(program), "a BDA program with barriers must keep its invocations running");
    IrProgram synchronized;
    synchronized.Info().usesDma = true;
    for (const auto stage : {IrShaderStage::TessellationControl, IrShaderStage::Mesh}) {
        synchronized.Resources().stage = stage;
        ValidateBdaTarget(synchronized, target);
        AgcDriver::Graphics::Require(!BdaInvocationsMayStop(synchronized), "a tessellation-control or mesh program without guest barriers must keep its invocations running at a BDA fault: its emission synchronizes them");
    }
    RecompileRequest request{};
    request.target.bdaAbiVersion = BdaAbi::Version;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.target.fragmentShaderBarycentricEnabled = false;
    const RequestSerializer serializer;
    const auto encoded = serializer.Serialize(request);
    const auto decoded = serializer.Deserialize(encoded);
    AgcDriver::Graphics::Require(decoded.request.target.bdaAbiVersion == BdaAbi::Version && decoded.request.target.supportedCapabilities.size() == capabilities.size() && decoded.request.target.supportedExtensions[1] == extensions[1], "BDA request serialization changed target contract");
    auto invalid = encoded;
    invalid[0] = invalid[0] == 'A' ? 'B' : 'A';
    reject([&] { static_cast<void>(serializer.Deserialize(invalid)); }, "request signature");
    reject([&] { static_cast<void>(serializer.Deserialize("NVNQQWMAAAA=")); }, "serialization version");
    for (const bool stops : {true, false}) {
        for (const bool coherent : {false, true}) {
            static_cast<void>(MakeBdaSpanReadTestShader(0x7fff56780000ull, 4u, 0b0101u, coherent, stops));
            static_cast<void>(MakeBdaDwordReadTestShader(0x7fff12340001ull, 1u, coherent, stops));
            static_cast<void>(MakeBdaDwordReadTestShader(0x7fff12340001ull, 4u, coherent, stops));
        }
    }
    static_cast<void>(MakeBdaCachedReadTestShader({{0x7fff12340000ull, 1u}, {0x7fff12340004ull, 3u}}));
    static_cast<void>(MakeBdaHeldFaultTestShader(0x7fff12340041ull, 2u, 0x5678u, 0x7fff12340000ull, 8u));
    static_cast<void>(MakeBdaHeldFaultTestShader(0x7fff12340041ull, 2u, 0x5678u, ~0ull - 1u, 32u));
    if (!BdaByteReadsForced()) static_cast<void>(MakeBdaSegmentTestShader(1u));
}
