#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "BdaAbi.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestTestMemory.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t Marker = 0xdeadbeefu;
constexpr std::uint32_t Raw = 0x31016facu;
constexpr std::uint32_t Formats = 77;
constexpr std::uint32_t Records = 0x300u;
constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t DataBytes = 0x400;
constexpr std::uint32_t ProgramWords = 256;
constexpr std::uint32_t LaneDwords = 8;
TestGuest::Array<std::uint32_t, Lanes * 8> Params;
TestGuest::Array<std::uint32_t, Formats * Lanes * LaneDwords> RuntimeOutput;
TestGuest::Array<std::uint32_t, 7 * Lanes * LaneDwords> TypedOutput;
TestGuest::Array<std::uint32_t, Formats * 4> Descriptors;
TestGuest::Array<std::uint8_t, DataBytes> Bound;

struct alignas(256) Program {
    std::array<std::uint32_t, ProgramWords> words{};
};

struct Layout {
    std::uint32_t first;
    std::uint32_t last;
};

constexpr std::array<Layout, 14> Layouts{{{1, 6}, {7, 13}, {14, 19}, {20, 22}, {23, 29}, {30, 36}, {37, 43}, {44, 49}, {50, 55}, {56, 61}, {62, 64}, {65, 71}, {72, 74}, {75, 77}}};

struct Malformed {
    const char* name;
    std::uint32_t format;
    std::uint32_t selectors;
};

constexpr std::array<Malformed, 3> MalformedDescriptors{{
    {"reserved selectors 2 and 3 with format 77", 77u, 0x69au},
    {"format 0 with selectors x y z 1", 0u, 0x3acu},
    {"format 100 with selectors x y z 1", 100u, 0x3acu},
}};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "buffer format layouts: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, false, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* Data() { return block; }

private:
    std::uint8_t* block = nullptr;
};

class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        file = std::tmpfile();
        Require(file != nullptr, "cannot open a temporary file for stderr");
#ifdef _WIN32
        saved = _dup(_fileno(stderr));
        _dup2(_fileno(file), _fileno(stderr));
#else
        saved = dup(fileno(stderr));
        dup2(fileno(file), fileno(stderr));
#endif
    }
    std::string Finish() {
        std::fflush(stderr);
#ifdef _WIN32
        _dup2(saved, _fileno(stderr));
        _close(saved);
#else
        dup2(saved, fileno(stderr));
        close(saved);
#endif
        std::rewind(file);
        std::string text;
        char buffer[4096];
        for (std::size_t read = 0; (read = std::fread(buffer, 1, sizeof(buffer), file)) != 0;) text.append(buffer, read);
        std::fclose(file);
        std::fwrite(text.data(), 1, text.size(), stderr);
        return text;
    }

private:
    std::FILE* file = nullptr;
    int saved = -1;
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes, std::uint32_t word3) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, word3};
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

std::uint32_t Hash(std::uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

std::uint32_t Components(std::uint32_t format) {
    if (format <= 13u || (format >= 20u && format <= 22u)) return 1u;
    if (format <= 19u || (format >= 23u && format <= 29u) || (format >= 62u && format <= 64u)) return 2u;
    if (format <= 43u || (format >= 72u && format <= 74u)) return 3u;
    return 4u;
}

std::uint32_t RuntimeWord3(std::uint32_t format) {
    std::uint32_t selectors = 0;
    for (std::uint32_t output = 0; output < 4u; ++output) selectors |= (output < Components(format) ? 4u + output : 0u) << (output * 3u);
    return (Raw & ~0x7ffffu) | (format << 12u) | selectors;
}

std::uint32_t Activity(std::uint32_t lane) {
    return std::array<std::uint32_t, 4>{1u, 1u, 0u, 1u}[lane % 4u];
}

std::uint32_t Offset(std::uint32_t lane) {
    constexpr std::array<std::uint32_t, 8> tail{Records - 16u, Records - 12u, Records - 8u, Records - 4u, Records, Records + 4u, 0xfffffff0u, 0x80000000u};
    return lane < 24u ? 4u + lane * 24u : tail[lane - 24u];
}

std::uint32_t Special(std::uint32_t index) {
    constexpr std::array<std::uint32_t, 12> values{0x7fc00000u, 0x7f800001u, 0xff800000u, 0x00000001u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x7bff7c01u, 0x03ff8001u, 0xfc00fe00u, 0x477fe000u, 0x00800000u};
    return values[index % values.size()];
}

std::uint32_t Mubuf(std::uint32_t opcode, std::uint32_t offset = 0u) {
    return 0xe0001000u | (opcode << 18u) | offset;
}

std::uint32_t Mtbuf(std::uint32_t opcode, std::uint32_t format) {
    return 0xe8001000u | (format << 19u) | ((opcode & 7u) << 16u);
}

std::uint32_t Operands(std::uint32_t address, std::uint32_t data, std::uint32_t resource, std::uint32_t soffset = 0x80u, bool highOpcode = false) {
    return address | (data << 8u) | (resource << 16u) | (highOpcode ? 0x200000u : 0u) | (soffset << 24u);
}

class Assembler {
public:
    explicit Assembler(Program& program) : words(program.words) { words.fill(0xbf800000u); }

    void operator()(std::uint32_t word) {
        Require(at < words.size(), "buffer format layouts: the program does not fit");
        words[at++] = word;
    }

    std::uint32_t Pc() const { return static_cast<std::uint32_t>(at * 4u); }

    void Prologue() {
        (*this)(0x34020085u);
        (*this)(0x34040085u);
        (*this)(0xe0381000u);
        (*this)(0x80020801u);
    }

    void Markers(std::uint32_t first, std::uint32_t count) {
        for (std::uint32_t reg = first; reg < first + count; ++reg) {
            (*this)(0x7e0002ffu | (reg << 17u));
            (*this)(Marker);
        }
    }

    void MaskInactive() {
        (*this)(0xbea8037eu);
        (*this)(0x363c1281u);
        (*this)(0x7d8a3c80u);
        (*this)(0x877e6a28u);
    }

    void Restore() {
        (*this)(0xbf8c3f70u);
        (*this)(0xbefe0328u);
    }

private:
    std::array<std::uint32_t, ProgramWords>& words;
    std::size_t at = 0;
};

std::uint32_t AssembleRuntime(Program& program, bool d16) {
    Assembler emit(program);
    emit.Prologue();
    emit(0x8f188410u);
    emit(0x8f198a10u);
    emit(0x871b8310u);
    emit(0x809a1b84u);
    emit(0xf4280506u);
    emit(0x30000000u);
    emit(0xbf8c3f70u);
    emit(0xbf8cc07fu);
    emit.Markers(4u, 4u);
    if (d16) emit.Markers(12u, 2u);
    emit.MaskInactive();
    const auto pc = emit.Pc();
    emit(Mubuf(0x03u));
    emit(Operands(10u, 4u, 5u, 26u));
    if (d16) {
        emit(Mubuf(0x83u));
        emit(Operands(10u, 12u, 5u, 26u));
    }
    emit.Restore();
    emit(Mubuf(0x1eu));
    emit(Operands(2u, 4u, 1u, 25u));
    if (d16) {
        emit(Mubuf(0x1du, 16u));
        emit(Operands(2u, 12u, 1u, 25u));
    }
    emit(0xbf810000u);
    return pc;
}

void AssembleTyped(Program& program, const Layout& layout) {
    Assembler emit(program);
    const auto members = layout.last - layout.first + 1u;
    emit.Prologue();
    for (std::uint32_t member = 0; member < members; ++member) emit(0xb0000000u | ((30u + member) << 16u) | (member * Lanes * LaneDwords * 4u));
    emit(0xbf8c3f70u);
    for (std::uint32_t member = 0; member < members; ++member) emit.Markers(16u + member * 8u, 6u);
    emit.MaskInactive();
    for (std::uint32_t member = 0; member < members; ++member) {
        emit(Mtbuf(3u, layout.first + member));
        emit(Operands(8u, 16u + member * 8u, 0u));
        emit(Mtbuf(11u, layout.first + member));
        emit(Operands(8u, 20u + member * 8u, 0u, 0x80u, true));
    }
    emit.Restore();
    for (std::uint32_t member = 0; member < members; ++member) {
        emit(Mubuf(0x1eu));
        emit(Operands(2u, 16u + member * 8u, 1u, 30u + member));
        emit(Mubuf(0x1du, 16u));
        emit(Operands(2u, 20u + member * 8u, 1u, 30u + member));
    }
    emit(0xbf810000u);
}

ShaderRecompiler::RecompileResult Dispatch(AgcDriver::VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, Program& program, const TestGuest::Vector<std::uint32_t>& userData, std::uint32_t groups) {
    const std::span<const std::uint32_t> code(program.words);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {Lanes, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, groups, 1, 1, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    return result;
}

TestGuest::Vector<std::uint32_t> UserData(const std::array<std::uint32_t, 4>& resource, const std::array<std::uint32_t, 4>& output) {
    TestGuest::Vector<std::uint32_t> userData;
    for (const auto& descriptor : {resource, output, BufferDescriptor(Params.data(), decltype(Params)::ByteSize, Raw), BufferDescriptor(Descriptors.data(), decltype(Descriptors)::ByteSize, Raw)}) {
        userData.insert(userData.end(), descriptor.begin(), descriptor.end());
    }
    return userData;
}

void CompareLayouts(AgcDriver::VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target) {
    const std::string pass = std::string("buffer format layouts (f16 denormals ") + (target.float16DenormPreserve ? "preserved" : "flushed") + "): ";
    RuntimeOutput.fill(0xcdcdcdcdu);
    static Program runtime;
    static_cast<void>(AssembleRuntime(runtime, true));
    static_cast<void>(Dispatch(device, target, runtime, UserData({}, BufferDescriptor(RuntimeOutput.data(), decltype(RuntimeOutput)::ByteSize, Raw)), Formats));
    static Program typed[Layouts.size()];
    for (std::uint32_t index = 0; index < Layouts.size(); ++index) {
        const auto& layout = Layouts[index];
        TypedOutput.fill(0xcdcdcdcdu);
        AssembleTyped(typed[index], layout);
        static_cast<void>(Dispatch(device, target, typed[index], UserData(BufferDescriptor(Bound.data(), Records, Raw), BufferDescriptor(TypedOutput.data(), decltype(TypedOutput)::ByteSize, Raw)), 1u));
        for (std::uint32_t format = layout.first; format <= layout.last; ++format) {
            for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
                for (std::uint32_t dword = 0; dword < 6u; ++dword) {
                    const auto actual = RuntimeOutput[((format - 1u) * Lanes + lane) * LaneDwords + dword];
                    const auto expected = TypedOutput[((format - layout.first) * Lanes + lane) * LaneDwords + dword];
                    const std::string where = pass + "format " + std::to_string(format) + (dword < 4u ? " xyzw" : " d16 xyzw") + " lane " + std::to_string(lane) + " dword " + std::to_string(dword < 4u ? dword : dword - 4u);
                    Require(actual == expected, where + " is " + Hex(actual) + " through the run-time format, " + Hex(expected) + " through the typed format");
                    if (Activity(lane) == 0u) Require(actual == Marker, where + " of an exec-off lane changed to " + Hex(actual));
                }
            }
        }
    }
}

void MalformedLoads(AgcDriver::VulkanDevice& device, const std::uint8_t* data) {
    static Program program;
    const auto pc = AssembleRuntime(program, false);
    for (std::uint32_t index = 0; index < MalformedDescriptors.size(); ++index) {
        const auto& test = MalformedDescriptors[index];
        const auto what = std::string("buffer format layouts: a run-time format load through a V# with ") + test.name;
        const auto* base = data + index;
        const auto descriptor = BufferDescriptor(base, Records, (Raw & ~0x7ffffu) | (test.format << 12u) | test.selectors);
        std::copy(descriptor.begin(), descriptor.end(), Descriptors.begin());
        RuntimeOutput.fill(0xcdcdcdcdu);
        StderrCapture log;
        try {
            static_cast<void>(Dispatch(device, device.Target(), program, UserData({}, BufferDescriptor(RuntimeOutput.data(), decltype(RuntimeOutput)::ByteSize, Raw)), 1u));
        } catch (...) {
            static_cast<void>(log.Finish());
            throw;
        }
        const auto text = log.Finish();
        std::size_t records = 0;
        std::string record;
        for (auto at = text.find("BDA access failed"); at != std::string::npos; at = text.find("BDA access failed", at + 1u)) {
            if (records++ == 0u) record = text.substr(at, text.find('\n', at) - at);
        }
        char expected[128];
        std::snprintf(expected, sizeof(expected), "BDA access failed: address=0x%llx instruction=0x%x bytes=0 stage=", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(base)), pc);
        const auto reason = record.find(" reason=");
        const auto recorded = reason != std::string::npos ? std::strtoul(record.c_str() + reason + 8u, nullptr, 10) : 0ul;
        const auto invalid = static_cast<unsigned long>(ShaderRecompiler::BdaAbi::FaultReason::InvalidDescriptor);
        Require(records == 1u && record.rfind(expected, 0u) == 0u && recorded == invalid, what + " recorded " + std::to_string(records) + " faults, expected one with reason " + std::to_string(invalid) + ", the V# base, 0 bytes and the load's pc " + Hex(pc) + ":\n" + text);
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            for (std::uint32_t dword = 0; dword < 4u; ++dword) {
                const auto actual = RuntimeOutput[lane * LaneDwords + dword];
                const auto wanted = Activity(lane) != 0u ? 0u : Marker;
                Require(actual == wanted, what + ": lane " + std::to_string(lane) + " xyzw dword " + std::to_string(dword) + " is " + Hex(actual) + ", expected " + Hex(wanted));
            }
        }
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock guest;
        auto* data = guest.Data();
        for (std::uint32_t offset = 0; offset < DataBytes; ++offset) data[offset] = static_cast<std::uint8_t>(Hash(offset * 131u) >> 7u);
        for (std::uint32_t index = 0; index < 12u; ++index) {
            const auto value = Special(index);
            std::memcpy(data + Offset(index * 2u + 1u), &value, 4u);
        }
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            Params[lane * 8u] = Offset(lane);
            Params[lane * 8u + 1u] = Activity(lane);
            Params[lane * 8u + 2u] = Offset(lane) - 4u;
        }
        for (std::uint32_t format = 1; format <= Formats; ++format) {
            const auto view = (format - 1u) % 4u;
            const auto descriptor = BufferDescriptor(data + view, Records - view, RuntimeWord3(format));
            std::copy(descriptor.begin(), descriptor.end(), Descriptors.begin() + (format - 1u) * 4u);
        }
        std::memcpy(Bound.data(), data, DataBytes);
        const auto target = device->Target();
        CompareLayouts(*device, target);
        if (target.float16DenormPreserve) {
            auto flushed = target;
            flushed.float16DenormPreserve = false;
            CompareLayouts(*device, flushed);
        }
        MalformedLoads(*device, data);
        std::puts("buffer format layouts tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
