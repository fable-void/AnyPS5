#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BDASHADER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BDASHADER_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <array>
#include <cstdint>
#include <vector>

std::vector<std::uint32_t> MakeBdaTranslateTestShader(std::uint32_t count);
std::vector<std::uint32_t> MakeBdaSegmentTestShader(std::uint32_t count);
std::vector<std::uint32_t> MakeBdaTestShader(std::uint64_t address, std::uint32_t bits, std::int64_t offset = 0);
std::vector<std::uint32_t> MakeBdaHeldFaultTestShader(std::uint64_t heldAddress, std::uint32_t heldBytes, std::uint32_t heldPc, std::uint64_t address, std::uint32_t bits);
std::vector<std::uint32_t> MakeBdaDwordReadTestShader(std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops);
std::vector<std::uint32_t> MakeBdaSpanReadTestShader(std::uint64_t address, std::uint32_t offset, std::uint32_t extracted, bool coherent, bool stops);
struct BdaCachedRead {
    std::uint64_t address;
    std::uint32_t dwords;
};
std::vector<std::uint32_t> MakeBdaCachedReadTestShader(const std::vector<BdaCachedRead>& reads);
struct ImageStoreRect {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t layer;
    std::uint32_t lod;
};
std::vector<std::uint32_t> MakeImageStoreRecordTestShader(const std::array<std::uint32_t, 8>& words, const ImageStoreRect& rect, bool subgroups);
std::vector<std::uint32_t> MakeBdaDwordWriteTestShader(std::uint64_t address, std::uint32_t dwords, const std::uint32_t* values);
std::vector<std::uint32_t> MakeBdaStoreTestShader(std::uint64_t address, std::uint32_t value, std::uint32_t bits, bool coherent);
std::vector<std::uint32_t> MakeBdaWriteTestShader(std::uint64_t address, std::uint32_t value, std::uint32_t bits, bool coherent, bool enabled = true);
enum class BdaLaneStore : std::uint32_t { Byte, Short, Dword, RuntimeDword, Dwordx4, Atomic };
inline constexpr std::uint32_t BdaLaneStoreValue = 0xa1b2c3d4u;
std::vector<std::uint32_t> MakeBdaLaneStoreTestShader(std::uint64_t address, std::uint64_t stride, std::uint64_t active, BdaLaneStore store);
void RunBdaStoreExecutionTests(const AgcDriver::Graphics::Context& context);
void RunBdaExecutionTests(const AgcDriver::Graphics::Context& context);
void RunBdaContractTests();

#endif
