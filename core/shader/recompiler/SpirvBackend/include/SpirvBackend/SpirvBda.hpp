#ifndef CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBDA_HPP
#define CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBDA_HPP

#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "SpirvBackend/SpirvEmitter.hpp"
#include "BdaAbi.hpp"
#include <array>
#include <functional>
#include <span>

namespace ShaderRecompiler {

std::uint32_t BdaConstant(SpirvEmitterState& state, std::uint64_t value);
std::uint32_t BdaWord(SpirvEmitterState& state, std::uint32_t variable, std::uint32_t index);
std::uint32_t BdaLoadWord(SpirvEmitterState& state, std::uint32_t index);
std::uint32_t BdaLoadAddress(SpirvEmitterState& state, std::uint32_t index);
void DefineBdaFaultFunction(SpirvEmitterState& state);
void DeclareBdaReadCache(SpirvEmitterState& state);
void InitializeBdaTable(SpirvEmitterState& state);
bool BdaReadCacheAvailable(const SpirvEmitterState& state);
void DeclareBdaVectorCache(SpirvEmitterState& state);
bool BdaVectorCacheAvailable(const SpirvEmitterState& state);
struct BdaCacheProbe {
    std::uint32_t hit;
    std::uint32_t device;
};
BdaCacheProbe ProbeBdaVectorCache(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes);
void RefreshBdaVectorCache(SpirvEmitterState& state, std::uint32_t address);
std::uint32_t DefineWrittenChunkNote(SpirvEmitterState& state);
void DefineBdaDwordReadFunctions(SpirvEmitterState& state);
void DefineBdaSpanReadFunctions(SpirvEmitterState& state);
void DefineBdaComponentReadFunctions(SpirvEmitterState& state);
bool BdaComponentReadsAvailable(const SpirvEmitterState& state, const IrValue& inst);
bool BdaMaskedDwordReadsAvailable(const SpirvEmitterState& state, const IrValue& inst);
std::array<std::uint32_t, 4> EmitBdaMaskedDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t used);
std::array<std::uint32_t, 4> EmitBdaComponentReads(SpirvValueEmitContext& ctx, const IrValue& inst, const std::array<std::uint32_t, 4>& addresses, const std::array<std::uint32_t, 4>& sizes);
void RecordBdaFault(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason, std::uint32_t condition = 0u);
void RecordBdaFaultReason(SpirvEmitterState& state, std::uint32_t reason, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t condition = 0u);
void FlushBdaFault(SpirvEmitterState& state);
void RecordBdaFaultWords(SpirvEmitterState& state, std::uint32_t addressLow, std::uint32_t addressHigh, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void ValidateBdaTarget(const IrProgram& program, const SpirvTargetOptions& target);
// Whether a faulting BDA access may end its invocation. Programs with workgroup barriers must keep
// every invocation running, so their faulting reads return zero instead.
bool BdaInvocationsMayStop(const IrProgram& program);
void StopBdaInvocationIf(SpirvEmitterState& state, std::uint32_t condition);
std::uint32_t BdaInstructionPc(SpirvEmitterState& state, const IrValue& inst);
std::uint32_t EmitBdaRead(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bits);
std::uint32_t EmitBdaCachedDword(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t enabled, const std::function<std::uint32_t()>& uncached, bool refill = true);
std::uint32_t EmitBdaCachedDwords(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t enabled, std::uint32_t dwords, std::uint32_t type, const std::function<std::uint32_t()>& miss);
void EmitBdaWrite(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits, std::uint32_t enabled);
void EmitBdaWriteBytes(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bytes, std::uint32_t enabled);
void EmitBdaStore(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits);
void DefineBdaByteWriteFunctions(SpirvEmitterState& state);
void DefineBdaWriteFunctions(SpirvEmitterState& state);
void EmitBdaDwordWrites(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, std::uint32_t value);
std::uint32_t EmitBdaAtomic(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bytes, const std::function<std::uint32_t(std::uint32_t)>& operation);
std::array<std::uint32_t, 4> EmitBdaDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword = false);
bool BdaByteReadsForced();
std::uint32_t AddBdaAddress(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, bool subtract);
std::uint32_t AddBdaImmediate(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::int32_t immediate);

}

#endif
