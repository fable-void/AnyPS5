#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "SpirvBackend/SpirvMemory/SpirvSubgroup.hpp"
#include <spirv/unified1/GLSL.std.450.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

namespace {

void EmitBdaOverflowCheck(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction) {
    const auto overflow = Binary(state, spv::OpUGreaterThan, TypeBool(state), address, BdaConstant(state, std::numeric_limits<std::uint64_t>::max() - bytes));
    RecordBdaFault(state, address, ConstantU32(state, bytes), instruction, BdaAbi::FaultReason::Overflow, overflow);
    StopBdaInvocationIf(state, overflow);
}

bool BdaCoherent(const SpirvEmitterState& state, const IrValue& inst) {
    const auto index = inst.Flags<MemoryFlags>().index;
    const auto& memory = state.program.Resources().memoryInfo;
    if (index >= memory.size()) return false;
    if (memory[index].coherent) return true;
    const auto access = BufferAccessOf(inst.Opcode());
    if (access == BufferAccess::None) return false;
    const bool store = access == BufferAccess::Write || access == BufferAccess::Atomic;
    return state.requirements.coherentBuffers && (store || !state.requirements.coherentBufferAlias);
}

std::uint32_t BdaAccessMask(bool coherent) {
    return spv::MemoryAccessAlignedMask | (coherent ? spv::MemoryAccessVolatileMask : 0u);
}

std::uint32_t BdaAccessMask(const SpirvEmitterState& state, const IrValue& inst) {
    return BdaAccessMask(BdaCoherent(state, inst));
}

std::uint32_t LoadBdaDevice(SpirvEmitterState& state, const IrValue& inst, std::uint32_t device) {
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, device);
    const auto value = state.module.AllocateId();
    if (BdaCoherent(state, inst)) state.module.AddFunction(spv::OpAtomicLoad, TypeU32(state), value, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u));
    else state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer, BdaAccessMask(false), 4u);
    return value;
}

std::uint32_t EmitBdaByte(SpirvEmitterState& state, std::uint32_t guest, std::uint32_t instruction, std::uint32_t accessMask) {
    const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
    const auto bytePointer = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType);
    const auto physical = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaPointerFunction, guest, ConstantU32(state, 1u), instruction);
    if (state.bdaStopsInvocations) {
        StopBdaInvocationIf(state, Binary(state, spv::OpIEqual, TypeBool(state), physical, BdaConstant(state, 0u)));
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, bytePointer, pointer, physical);
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, accessMask, 1u);
        return Unary(state, spv::OpUConvert, TypeU32(state), loaded);
    }
    // Unmapped bytes read as zero; the fault is recorded and every invocation reaches the
    // program's barriers.
    const auto mapped = Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u));
    const auto before = state.currentLabel;
    const auto loadLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, mapped, loadLabel, merge);
    EmitLabel(state, loadLabel);
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, bytePointer, pointer, physical);
    const auto loaded = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, accessMask, 1u);
    const auto widened = Unary(state, spv::OpUConvert, TypeU32(state), loaded);
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, TypeU32(state), value, widened, loadLabel, ConstantU32(state, 0u), before);
    return value;
}

std::uint32_t EmitBdaByteLoop(SpirvEmitterState& state, std::uint32_t address, std::uint32_t count, std::uint32_t instruction, std::uint32_t accessMask) {
    const auto u32 = TypeU32(state);
    const auto entry = state.currentLabel;
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    const auto byte = state.module.AllocateId();
    const auto result = state.module.AllocateId();
    const auto nextByte = state.module.AllocateId();
    const auto nextResult = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    state.module.AddFunction(spv::OpPhi, u32, byte, ConstantU32(state, 0u), entry, nextByte, continuation);
    state.module.AddFunction(spv::OpPhi, u32, result, ConstantU32(state, 0u), entry, nextResult, continuation);
    const auto more = Binary(state, spv::OpULessThan, TypeBool(state), byte, count);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, more, body, merge);
    EmitLabel(state, body);
    const auto guest = Binary(state, spv::OpIAdd, TypeScalarU64(state), address, Unary(state, spv::OpUConvert, TypeScalarU64(state), byte));
    const auto value = EmitBdaByte(state, guest, instruction, accessMask);
    state.module.AddFunction(spv::OpBitwiseOr, u32, nextResult, result, Binary(state, spv::OpShiftLeftLogical, u32, value, Binary(state, spv::OpShiftLeftLogical, u32, byte, ConstantU32(state, 3u))));
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpIAdd, u32, nextByte, byte, ConstantU32(state, 1u));
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    return result;
}

std::uint32_t EmitBdaBytes(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t accessMask) {
    if (bytes == 1u) return EmitBdaByte(state, address, instruction, accessMask);
    return EmitBdaByteLoop(state, address, ConstantU32(state, bytes), instruction, accessMask);
}

// The dwords of a wide load that the program extracts (all of them when the composite is used
// whole). Dead-code elimination has dropped the unused extracts, as it dropped unused dword loads
// when every dword was its own instruction.
std::uint32_t UsedBdaDwords(const IrValue& inst, std::uint32_t dwords) {
    const auto all = (1u << dwords) - 1u;
    if (dwords == 1u) return all;
    std::uint32_t used = 0;
    for (const IrValue* user : inst.Uses()) {
        const auto opcode = user->Opcode();
        if (opcode != IrOpcode::CompositeExtractU32x2 && opcode != IrOpcode::CompositeExtractU32x3 && opcode != IrOpcode::CompositeExtractU32x4) return all;
        const IrValue* index = user->ArgumentCount() > 1u ? user->Argument(1) : nullptr;
        if (index == nullptr || !index->HasImmediate() || index->ImmediateU32() >= dwords) return all;
        used |= 1u << index->ImmediateU32();
    }
    return used;
}

std::uint32_t BdaAddressUnaligned(SpirvEmitterState& state, std::uint32_t address) {
    return Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), address, BdaConstant(state, 3u)), BdaConstant(state, 0u));
}

std::uint32_t BdaSpanReadResultType(SpirvEmitterState& state) {
    const auto values = TypeU32Vector(state, 4);
    return state.bdaStopsInvocations ? state.module.Type(spv::OpTypeStruct, values, TypeU32(state)) : values;
}

std::uint32_t DefineBdaSpanReadFunction(SpirvEmitterState& state, bool coherent) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto values = TypeU32Vector(state, 4);
    const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
    const auto zero = ConstantU32(state, 0u);
    const auto one = ConstantU32(state, 1u);
    const bool stops = state.bdaStopsInvocations;
    const auto result = BdaSpanReadResultType(state);
    const auto& cache = state.bdaVectorCache;
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalAnd, boolean, left, right); };
    const auto either = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpLogicalOr, boolean, left, right); };
    const auto widen = [&](std::uint32_t value) { return Unary(state, spv::OpUConvert, u64, value); };
    const auto load = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u64, value, variable);
        return value;
    };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, std::string("read_bda_span") + (stops ? "_stop" : "") + (coherent ? "_coherent" : ""));
    state.module.AddFunction(spv::OpFunction, result, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, result, u64, u32, u32, u32));
    const auto address = state.module.AllocateId();
    const auto offset = state.module.AllocateId();
    const auto used = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, offset);
    state.module.AddFunction(spv::OpFunctionParameter, u32, used);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto firstBytes = binary(spv::OpShiftLeftLogical, u32, EmitGlsl<GLSLstd450FindILsb, IrType::U32>(state, used), constant(2u));
    const auto end = binary(spv::OpShiftLeftLogical, u32, binary(spv::OpIAdd, u32, EmitGlsl<GLSLstd450FindUMsb, IrType::U32>(state, used), one), constant(2u));
    const auto span = binary(spv::OpISub, u32, end, firstBytes);
    const auto zeros = state.module.Constant(spv::OpConstantComposite, values, zero, zero, zero, zero);
    const auto entry = state.currentLabel;
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    const auto position = state.module.AllocateId();
    const auto piece = state.module.AllocateId();
    const auto accumulated = state.module.AllocateId();
    const auto nextPosition = state.module.AllocateId();
    const auto nextPiece = coherent ? state.module.AllocateId() : one;
    const auto nextGathered = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    state.module.AddFunction(spv::OpPhi, u32, position, firstBytes, entry, nextPosition, continuation);
    state.module.AddFunction(spv::OpPhi, u32, piece, span, entry, nextPiece, continuation);
    state.module.AddFunction(spv::OpPhi, values, accumulated, zeros, entry, nextGathered, continuation);
    const auto pending = binary(spv::OpULessThan, boolean, position, end);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, pending, body, merge);
    EmitLabel(state, body);
    const auto dword = binary(spv::OpShiftRightLogical, u32, position, constant(2u));
    const auto relative = binary(spv::OpIAdd, u32, offset, binary(spv::OpShiftLeftLogical, u32, dword, constant(2u)));
    const auto negative = binary(spv::OpSLessThan, boolean, relative, zero);
    const auto magnitude = widen(Select(state, u32, negative, binary(spv::OpISub, u32, zero, relative), relative));
    const auto sum = binary(spv::OpIAdd, u64, address, magnitude);
    const auto dwordAddress = Select(state, u64, negative, binary(spv::OpISub, u64, address, magnitude), sum);
    const auto early = Select(state, boolean, negative, binary(spv::OpUGreaterThan, boolean, magnitude, address), binary(spv::OpULessThan, boolean, sum, address));
    const auto late = binary(spv::OpUGreaterThan, boolean, dwordAddress, BdaConstant(state, std::numeric_limits<std::uint64_t>::max() - 4u));
    const auto within = binary(spv::OpBitwiseAnd, u32, position, constant(3u));
    const auto target = binary(spv::OpIAdd, u64, dwordAddress, widen(within));
    const auto targetEnd = binary(spv::OpIAdd, u64, target, widen(piece));
    const auto inside = [&] { return both(binary(spv::OpULessThan, boolean, target, targetEnd), both(binary(spv::OpULessThanEqual, boolean, load(cache[0]), target), binary(spv::OpULessThanEqual, boolean, targetEnd, load(cache[1])))); };
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, inside()), [&] { RefreshBdaVectorCache(state, target); });
    const auto mapped = inside();
    const auto delta = load(cache[2]);
    const auto device = binary(spv::OpIAdd, u64, target, delta);
    const auto single = binary(spv::OpIEqual, boolean, piece, one);
    const auto served = both(mapped, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u64, device, BdaConstant(state, 3u)), BdaConstant(state, 0u)));
    const auto mask = coherent ? Select(state, u32, binary(spv::OpUGreaterThan, boolean, piece, constant(4u)), used, binary(spv::OpShiftLeftLogical, u32, one, dword)) : used;
    const auto spanned = EmitValueOrDefaultIfCondition(state, both(Unary(state, spv::OpLogicalNot, boolean, single), served), values, accumulated, [&] {
        std::array<std::uint32_t, 4> parts{};
        for (std::uint32_t index = 0; index < 4u; ++index) {
            const auto wanted = binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, mask, constant(1u << index)), zero);
            parts[index] = EmitValueOrZeroIfCondition(state, wanted, [&] {
                const auto pointer = state.module.AllocateId();
                state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, binary(spv::OpIAdd, u64, device, widen(binary(spv::OpISub, u32, constant(index * 4u), position))));
                const auto value = state.module.AllocateId();
                if (coherent) state.module.AddFunction(spv::OpAtomicLoad, u32, value, pointer, constant(spv::ScopeDevice), zero);
                else state.module.AddFunction(spv::OpLoad, u32, value, pointer, BdaAccessMask(false), 4u);
                return value;
            });
        }
        const auto composite = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, values, composite, parts[0], parts[1], parts[2], parts[3]);
        return coherent ? binary(spv::OpBitwiseOr, values, accumulated, composite) : composite;
    });
    const auto byteMapped = both(single, mapped);
    const auto before = state.currentLabel;
    const auto byteLabel = state.module.AllocateId();
    const auto byteMerge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, byteMerge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, byteMapped, byteLabel, byteMerge);
    EmitLabel(state, byteLabel);
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType), pointer, device);
    const auto loaded = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, BdaAccessMask(coherent), 1u);
    const auto part = state.module.AllocateId();
    state.module.AddFunction(spv::OpVectorExtractDynamic, u32, part, spanned, dword);
    const auto inserted = state.module.AllocateId();
    state.module.AddFunction(spv::OpVectorInsertDynamic, values, inserted, spanned, binary(spv::OpBitwiseOr, u32, part, binary(spv::OpShiftLeftLogical, u32, Unary(state, spv::OpUConvert, u32, loaded), binary(spv::OpShiftLeftLogical, u32, within, constant(3u)))), dword);
    state.module.AddFunction(spv::OpBranch, byteMerge);
    EmitLabel(state, byteMerge);
    state.module.AddFunction(spv::OpPhi, values, nextGathered, inserted, byteLabel, spanned, before);
    const auto overflow = either(early, late);
    const auto fires = either(overflow, both(single, Unary(state, spv::OpLogicalNot, boolean, mapped)));
    EmitIfCondition(state, fires, [&] {
        const auto reason = Select(state, u32, overflow, constant(static_cast<std::uint32_t>(BdaAbi::FaultReason::Overflow)), Unary(state, spv::OpUConvert, u32, delta));
        const auto at = Select(state, u64, early, address, Select(state, u64, late, dwordAddress, target));
        const auto bytes = Select(state, u32, early, zero, Select(state, u32, late, constant(4u), one));
        RecordBdaFaultReason(state, reason, at, bytes, instruction);
    });
    StopBdaInvocationIf(state, fires);
    const auto next = binary(spv::OpIAdd, u32, position, piece);
    const auto rest = binary(spv::OpShiftRightLogical, u32, used, binary(spv::OpShiftRightLogical, u32, next, constant(2u)));
    const auto skipped = Select(state, u32, binary(spv::OpIEqual, boolean, rest, zero), end, binary(spv::OpIAdd, u32, next, binary(spv::OpShiftLeftLogical, u32, EmitGlsl<GLSLstd450FindILsb, IrType::U32>(state, rest), constant(2u))));
    const auto advanced = either(single, served);
    state.module.AddFunction(spv::OpSelect, u32, nextPosition, advanced, skipped, position);
    if (coherent) state.module.AddFunction(spv::OpSelect, u32, nextPiece, advanced, Select(state, u32, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, skipped, constant(3u)), zero), constant(4u), one), Select(state, u32, binary(spv::OpUGreaterThan, boolean, piece, constant(4u)), constant(4u), one));
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    auto returned = accumulated;
    if (stops) {
        returned = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, result, returned, accumulated, zero);
    }
    state.module.AddFunction(spv::OpReturnValue, returned);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

void DefineBdaDwordReadFunctions(SpirvEmitterState& state) {
    if (BdaVectorCacheAvailable(state)) return;
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto pair = TypeU32Vector(state, 2);
    const auto type = state.module.Type(spv::OpTypeFunction, pair, u64, u32);
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    const bool stops = state.bdaStopsInvocations;
    if (stops) state.bdaStopValue = state.module.Constant(spv::OpConstantComposite, pair, ConstantU32(state, 0u), ConstantU32(state, 1u));
    for (const bool stop : {false, true}) {
        for (const bool coherent : {false, true}) {
            if ((stop && !stops) || (coherent && !coherentAccesses)) continue;
            const auto function = state.module.AllocateId();
            state.module.AddName(function, std::string("read_bda_dword_bytes") + (stop ? "_stop" : "") + (coherent ? "_coherent" : ""));
            state.module.AddFunction(spv::OpFunction, pair, function, spv::FunctionControlDontInlineMask, type);
            const auto address = state.module.AllocateId();
            const auto instruction = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionParameter, u64, address);
            state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
            EmitLabel(state, state.module.AllocateId());
            state.bdaStopsInvocations = stop;
            EmitBdaOverflowCheck(state, address, 4u, instruction);
            const auto value = EmitBdaBytes(state, address, 4u, instruction, BdaAccessMask(coherent));
            const auto result = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, pair, result, value, ConstantU32(state, 0u));
            state.module.AddFunction(spv::OpReturnValue, result);
            state.module.AddFunction(spv::OpFunctionEnd);
            state.bdaDwordReadFunctions[stop][coherent] = function;
        }
    }
    state.bdaStopsInvocations = stops;
    state.bdaStopValue = 0;
}

void DefineBdaSpanReadFunctions(SpirvEmitterState& state) {
    if (!BdaVectorCacheAvailable(state)) return;
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    const bool stops = state.bdaStopsInvocations;
    const auto zero = ConstantU32(state, 0u);
    for (const bool stop : {false, true}) {
        if (stop && !stops) continue;
        state.bdaStopsInvocations = stop;
        state.bdaStopValue = stop ? state.module.Constant(spv::OpConstantComposite, BdaSpanReadResultType(state), state.module.Constant(spv::OpConstantComposite, TypeU32Vector(state, 4), zero, zero, zero, zero), ConstantU32(state, 1u)) : 0u;
        for (const bool coherent : {false, true}) {
            if (coherent && !coherentAccesses) continue;
            state.bdaSpanReadFunctions[stop][coherent] = DefineBdaSpanReadFunction(state, coherent);
        }
    }
    state.bdaStopsInvocations = stops;
    state.bdaStopValue = 0;
}

// APS5_BDA_BYTE_READS=1 restores one lookup and one byte load per byte for every read.
bool BdaByteReadsForced() {
    static const bool forced = std::getenv("APS5_BDA_BYTE_READS") != nullptr;
    return forced;
}

std::uint32_t BdaInstructionPc(SpirvEmitterState& state, const IrValue& inst) {
    return state.bdaPcOverride != 0u ? state.bdaPcOverride : ConstantU32(state, inst.Flags<MemoryFlags>().pc);
}

std::uint32_t AddBdaAddress(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, bool subtract) {
    auto& state = ctx.state;
    const auto result = Binary(state, subtract ? spv::OpISub : spv::OpIAdd, TypeScalarU64(state), address, offset);
    const auto overflow = subtract ? Binary(state, spv::OpUGreaterThan, TypeBool(state), offset, address) : Binary(state, spv::OpULessThan, TypeBool(state), result, address);
    RecordBdaFault(state, address, ConstantU32(state, 0u), BdaInstructionPc(state, inst), BdaAbi::FaultReason::Overflow, overflow);
    StopBdaInvocationIf(state, overflow);
    return result;
}

std::uint32_t AddBdaImmediate(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::int32_t immediate) {
    if (immediate == 0) {
        return address;
    }
    const auto magnitude = immediate < 0 ? -static_cast<std::int64_t>(immediate) : static_cast<std::int64_t>(immediate);
    return AddBdaAddress(ctx, inst, address, BdaConstant(ctx.state, static_cast<std::uint64_t>(magnitude)), immediate < 0);
}

void ValidateBdaTarget(const IrProgram& program, const SpirvTargetOptions& target) {
    if (program.Info().usesFaultBuffer && target.bdaAbiVersion != BdaAbi::Version) throw std::runtime_error("the fault buffer needs the BDA fault ABI, which the target lacks");
    if (!program.Info().usesDma) return;
    if (target.bdaAbiVersion != BdaAbi::Version) throw std::runtime_error("unsupported BDA ABI version");
    for (const auto capability : {spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess}) {
        if (std::find(target.supportedCapabilities.begin(), target.supportedCapabilities.end(), static_cast<std::uint32_t>(capability)) == target.supportedCapabilities.end()) throw std::runtime_error("BDA requires unsupported SPIR-V capability " + std::to_string(capability));
    }
    for (const auto extension : {"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"}) {
        if (std::find(target.supportedExtensions.begin(), target.supportedExtensions.end(), extension) == target.supportedExtensions.end()) throw std::runtime_error(std::string("BDA requires unsupported extension ") + extension);
    }
}

bool BdaInvocationsMayStop(const IrProgram& program) {
    const auto stage = program.Resources().stage;
    if (stage == IrShaderStage::Mesh || stage == IrShaderStage::TessellationControl) return false;
    for (const auto* block : program.BlockOrder()) {
        for (const auto* instruction : block->Instructions()) {
            if (instruction->Opcode() == IrOpcode::Barrier) return false;
        }
    }
    return true;
}

std::uint32_t EmitBdaRead(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bits) {
    auto& state = ctx.state;
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA read width");
    if (state.bdaPointerFunction == 0) ctx.Fail(inst, "BDA lookup function is missing");
    if (bits == 32u) return EmitBdaDwordReads(ctx, inst, address, 0u, 1u)[0];
    const auto instruction = BdaInstructionPc(state, inst);
    EmitBdaOverflowCheck(state, address, bits / 8u, instruction);
    return EmitBdaBytes(state, address, bits / 8u, instruction, BdaAccessMask(state, inst));
}

namespace {

bool BdaSegmentsShared(SpirvEmitterState& state) {
    const auto& capabilities = state.supportedCapabilities;
    const bool ballot = std::find(capabilities.begin(), capabilities.end(), static_cast<std::uint32_t>(spv::CapabilityGroupNonUniformBallot)) != capabilities.end();
    if (HostSubgroupSingleLane(state) || !ballot) return false;
    state.module.EmitCapability(spv::CapabilityGroupNonUniform);
    state.module.EmitCapability(spv::CapabilityGroupNonUniformBallot);
    return true;
}

std::uint32_t BroadcastFirstU64(SpirvEmitterState& state, std::uint32_t value) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto first = [&](std::uint32_t word) {
        const auto result = state.module.AllocateId();
        state.module.AddFunction(spv::OpGroupNonUniformBroadcastFirst, u32, result, ConstantU32(state, spv::ScopeSubgroup), word);
        return result;
    };
    const auto low = first(Unary(state, spv::OpUConvert, u32, value));
    const auto high = first(Unary(state, spv::OpUConvert, u32, Binary(state, spv::OpShiftRightLogical, u64, value, BdaConstant(state, 32u))));
    return Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, low), Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, high), BdaConstant(state, 32u)));
}

}

namespace {

struct BdaReadRefill {
    std::array<std::uint32_t, 3> fields{};
    std::uint32_t covers = 0;
};

BdaReadRefill RefillBdaReadCache(SpirvValueEmitContext& ctx, std::uint32_t address, std::uint32_t end, std::uint32_t enabled) {
    auto& state = ctx.state;
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto& cache = state.bdaReadCache;
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpLogicalAnd, boolean, left, right); };
    const auto segmentType = state.module.Type(spv::OpTypeStruct, u64, u64, u64);
    const auto translate = [&] {
        const auto segment = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, segmentType, segment, state.bdaSegmentFunction, address);
        return segment;
    };
    const bool shared = BdaSegmentsShared(state);
    std::uint32_t segment = 0;
    if (shared) {
        const auto elected = state.module.AllocateId();
        EmitSubgroupElect(state, elected);
        const auto zero = BdaConstant(state, 0u);
        segment = EmitValueOrDefaultIfCondition(state, elected, segmentType, state.module.Constant(spv::OpConstantComposite, segmentType, zero, zero, zero), translate);
    } else {
        segment = translate();
    }
    BdaReadRefill refill;
    for (std::uint32_t field = 0; field < 3u; ++field) {
        refill.fields[field] = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u64, refill.fields[field], segment, field);
        if (shared) refill.fields[field] = BroadcastFirstU64(state, refill.fields[field]);
    }
    const auto aligned = Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u64, Binary(state, spv::OpIAdd, u64, address, refill.fields[2]), BdaConstant(state, 3u)), BdaConstant(state, 0u));
    const auto within = both(Binary(state, spv::OpULessThanEqual, boolean, refill.fields[0], address), Binary(state, spv::OpULessThanEqual, boolean, end, refill.fields[1]));
    refill.covers = both(both(enabled, both(Binary(state, spv::OpULessThan, boolean, address, end), aligned)), within);
    for (std::uint32_t field = 0; field < 3u; ++field) {
        const auto previous = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u64, previous, cache[field]);
        state.module.AddFunction(spv::OpStore, cache[field], Select(state, u64, refill.covers, refill.fields[field], previous));
    }
    return refill;
}

}

std::uint32_t EmitBdaCachedDword(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t enabled, const std::function<std::uint32_t()>& uncached, bool refill) {
    auto& state = ctx.state;
    if (!BdaReadCacheAvailable(state)) ctx.Fail(inst, "BDA read cache is missing");
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto& cache = state.bdaReadCache;
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpLogicalAnd, boolean, left, right); };
    const auto load = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u64, value, variable);
        return value;
    };
    const auto end = Binary(state, spv::OpIAdd, u64, address, BdaConstant(state, 4u));
    const auto within = [&](std::uint32_t begin, std::uint32_t limit) { return both(Binary(state, spv::OpULessThanEqual, boolean, begin, address), Binary(state, spv::OpULessThanEqual, boolean, end, limit)); };
    const auto fast = [&] {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, Binary(state, spv::OpIAdd, u64, address, load(cache[2])));
        const auto value = state.module.AllocateId();
        if (BdaCoherent(state, inst)) state.module.AddFunction(spv::OpAtomicLoad, u32, value, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u));
        else state.module.AddFunction(spv::OpLoad, u32, value, pointer, BdaAccessMask(false), 4u);
        return value;
    };
    const auto hit = both(enabled, within(load(cache[0]), load(cache[1])));
    if (!refill) return EmitValueIfElse(state, hit, u32, fast, uncached);
    return EmitValueIfElse(state, hit, u32, fast, [&] {
        const auto refilled = RefillBdaReadCache(ctx, address, end, enabled);
        return EmitValueIfElse(state, refilled.covers, u32, fast, uncached);
    });
}

std::uint32_t EmitBdaCachedDwords(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t enabled, std::uint32_t dwords, std::uint32_t type, const std::function<std::uint32_t()>& miss) {
    auto& state = ctx.state;
    if (!BdaReadCacheAvailable(state)) ctx.Fail(inst, "BDA read cache is missing");
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto& cache = state.bdaReadCache;
    const auto both = [&](std::uint32_t left, std::uint32_t right) { return Binary(state, spv::OpLogicalAnd, boolean, left, right); };
    const auto load = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u64, value, variable);
        return value;
    };
    const auto end = Binary(state, spv::OpIAdd, u64, address, BdaConstant(state, dwords * 4u));
    const auto within = both(Binary(state, spv::OpULessThanEqual, boolean, load(cache[0]), address), Binary(state, spv::OpULessThanEqual, boolean, end, load(cache[1])));
    const auto fast = [&](std::uint32_t delta) {
        const auto device = Binary(state, spv::OpIAdd, u64, address, delta);
        std::vector<std::uint32_t> construct{spv::OpCompositeConstruct, type, state.module.AllocateId()};
        const bool coherent = BdaCoherent(state, inst);
        for (std::uint32_t dword = 0; dword < dwords;) {
            const auto width = coherent ? 1u : std::min(4u, dwords - dword);
            const auto element = dword == 0u ? device : Binary(state, spv::OpIAdd, u64, device, BdaConstant(state, dword * 4u));
            const auto pointer = state.module.AllocateId();
            const auto loaded = state.module.AllocateId();
            if (width == 1u) {
                state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, element);
                if (coherent) state.module.AddFunction(spv::OpAtomicLoad, u32, loaded, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u));
                else state.module.AddFunction(spv::OpLoad, u32, loaded, pointer, BdaAccessMask(false), 4u);
                construct.push_back(loaded);
            } else {
                const auto vector = TypeU32Vector(state, width);
                state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, vector), pointer, element);
                state.module.AddFunction(spv::OpLoad, vector, loaded, pointer, BdaAccessMask(false), 4u);
                for (std::uint32_t component = 0; component < width; ++component) {
                    const auto value = state.module.AllocateId();
                    state.module.AddFunction(spv::OpCompositeExtract, u32, value, loaded, component);
                    construct.push_back(value);
                }
            }
            dword += width;
        }
        state.module.AddFunction(construct);
        return construct[2];
    };
    return EmitValueIfElse(state, both(enabled, within), type, [&] { return fast(load(cache[2])); }, [&] {
        const auto refilled = RefillBdaReadCache(ctx, address, end, enabled);
        return EmitValueIfElse(state, refilled.covers, type, [&] { return fast(refilled.fields[2]); }, miss);
    });
}

bool BdaMaskedDwordReadsAvailable(const SpirvEmitterState& state, const IrValue& inst) {
    return state.bdaProbeFunction != 0u && state.bdaSpanReadFunctions[0][BdaCoherent(state, inst)] != 0u;
}

std::array<std::uint32_t, 4> EmitBdaMaskedDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t used) {
    auto& state = ctx.state;
    if (state.bdaStopsInvocations) ctx.Fail(inst, "masked BDA dword reads cannot stop invocations");
    const auto function = state.bdaSpanReadFunctions[0][BdaCoherent(state, inst)];
    if (function == 0) ctx.Fail(inst, "BDA span read function is missing");
    const auto u32 = TypeU32(state);
    const auto instruction = BdaInstructionPc(state, inst);
    return EmitValuesOrDefaultIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), used, ConstantU32(state, 0u)), u32, ConstantU32(state, 0u), 4u, [&] {
        const auto read = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, BdaSpanReadResultType(state), read, function, address, ConstantU32(state, 0u), used, instruction);
        std::array<std::uint32_t, 4> values{};
        for (std::uint32_t dword = 0; dword < 4u; ++dword) {
            values[dword] = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, u32, values[dword], read, dword);
        }
        return values;
    });
}

void EmitBdaWriteBytes(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bytes, std::uint32_t enabled) {
    auto& state = ctx.state;
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto function = state.bdaWriteFunctions[BdaCoherent(state, inst)];
    if (function == 0) ctx.Fail(inst, "BDA write function is missing");
    state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), function, address, value, bytes, BdaInstructionPc(state, inst), enabled);
}

void EmitBdaWrite(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits, std::uint32_t enabled) {
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA write width");
    EmitBdaWriteBytes(ctx, inst, address, value, ConstantU32(ctx.state, bits / 8u), enabled);
}

void DefineBdaByteWriteFunctions(SpirvEmitterState& state) {
    if (state.bdaFaultFunction == 0) throw std::runtime_error("BDA fault function is missing");
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
    const auto bytePointer = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType);
    const auto translation = state.module.Type(spv::OpTypeStruct, u64, u32);
    const auto type = state.module.Type(spv::OpTypeFunction, voidType, u64, u32, u32, u32, boolean, boolean);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t resultType, std::uint32_t left, std::uint32_t right) { return Binary(state, op, resultType, left, right); };
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    for (const bool coherent : {false, true}) {
        if (coherent && !coherentAccesses) continue;
        const auto function = state.module.AllocateId();
        state.module.AddName(function, std::string("write_bda_bytes") + (coherent ? "_coherent" : ""));
        state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlDontInlineMask, type);
        const auto address = state.module.AllocateId();
        const auto value = state.module.AllocateId();
        const auto bytes = state.module.AllocateId();
        const auto instruction = state.module.AllocateId();
        const auto dword = state.module.AllocateId();
        const auto overflow = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionParameter, u64, address);
        state.module.AddFunction(spv::OpFunctionParameter, u32, value);
        state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
        state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
        state.module.AddFunction(spv::OpFunctionParameter, boolean, dword);
        state.module.AddFunction(spv::OpFunctionParameter, boolean, overflow);
        EmitLabel(state, state.module.AllocateId());
        const auto pieces = Select(state, u32, dword, constant(1u), bytes);
        const auto width = Select(state, u32, dword, constant(4u), constant(1u));
        EmitCountedLoop(state, pieces, [&](std::uint32_t piece) {
            const auto guest = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, piece));
            const auto translated = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, translation, translated, state.bdaWriteTranslateFunction, guest, width, constant(BdaAbi::Write));
            const auto device = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, u64, device, translated, 0u);
            const auto reason = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, u32, reason, translated, 1u);
            EmitIfCondition(state, binary(spv::OpLogicalOr, boolean, overflow, binary(spv::OpINotEqual, boolean, reason, constant(0u))), [&] {
                const auto recorded = Select(state, u32, overflow, constant(static_cast<std::uint32_t>(BdaAbi::FaultReason::Overflow)), reason);
                RecordBdaFaultReason(state, recorded, Select(state, u64, overflow, address, guest), Select(state, u32, overflow, bytes, width), instruction);
            });
            EmitIfCondition(state, binary(spv::OpINotEqual, boolean, device, BdaConstant(state, 0u)), [&] {
                EmitIfCondition(state, dword, [&] {
                    const auto pointer = state.module.AllocateId();
                    state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, device);
                    state.module.AddFunction(spv::OpStore, pointer, value, BdaAccessMask(coherent), 4u);
                });
                EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, dword), [&] {
                    const auto shifted = binary(spv::OpShiftRightLogical, u32, value, binary(spv::OpShiftLeftLogical, u32, piece, constant(3u)));
                    const auto pointer = state.module.AllocateId();
                    state.module.AddFunction(spv::OpConvertUToPtr, bytePointer, pointer, device);
                    state.module.AddFunction(spv::OpStore, pointer, Unary(state, spv::OpUConvert, byteType, shifted), BdaAccessMask(coherent), 1u);
                });
                state.module.AddFunction(spv::OpFunctionCall, voidType, state.module.AllocateId(), state.bdaNoteWriteFunction, guest);
            });
        });
        state.module.AddFunction(spv::OpReturn);
        state.module.AddFunction(spv::OpFunctionEnd);
        state.bdaByteWriteFunctions[coherent] = function;
    }
}

void DefineBdaWriteFunctions(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto type = state.module.Type(spv::OpTypeFunction, voidType, u64, u32, u32, u32, boolean);
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    for (const bool coherent : {false, true}) {
        if (coherent && !coherentAccesses) continue;
        const auto function = state.module.AllocateId();
        state.module.AddName(function, std::string("write_bda") + (coherent ? "_coherent" : ""));
        state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlDontInlineMask, type);
        const auto address = state.module.AllocateId();
        const auto value = state.module.AllocateId();
        const auto bytes = state.module.AllocateId();
        const auto instruction = state.module.AllocateId();
        const auto enabled = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionParameter, u64, address);
        state.module.AddFunction(spv::OpFunctionParameter, u32, value);
        state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
        state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
        state.module.AddFunction(spv::OpFunctionParameter, boolean, enabled);
        EmitLabel(state, state.module.AllocateId());
        EmitIfCondition(state, enabled, [&] {
            const auto dword = Binary(state, spv::OpLogicalAnd, boolean, Binary(state, spv::OpIEqual, boolean, bytes, ConstantU32(state, 4u)), Unary(state, spv::OpLogicalNot, boolean, BdaAddressUnaligned(state, address)));
            const auto limit = Binary(state, spv::OpISub, u64, BdaConstant(state, std::numeric_limits<std::uint64_t>::max()), Unary(state, spv::OpUConvert, u64, bytes));
            const auto overflow = Binary(state, spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, dword), Binary(state, spv::OpUGreaterThan, boolean, address, limit));
            state.module.AddFunction(spv::OpFunctionCall, voidType, state.module.AllocateId(), state.bdaByteWriteFunctions[coherent], address, value, bytes, instruction, dword, overflow);
        });
        state.module.AddFunction(spv::OpReturn);
        state.module.AddFunction(spv::OpFunctionEnd);
        state.bdaWriteFunctions[coherent] = function;
    }
}

namespace {

std::uint32_t DefineBdaComponentReadFunction(SpirvEmitterState& state, bool coherent) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto values = TypeU32Vector(state, 4);
    const auto addressVector = state.module.Type(spv::OpTypeVector, u64, 4u);
    const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
    const auto bytePointer = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType);
    const auto translation = state.module.Type(spv::OpTypeStruct, u64, u32);
    const auto accessMask = BdaAccessMask(coherent);
    const auto zero = ConstantU32(state, 0u);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto equal = [&](std::uint32_t left, std::uint32_t right) { return binary(spv::OpIEqual, boolean, left, right); };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, std::string("read_bda_components") + (coherent ? "_coherent" : ""));
    state.module.AddFunction(spv::OpFunction, values, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, values, addressVector, values, u32));
    const auto addresses = state.module.AllocateId();
    const auto sizes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, addressVector, addresses);
    state.module.AddFunction(spv::OpFunctionParameter, values, sizes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto variable = [&](std::uint32_t type) {
        const auto id = state.module.AllocateId();
        state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, type), id, spv::StorageClassFunction);
        return id;
    };
    const auto cursor = variable(u32);
    const auto bytewise = variable(boolean);
    const auto gathered = variable(values);
    state.module.AddFunction(spv::OpStore, cursor, zero);
    state.module.AddFunction(spv::OpStore, bytewise, ConstantBool(state, false));
    state.module.AddFunction(spv::OpStore, gathered, state.module.Constant(spv::OpConstantComposite, values, zero, zero, zero, zero));
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto load = [&](std::uint32_t type, std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, pointer);
        return value;
    };
    const auto position = load(u32, cursor);
    const auto byteMode = load(boolean, bytewise);
    const auto accumulated = load(values, gathered);
    const auto pending = binary(spv::OpULessThan, boolean, position, constant(16u));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, pending, body, merge);
    EmitLabel(state, body);
    const auto component = binary(spv::OpShiftRightLogical, u32, position, constant(2u));
    const auto within = binary(spv::OpBitwiseAnd, u32, position, constant(3u));
    const auto size = state.module.AllocateId();
    state.module.AddFunction(spv::OpVectorExtractDynamic, u32, size, sizes, component);
    const auto active = binary(spv::OpULessThan, boolean, within, size);
    state.module.AddFunction(spv::OpStore, cursor, binary(spv::OpShiftLeftLogical, u32, binary(spv::OpIAdd, u32, component, constant(1u)), constant(2u)));
    state.module.AddFunction(spv::OpStore, bytewise, ConstantBool(state, false));
    const auto nextByte = [&] {
        const auto next = binary(spv::OpIAdd, u32, position, constant(1u));
        state.module.AddFunction(spv::OpStore, cursor, next);
        state.module.AddFunction(spv::OpStore, bytewise, binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, next, constant(3u)), zero));
    };
    const auto insert = [&](std::uint32_t value) {
        const auto updated = state.module.AllocateId();
        state.module.AddFunction(spv::OpVectorInsertDynamic, values, updated, accumulated, value, component);
        state.module.AddFunction(spv::OpStore, gathered, updated);
    };
    const auto byteAt = [&](std::uint32_t at) {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, bytePointer, pointer, at);
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, accessMask, 1u);
        return Unary(state, spv::OpUConvert, u32, loaded);
    };
    EmitIfCondition(state, active, [&] {
        const auto base = state.module.AllocateId();
        state.module.AddFunction(spv::OpVectorExtractDynamic, u64, base, addresses, component);
        const auto guest = binary(spv::OpIAdd, u64, base, Unary(state, spv::OpUConvert, u64, within));
        const auto single = binary(spv::OpLogicalOr, boolean, byteMode, binary(spv::OpINotEqual, boolean, within, zero));
        const auto whole = Unary(state, spv::OpLogicalNot, boolean, single);
        const auto overflow = binary(spv::OpLogicalAnd, boolean, whole, binary(spv::OpUGreaterThan, boolean, guest, binary(spv::OpISub, u64, BdaConstant(state, std::numeric_limits<std::uint64_t>::max()), Unary(state, spv::OpUConvert, u64, size))));
        const auto want = Select(state, u32, single, constant(1u), size);
        const auto translate = [&] {
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, translation, value, state.bdaTranslateFunction, guest, want, constant(BdaAbi::Read));
            return value;
        };
        auto device = 0u;
        auto reason = 0u;
        auto mapped = 0u;
        if (BdaVectorCacheAvailable(state)) {
            const auto& cache = state.bdaVectorCache;
            const auto end = binary(spv::OpIAdd, u64, guest, Unary(state, spv::OpUConvert, u64, want));
            const auto inside = [&] {
                const auto held = binary(spv::OpLogicalAnd, boolean, binary(spv::OpULessThanEqual, boolean, load(u64, cache[0]), guest), binary(spv::OpULessThanEqual, boolean, end, load(u64, cache[1])));
                return binary(spv::OpLogicalAnd, boolean, binary(spv::OpULessThan, boolean, guest, end), held);
            };
            EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, inside()), [&] { RefreshBdaVectorCache(state, guest); });
            mapped = inside();
            const auto delta = load(u64, cache[2]);
            device = binary(spv::OpIAdd, u64, guest, delta);
            reason = Select(state, u32, mapped, zero, Unary(state, spv::OpUConvert, u32, delta));
        } else {
            const auto translated = translate();
            device = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, u64, device, translated, 0u);
            reason = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, u32, reason, translated, 1u);
            mapped = equal(reason, zero);
        }
        EmitIfCondition(state, mapped, [&] {
            EmitIfCondition(state, whole, [&] {
                const auto dword = binary(spv::OpLogicalAnd, boolean, equal(size, constant(4u)), Unary(state, spv::OpLogicalNot, boolean, BdaAddressUnaligned(state, device)));
                insert(EmitValueIfElse(state, dword, u32, [&] {
                    const auto pointer = state.module.AllocateId();
                    state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, device);
                    const auto value = state.module.AllocateId();
                    if (coherent) state.module.AddFunction(spv::OpAtomicLoad, u32, value, pointer, ConstantU32(state, spv::ScopeDevice), zero);
                    else state.module.AddFunction(spv::OpLoad, u32, value, pointer, accessMask, 4u);
                    return value;
                }, [&] {
                    auto value = byteAt(device);
                    for (std::uint32_t byte = 1u; byte < 4u; ++byte) {
                        const auto part = EmitValueOrZeroIfCondition(state, binary(spv::OpULessThan, boolean, constant(byte), size), [&] { return byteAt(binary(spv::OpIAdd, u64, device, BdaConstant(state, byte))); });
                        value = binary(spv::OpBitwiseOr, u32, value, binary(spv::OpShiftLeftLogical, u32, part, constant(byte * 8u)));
                    }
                    return value;
                }));
            });
            EmitIfCondition(state, single, [&] {
                const auto part = state.module.AllocateId();
                state.module.AddFunction(spv::OpVectorExtractDynamic, u32, part, accumulated, component);
                insert(binary(spv::OpBitwiseOr, u32, part, binary(spv::OpShiftLeftLogical, u32, byteAt(device), binary(spv::OpShiftLeftLogical, u32, within, constant(3u)))));
                nextByte();
            });
        });
        const auto unmapped = Unary(state, spv::OpLogicalNot, boolean, mapped);
        const auto again = binary(spv::OpLogicalAnd, boolean, whole, binary(spv::OpUGreaterThan, boolean, size, constant(1u)));
        const auto settled = Unary(state, spv::OpLogicalNot, boolean, again);
        EmitIfCondition(state, unmapped, [&] {
            EmitIfCondition(state, again, [&] {
                state.module.AddFunction(spv::OpStore, cursor, position);
                state.module.AddFunction(spv::OpStore, bytewise, ConstantBool(state, true));
            });
            EmitIfCondition(state, settled, nextByte);
        });
        EmitIfCondition(state, binary(spv::OpLogicalOr, boolean, overflow, binary(spv::OpLogicalAnd, boolean, unmapped, settled)), [&] {
            const auto faultReason = Select(state, u32, overflow, constant(static_cast<std::uint32_t>(BdaAbi::FaultReason::Overflow)), reason);
            RecordBdaFaultReason(state, faultReason, guest, Select(state, u32, overflow, size, constant(1u)), instruction);
        });
    });
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    state.module.AddFunction(spv::OpReturnValue, load(values, gathered));
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

void DefineBdaComponentReadFunctions(SpirvEmitterState& state) {
    if (state.bdaTranslateFunction == 0 || state.bdaFaultFunction == 0) return;
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    for (const bool coherent : {false, true}) {
        if (coherent && !coherentAccesses) continue;
        state.bdaComponentReadFunctions[coherent] = DefineBdaComponentReadFunction(state, coherent);
    }
}

bool BdaComponentReadsAvailable(const SpirvEmitterState& state, const IrValue& inst) {
    return state.bdaComponentReadFunctions[BdaCoherent(state, inst)] != 0;
}

std::array<std::uint32_t, 4> EmitBdaComponentReads(SpirvValueEmitContext& ctx, const IrValue& inst, const std::array<std::uint32_t, 4>& addresses, const std::array<std::uint32_t, 4>& sizes) {
    auto& state = ctx.state;
    if (state.bdaStopsInvocations) ctx.Fail(inst, "BDA component reads cannot stop invocations");
    const auto function = state.bdaComponentReadFunctions[BdaCoherent(state, inst)];
    if (function == 0) ctx.Fail(inst, "BDA component read function is missing");
    const auto u32 = TypeU32(state);
    const auto addressVector = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, state.module.Type(spv::OpTypeVector, TypeScalarU64(state), 4u), addressVector, addresses[0], addresses[1], addresses[2], addresses[3]);
    const auto sizeVector = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4u), sizeVector, sizes[0], sizes[1], sizes[2], sizes[3]);
    const auto read = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 4u), read, function, addressVector, sizeVector, BdaInstructionPc(state, inst));
    std::array<std::uint32_t, 4> values{};
    for (std::uint32_t component = 0; component < 4u; ++component) {
        values[component] = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, values[component], read, component);
    }
    return values;
}

void EmitBdaStore(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits) {
    auto& state = ctx.state;
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA store width");
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto function = state.bdaByteWriteFunctions[BdaCoherent(state, inst)];
    if (function == 0) ctx.Fail(inst, "BDA byte write function is missing");
    const auto dword = bits == 32u ? Unary(state, spv::OpLogicalNot, TypeBool(state), BdaAddressUnaligned(state, address)) : ConstantBool(state, false);
    state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), function, address, value, ConstantU32(state, bits / 8u), BdaInstructionPc(state, inst), dword, ConstantBool(state, false));
}

void EmitBdaDwordWrites(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, std::uint32_t value) {
    auto& state = ctx.state;
    if (dwords < 2u || dwords > 4u) ctx.Fail(inst, "unsupported BDA dword store count");
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    std::array<std::uint32_t, 4> values{};
    std::array<std::uint32_t, 4> addresses{};
    for (std::uint32_t dword = 0; dword < dwords; ++dword) {
        values[dword] = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u32, values[dword], value, dword);
        addresses[dword] = AddBdaImmediate(ctx, inst, address, static_cast<std::int32_t>(offset + dword * 4u));
    }
    const auto storeDwords = [&] {
        const auto vector = state.module.AllocateId();
        std::vector<std::uint32_t> construct{spv::OpCompositeConstruct, TypeU32Vector(state, dwords), vector};
        construct.insert(construct.end(), values.begin(), values.begin() + dwords);
        state.module.AddFunction(construct);
        EmitCountedLoop(state, ConstantU32(state, dwords), [&](std::uint32_t dword) {
            auto selected = addresses[0];
            for (std::uint32_t index = 1; index < dwords; ++index) selected = Select(state, u64, Binary(state, spv::OpIEqual, boolean, dword, ConstantU32(state, index)), addresses[index], selected);
            const auto element = state.module.AllocateId();
            state.module.AddFunction(spv::OpVectorExtractDynamic, u32, element, vector, dword);
            EmitBdaStore(ctx, inst, selected, element, 32u);
        });
    };
    if (state.bdaWriteProbeFunction == 0) {
        storeDwords();
        return;
    }
    const auto instruction = BdaInstructionPc(state, inst);
    const auto accessMask = BdaAccessMask(state, inst);
    const auto physical = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, u64, physical, state.bdaWriteProbeFunction, addresses[0], ConstantU32(state, dwords * 4u), instruction);
    const auto mapped = Binary(state, spv::OpINotEqual, boolean, physical, BdaConstant(state, 0u));
    const auto aligned = Binary(state, spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, BdaAddressUnaligned(state, addresses[0])), Unary(state, spv::OpLogicalNot, boolean, BdaAddressUnaligned(state, physical)));
    const auto wide = Binary(state, spv::OpLogicalAnd, boolean, mapped, aligned);
    const auto wideLabel = state.module.AllocateId();
    const auto dwordLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, wide, wideLabel, dwordLabel);
    EmitLabel(state, wideLabel);
    const auto vectorBytes = dwords == 3u ? 4u : dwords * 4u;
    const auto separate = [&] {
        for (std::uint32_t dword = 0; dword < dwords; ++dword) {
            const auto element = dword == 0u ? physical : Binary(state, spv::OpIAdd, u64, physical, BdaConstant(state, dword * 4u));
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, element);
            state.module.AddFunction(spv::OpStore, pointer, values[dword], accessMask, 4u);
        }
    };
    if (dwords == 3u) {
        separate();
    } else {
        const auto vectorAligned = Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u64, physical, BdaConstant(state, vectorBytes - 1u)), BdaConstant(state, 0u));
        const auto vectorType = TypeU32Vector(state, dwords);
        const auto vector = state.module.AllocateId();
        if (dwords == 2u) state.module.AddFunction(spv::OpCompositeConstruct, vectorType, vector, values[0], values[1]);
        else state.module.AddFunction(spv::OpCompositeConstruct, vectorType, vector, values[0], values[1], values[2], values[3]);
        EmitIfCondition(state, vectorAligned, [&] {
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, vectorType), pointer, physical);
            state.module.AddFunction(spv::OpStore, pointer, vector, accessMask, vectorBytes);
        });
        EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, vectorAligned), separate);
    }
    const auto last = addresses[dwords - 1u];
    state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, addresses[0]);
    const auto pageShift = BdaConstant(state, BdaAbi::WrittenPageShift);
    const auto crosses = Binary(state, spv::OpINotEqual, boolean, Binary(state, spv::OpShiftRightLogical, u64, addresses[0], pageShift), Binary(state, spv::OpShiftRightLogical, u64, last, pageShift));
    EmitIfCondition(state, crosses, [&] {
        state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, last);
    });
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, dwordLabel);
    storeDwords();
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
}

std::uint32_t EmitBdaAtomic(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bytes, const std::function<std::uint32_t(std::uint32_t)>& operation) {
    auto& state = ctx.state;
    if (bytes != 4u && bytes != 8u) ctx.Fail(inst, "unsupported BDA atomic width");
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto instruction = BdaInstructionPc(state, inst);
    const auto type = bytes == 8u ? TypeScalarU64(state) : TypeU32(state);
    const auto zero = bytes == 8u ? BdaConstant(state, 0u) : ConstantU32(state, 0u);
    const auto unaligned = Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), address, BdaConstant(state, bytes - 1u)), BdaConstant(state, 0u));
    RecordBdaFault(state, address, ConstantU32(state, bytes), instruction, BdaAbi::FaultReason::Unaligned, unaligned);
    return EmitValueOrDefaultIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), type, zero, [&] {
        const auto physical = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaWritePointerFunction, address, ConstantU32(state, bytes), instruction);
        return EmitValueOrDefaultIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), type, zero, [&] {
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, type), pointer, physical);
            const auto old = operation(pointer);
            state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, address);
            return old;
        });
    });
}

std::array<std::uint32_t, 4> EmitBdaDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword) {
    auto& state = ctx.state;
    if (dwords == 0u || dwords > 4u) ctx.Fail(inst, "unsupported BDA read width");
    if (state.bdaPointerFunction == 0) ctx.Fail(inst, "BDA lookup function is missing");
    const auto instruction = BdaInstructionPc(state, inst);
    const auto used = everyDword ? (1u << dwords) - 1u : UsedBdaDwords(inst, dwords);
    const auto isUsed = [&](std::uint32_t dword) { return (used & (1u << dword)) != 0u; };
    std::array<std::uint32_t, 4> values{};
    std::uint32_t first = dwords;
    std::uint32_t last = 0;
    for (std::uint32_t dword = 0; dword < dwords; ++dword) {
        if (!isUsed(dword)) {
            values[dword] = ConstantU32(state, 0u);
            continue;
        }
        first = std::min(first, dword);
        last = dword;
    }
    if (used == 0u) return values;
    std::array<std::uint32_t, 4> addresses{};
    const auto dwordAddress = [&](std::uint32_t dword) { return AddBdaImmediate(ctx, inst, address, static_cast<std::int32_t>(offset + dword * 4u)); };
    const auto readBytes = [&](std::array<std::uint32_t, 4>& into) {
        const auto function = state.bdaDwordReadFunctions[state.bdaStopsInvocations][BdaCoherent(state, inst)];
        if (function == 0) ctx.Fail(inst, "BDA dword read function is missing");
        for (std::uint32_t dword = first; dword <= last; ++dword) {
            if (!isUsed(dword)) continue;
            if (dword != first) addresses[dword] = dwordAddress(dword);
            const auto read = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 2), read, function, addresses[dword], instruction);
            into[dword] = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), into[dword], read, 0u);
            if (!state.bdaStopsInvocations) continue;
            const auto stopped = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), stopped, read, 1u);
            StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, TypeBool(state), stopped, ConstantU32(state, 0u)));
        }
    };
    if (!BdaVectorCacheAvailable(state)) {
        addresses[first] = dwordAddress(first);
        readBytes(values);
        return values;
    }
    const auto function = state.bdaSpanReadFunctions[state.bdaStopsInvocations][BdaCoherent(state, inst)];
    if (function == 0) ctx.Fail(inst, "BDA span read function is missing");
    const auto vector = TypeU32Vector(state, 4);
    const auto span = [&] {
        const auto read = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, BdaSpanReadResultType(state), read, function, address, ConstantU32(state, offset), ConstantU32(state, used), instruction);
        if (!state.bdaStopsInvocations) return read;
        const auto stopped = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), stopped, read, 1u);
        StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, TypeBool(state), stopped, ConstantU32(state, 0u)));
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, vector, loaded, read, 0u);
        return loaded;
    };
    auto read = 0u;
    if (state.module.InHelperFunction()) {
        const auto u64 = TypeScalarU64(state);
        const auto boolean = TypeBool(state);
        const auto start = static_cast<std::int32_t>(offset + first * 4u);
        const auto magnitude = BdaConstant(state, start < 0 ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(start)) : static_cast<std::uint64_t>(start));
        const auto firstAddress = Binary(state, start < 0 ? spv::OpISub : spv::OpIAdd, u64, address, magnitude);
        const auto unwrapped = Binary(state, spv::OpUGreaterThanEqual, boolean, start < 0 ? address : firstAddress, start < 0 ? magnitude : address);
        const auto cached = ProbeBdaVectorCache(state, firstAddress, BdaConstant(state, (last - first + 1u) * 4u));
        read = EmitValueIfElse(state, Binary(state, spv::OpLogicalAnd, boolean, unwrapped, cached.hit), vector, [&] {
            std::array<std::uint32_t, 4> loaded{};
            for (std::uint32_t dword = 0; dword < 4u; ++dword) {
                loaded[dword] = ConstantU32(state, 0u);
                if (dword < first || dword > last || !isUsed(dword)) continue;
                loaded[dword] = LoadBdaDevice(state, inst, dword == first ? cached.device : Binary(state, spv::OpIAdd, u64, cached.device, BdaConstant(state, (dword - first) * 4u)));
            }
            const auto composite = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, vector, composite, loaded[0], loaded[1], loaded[2], loaded[3]);
            return composite;
        }, span);
    } else {
        read = span();
    }
    for (std::uint32_t dword = first; dword <= last; ++dword) {
        if (!isUsed(dword)) continue;
        values[dword] = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), values[dword], read, dword);
    }
    return values;
}

}
