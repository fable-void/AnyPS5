#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <array>
#include <stdexcept>
#include <vector>

namespace ShaderRecompiler {

namespace {

void HoldBdaFault(SpirvEmitterState& state, std::uint32_t condition, const std::array<std::uint32_t, 5>& fault) {
    if (state.bdaFaultPending[0] == 0u) throw std::runtime_error("BDA fault function is missing");
    const auto held = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), held, state.bdaFaultPending[0]);
    const auto vacant = Binary(state, spv::OpIEqual, TypeBool(state), held, ConstantU32(state, 0u));
    EmitIfCondition(state, condition != 0u ? Binary(state, spv::OpLogicalAnd, TypeBool(state), condition, vacant) : vacant, [&] {
        for (std::size_t field = 0; field < fault.size(); ++field) {
            if (state.bdaFaultPending[field] != 0u) state.module.AddFunction(spv::OpStore, state.bdaFaultPending[field], fault[field]);
        }
    });
}

}

std::uint32_t BdaConstant(SpirvEmitterState& state, std::uint64_t value) {
    return state.module.Constant(spv::OpConstant, TypeScalarU64(state), static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32u));
}

std::uint32_t BdaWord(SpirvEmitterState& state, std::uint32_t variable, std::uint32_t index) {
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer, variable, ConstantU32(state, 0u), index);
    return pointer;
}

std::uint32_t BdaLoadWord(SpirvEmitterState& state, std::uint32_t index) {
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), value, BdaWord(state, state.bdaPagetableVariable, index));
    return value;
}

std::uint32_t BdaLoadAddress(SpirvEmitterState& state, std::uint32_t index) {
    const auto low = Unary(state, spv::OpUConvert, TypeScalarU64(state), BdaLoadWord(state, index));
    const auto high = Unary(state, spv::OpUConvert, TypeScalarU64(state), BdaLoadWord(state, Binary(state, spv::OpIAdd, TypeU32(state), index, ConstantU32(state, 1u))));
    return Binary(state, spv::OpBitwiseOr, TypeScalarU64(state), low, Binary(state, spv::OpShiftLeftLogical, TypeScalarU64(state), high, BdaConstant(state, 32u)));
}

void DefineBdaFaultFunction(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const bool words = !state.program.Info().usesDma;
    const auto addressType = words ? u32 : TypeScalarU64(state);
    state.bdaFaultFunction = state.module.AllocateId();
    state.module.AddName(state.bdaFaultFunction, "record_bda_fault");
    const auto type = words ? state.module.Type(spv::OpTypeFunction, TypeVoid(state), u32, addressType, u32, u32, u32) : state.module.Type(spv::OpTypeFunction, TypeVoid(state), u32, addressType, u32, u32);
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), state.bdaFaultFunction, spv::FunctionControlDontInlineMask, type);
    const auto reason = state.module.AllocateId();
    const auto address = state.module.AllocateId();
    const auto addressHigh = words ? state.module.AllocateId() : 0u;
    const auto bytes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u32, reason);
    state.module.AddFunction(spv::OpFunctionParameter, addressType, address);
    if (words) state.module.AddFunction(spv::OpFunctionParameter, u32, addressHigh);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto pointer = BdaWord(state, state.faultBufferVariable, ConstantU32(state, 0u));
    const auto previous = state.module.AllocateId();
    const auto scope = ConstantU32(state, spv::ScopeDevice);
    const auto relaxed = ConstantU32(state, spv::MemorySemanticsMaskNone);
    state.module.AddFunction(spv::OpAtomicCompareExchange, TypeU32(state), previous, pointer, scope, relaxed, relaxed, ConstantU32(state, static_cast<std::uint32_t>(BdaAbi::FaultState::Writing)), ConstantU32(state, 0u));
    const auto won = Binary(state, spv::OpIEqual, TypeBool(state), previous, ConstantU32(state, 0u));
    EmitIfCondition(state, won, [&] {
        const auto store = [&](std::uint32_t index, std::uint32_t value) {
            state.module.AddFunction(spv::OpStore, BdaWord(state, state.faultBufferVariable, ConstantU32(state, index)), value);
        };
        store(1, reason);
        store(2, words ? address : Unary(state, spv::OpUConvert, TypeU32(state), address));
        store(3, words ? addressHigh : Unary(state, spv::OpUConvert, TypeU32(state), Binary(state, spv::OpShiftRightLogical, TypeScalarU64(state), address, BdaConstant(state, 32u))));
        store(4, bytes);
        store(5, ConstantU32(state, static_cast<std::uint32_t>(state.program.Resources().stage)));
        store(6, instruction);
        store(7, ConstantU32(state, 0u));
        const auto release = ConstantU32(state, spv::MemorySemanticsReleaseMask | spv::MemorySemanticsUniformMemoryMask);
        state.module.AddFunction(spv::OpAtomicStore, pointer, scope, release, ConstantU32(state, static_cast<std::uint32_t>(BdaAbi::FaultState::Ready)));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    const std::array<std::uint32_t, 5> types{u32, addressType, words ? u32 : 0u, u32, u32};
    static constexpr std::array<const char*, 5> names{"bda_fault_reason", "bda_fault_address", "bda_fault_address_high", "bda_fault_bytes", "bda_fault_instruction"};
    for (std::size_t field = 0; field < types.size(); ++field) {
        state.bdaFaultPending[field] = 0u;
        if (types[field] == 0u) continue;
        state.bdaFaultPending[field] = state.module.AllocateId();
        state.module.AddName(state.bdaFaultPending[field], names[field]);
        state.module.EmitGlobalVariable({spv::OpVariable, TypePointer(state, spv::StorageClassPrivate, types[field]), state.bdaFaultPending[field], spv::StorageClassPrivate, types[field] == u32 ? ConstantU32(state, 0u) : BdaConstant(state, 0u)});
    }
}

void RecordBdaFaultReason(SpirvEmitterState& state, std::uint32_t reason, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t condition) {
    if (state.bdaFaultFunction == 0 || !state.program.Info().usesDma) throw std::runtime_error("BDA fault function is missing");
    HoldBdaFault(state, condition, {reason, address, 0u, bytes, instruction});
}

void RecordBdaFault(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason, std::uint32_t condition) {
    RecordBdaFaultReason(state, ConstantU32(state, static_cast<std::uint32_t>(reason)), address, bytes, instruction, condition);
}

void RecordBdaFaultWords(SpirvEmitterState& state, std::uint32_t addressLow, std::uint32_t addressHigh, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason) {
    if (state.program.Info().usesDma) {
        const auto u64 = TypeScalarU64(state);
        const auto high = Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, addressHigh), BdaConstant(state, 32u));
        RecordBdaFault(state, Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, addressLow), high), bytes, instruction, reason);
        return;
    }
    HoldBdaFault(state, 0u, {ConstantU32(state, static_cast<std::uint32_t>(reason)), addressLow, addressHigh, bytes, instruction});
}

void FlushBdaFault(SpirvEmitterState& state) {
    if (state.bdaFaultPending[0] == 0u) return;
    const auto load = [&](std::size_t field) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, field == 1u && state.program.Info().usesDma ? TypeScalarU64(state) : TypeU32(state), value, state.bdaFaultPending[field]);
        return value;
    };
    const auto reason = load(0u);
    EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), reason, ConstantU32(state, 0u)), [&] {
        std::vector<std::uint32_t> call{spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.bdaFaultFunction, reason};
        for (std::size_t field = 1u; field < state.bdaFaultPending.size(); ++field) {
            if (state.bdaFaultPending[field] != 0u) call.push_back(load(field));
        }
        state.module.AddFunction(call);
    });
}

void StopBdaInvocationIf(SpirvEmitterState& state, std::uint32_t condition) {
    if (!state.bdaStopsInvocations) return;
    const auto failed = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
    EmitLabel(state, failed);
    if (state.bdaStopValue != 0) {
        state.module.AddFunction(spv::OpReturnValue, state.bdaStopValue);
    } else {
        FlushBdaFault(state);
        if (OrderedPixelShader(state)) {
            state.module.AddFunction(spv::OpEndInvocationInterlockEXT);
        }
        state.module.AddFunction(state.program.Resources().stage == IrShaderStage::Pixel ? spv::OpKill : spv::OpReturn);
    }
    EmitLabel(state, next);
}

}
