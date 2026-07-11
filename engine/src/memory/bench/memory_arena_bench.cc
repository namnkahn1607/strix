// Author: namnkahn1607
//
// Benchmarks for MemoryArena::WritePayload() and ReadPayload().
//
// Two buffer sizes, two write patterns = 4 benchmark families:
//   Buffer sizes:
//     8 MB  : fits in L3 cache.
//             Isolates instruction overhead and branch misprediction.
//     128 MB: exceeds any L3 cache.
//             Measures memory bus bandwidth and TLB miss cost.
//
//   Write patterns:
//     Sequential : write_head positioned so data never straddles the
//                  boundary. Single memcpy path every time.
//                  Measures best-case throughput.
//     Wrap       : write_head pinned exactly sizeof(PayloadHeader) bytes
//                  before the boundary via start_point. Every write hits
//                  the two-memcpy path.
//                  Measures branching penalty + double-memcpy overhead.
//
// All configs use lazy_mapping = true (no MAP_POPULATE).

#include <benchmark/benchmark.h>

#include "memory_arena.h"

namespace {

inline constexpr size_t   kSlots      = 4;
inline constexpr size_t   kPayloadLen = 1024;
inline constexpr uint32_t kNode       = 0;
inline constexpr size_t   kBuf8MB     = 8ULL * 1024 * 1024;
inline constexpr size_t   kBuf128MB   = 128ULL * 1024 * 1024;
inline constexpr size_t   kHeaderSize = sizeof(PayloadHeader);

// Global payload - not semantically meaningful.
alignas(32) uint8_t g_payload[kPayloadLen];

// InitPayload(): inititalize the global payload data.
void InitPayload() {
    for (size_t i = 0; i < kPayloadLen; ++i) {
        g_payload[i] = static_cast<uint8_t>((i * 37 + 13) % 251);
    }
}

// WrapStartPoint(): calculates placing index to force data split.
constexpr uint64_t WrapStartPoint(const size_t buf) {
    return buf - kHeaderSize - (kPayloadLen - 2);
}

}  // namespace

// -----------------------------------------------------------------------------
// BenchSequential_8MB
// Measures best-case Write + Read throughput when no wrap occurs.
// At some point the Write operation will perform WRAPPING, but the benchmark
// runs long enough that the average is dominated by the sequential path.
// -----------------------------------------------------------------------------

static void BenchSequential_8MB(benchmark::State& state) {
    InitPayload();
    std::string out;

    ArenaConfig config{kSlots, kBuf8MB, false};

    for ([[maybe_unused]] auto _ : state) {
        state.PauseTiming();
        {
            MemoryArena arena{config};
            arena.PrefaultBuffer();

            state.ResumeTiming();

            std::optional<uint64_t> offset_opt =
                arena.WritePayload(kNode, g_payload, kPayloadLen);
            benchmark::DoNotOptimize(offset_opt);
            benchmark::ClobberMemory();

            arena.ReadPayload(*offset_opt, kPayloadLen, &out);
            benchmark::DoNotOptimize(out);

            state.PauseTiming();
        }
        state.ResumeTiming();
    }

    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(kPayloadLen));
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BenchSequential_8MB)
    ->Iterations(10'000)
    ->Unit(benchmark::kNanosecond)
    ->MinTime(2.0);

// -----------------------------------------------------------------------------
// BenchSequential_128MB
// Same pattern, 128 MB buffer. Working set exceeds L3 cache.
// Measures memory bus bandwidth when data must be fetched from DRAM.
// -----------------------------------------------------------------------------

static void BenchSequential_128MB(benchmark::State& state) {
    InitPayload();
    std::string out;

    ArenaConfig config{kSlots, kBuf128MB, false};

    for ([[maybe_unused]] auto _ : state) {
        state.PauseTiming();
        {
            MemoryArena arena{config};
            arena.PrefaultBuffer();

            state.ResumeTiming();

            auto offset_opt = arena.WritePayload(kNode, g_payload, kPayloadLen);
            benchmark::DoNotOptimize(offset_opt);
            benchmark::ClobberMemory();

            arena.ReadPayload(*offset_opt, kPayloadLen, &out);
            benchmark::DoNotOptimize(out);

            state.PauseTiming();
        }
        state.ResumeTiming();
    }

    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(kPayloadLen));
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BenchSequential_128MB)
    ->Iterations(10'000)
    ->Unit(benchmark::kNanosecond)
    ->MinTime(2.0);

// -----------------------------------------------------------------------------
// BenchWrap_8MB
// write_head pre-positioned to force data split on every write.
// WRAP happens at every iteration. Measures branch misprediction penalty
// and double-memcpy overhead with working set inside L3 cache.
// -----------------------------------------------------------------------------

static void BenchWrap_8MB(benchmark::State& state) {
    InitPayload();
    std::string out;

    const uint64_t start_point = WrapStartPoint(kBuf8MB);
    ArenaConfig    config{kSlots, kBuf8MB, true, start_point};

    for ([[maybe_unused]] auto _ : state) {
        state.PauseTiming();
        {
            MemoryArena arena{config};
            arena.PrefaultBuffer();

            state.ResumeTiming();

            auto offset_opt = arena.WritePayload(kNode, g_payload, kPayloadLen);
            benchmark::DoNotOptimize(offset_opt);
            benchmark::ClobberMemory();

            arena.ReadPayload(*offset_opt, kPayloadLen, &out);
            benchmark::DoNotOptimize(out);

            state.PauseTiming();
        }
        state.ResumeTiming();
    }

    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(kPayloadLen));
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BenchWrap_8MB)
    ->Iterations(10'000)
    ->Unit(benchmark::kNanosecond)
    ->MinTime(2.0);

// -----------------------------------------------------------------------------
// BenchWrap_128MB
// Same as BenchWrap_8MB but with 128 MB buffer.
// Wrapping into a cold page adds TLB miss cost on top of branch
// misprediction, which turns out measuring combined worst case.
// -----------------------------------------------------------------------------

static void BenchWrap_128MB(benchmark::State& state) {
    InitPayload();
    std::string out;

    const uint64_t start_point = WrapStartPoint(kBuf128MB);
    ArenaConfig    config{kSlots, kBuf128MB, true, start_point};

    for ([[maybe_unused]] auto _ : state) {
        state.PauseTiming();
        {
            MemoryArena arena{config};
            arena.PrefaultBuffer();

            state.ResumeTiming();

            auto offset_opt = arena.WritePayload(kNode, g_payload, kPayloadLen);
            benchmark::DoNotOptimize(offset_opt);
            benchmark::ClobberMemory();

            arena.ReadPayload(*offset_opt, kPayloadLen, &out);
            benchmark::DoNotOptimize(out);

            state.PauseTiming();
        }
        state.ResumeTiming();
    }

    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(kPayloadLen));
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BenchWrap_128MB)
    ->Iterations(10'000)
    ->Unit(benchmark::kNanosecond)
    ->MinTime(2.0);

BENCHMARK_MAIN();
