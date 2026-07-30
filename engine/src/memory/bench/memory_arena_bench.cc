// Benchmarks for payload read/write operation in Memory Arena.
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
//                  the double-memcpy path.
//                  Measures branching penalty + double-memcpy overhead.
//
// All configs have MAP_POPULATED disabled.

#include <benchmark/benchmark.h>

#include "memory_arena.h"
#include "payload_header.h"

namespace {

inline constexpr size_t   kSlots      = 4;
inline constexpr size_t   kPayloadLen = 1024;
inline constexpr uint32_t kNode       = 0;
inline constexpr size_t   kHeaderSize = sizeof(PayloadHeader);

inline constexpr size_t   kBuf8MB     = 8ULL * 1024 * 1024;
inline constexpr size_t   kBuf128MB   = 128ULL * 1024 * 1024;

// Global payload buffer.
alignas(32) uint8_t g_payload[kPayloadLen];

// InitPayload initializes the global payload buffer.
void InitPayload() {
    for (size_t i = 0; i < kPayloadLen; ++i) {
        g_payload[i] = static_cast<uint8_t>((i * 37 + 13) % 251);
    }
}

// WrapStartPoint calculates `start_point` value in order to force triggering
// data split.
constexpr uint64_t WrapStartPoint(const size_t buf) {
    return buf - kHeaderSize - (kPayloadLen - 2);
}

}  // namespace

// BenchSequential_8MB measures best-case non-wrap Write + Read throughput.
// At some point the Write operation will perform WRAPPING, but the benchmark
// runs long enough that the average is dominated by the sequential path.
static void BenchSequential_8MB(benchmark::State& state) {
    ArenaConfig config{/*max_slots=*/kSlots, /*payload_buf_size=*/kBuf8MB};
    std::string out;
    InitPayload();

    for ([[maybe_unused]] auto _ : state) {
        // Pause timing for construction.
        state.PauseTiming();
        {
            MemoryArena arena{config};
            MemoryArenaPrivateAccess::PrefaultBuffer(arena);

            state.ResumeTiming();

            auto offset_opt = arena.WritePayload(kNode, g_payload, kPayloadLen);
            benchmark::DoNotOptimize(offset_opt);
            benchmark::ClobberMemory();

            arena.ReadPayload(*offset_opt, kPayloadLen, &out);
            benchmark::DoNotOptimize(out);

            // Pause timing for destruction.
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

// BenchSequential_128MB is the same as `BenchSequential_8MB() but with a
// working set exceeding L3 cache (128 MB).
// Measures memory bus bandwidth when data must be fetched from DRAM.
static void BenchSequential_128MB(benchmark::State& state) {
    ArenaConfig config{/*max_slots=*/kSlots, /*payload_buf_size=*/kBuf128MB};
    std::string out;
    InitPayload();

    for ([[maybe_unused]] auto _ : state) {
        // Pause timing for construction.
        state.PauseTiming();
        {
            MemoryArena arena{config};
            MemoryArenaPrivateAccess::PrefaultBuffer(arena);

            state.ResumeTiming();

            auto offset_opt = arena.WritePayload(kNode, g_payload, kPayloadLen);
            benchmark::DoNotOptimize(offset_opt);
            benchmark::ClobberMemory();

            arena.ReadPayload(*offset_opt, kPayloadLen, &out);
            benchmark::DoNotOptimize(out);

            // Pause timing for destruction.
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

// BenchWrap_8MB has `write_head` pre-positioned to force data split on every
// write. WRAP happens at every iteration.
// Measures branch misprediction penalty and double-memcpy overhead with working
// set inside L3 cache.
static void BenchWrap_8MB(benchmark::State& state) {
    ArenaConfig config{/*max_slots=*/kSlots, /*payload_buf_size=*/kBuf8MB,
                       /*prefault=*/false,
                       /*start_point=*/WrapStartPoint(kBuf8MB)};
    std::string out;
    InitPayload();

    for ([[maybe_unused]] auto _ : state) {
        // Pause timing for construction.
        state.PauseTiming();
        {
            MemoryArena arena{config};
            MemoryArenaPrivateAccess::PrefaultBuffer(arena);

            state.ResumeTiming();

            auto offset_opt = arena.WritePayload(kNode, g_payload, kPayloadLen);
            benchmark::DoNotOptimize(offset_opt);
            benchmark::ClobberMemory();

            arena.ReadPayload(*offset_opt, kPayloadLen, &out);
            benchmark::DoNotOptimize(out);

            // Pause timing for destruction.
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

// BenchWrap_128MB is the same as BenchWrap_8MB but with a working
// set exceeding L3 cache (128 MB).
// Wrapping into a cold page adds TLB miss cost on top of branch
// misprediction, which turns out measuring combined worst case.
static void BenchWrap_128MB(benchmark::State& state) {
    ArenaConfig config{/*max_slots=*/kSlots, /*payload_buf_size=*/kBuf128MB,
                       /*prefault=*/false,
                       /*start_point=*/WrapStartPoint(kBuf128MB)};
    std::string out;
    InitPayload();

    for ([[maybe_unused]] auto _ : state) {
        // Pause timing for construction.
        state.PauseTiming();
        {
            MemoryArena arena{config};
            MemoryArenaPrivateAccess::PrefaultBuffer(arena);

            state.ResumeTiming();

            auto offset_opt = arena.WritePayload(kNode, g_payload, kPayloadLen);
            benchmark::DoNotOptimize(offset_opt);
            benchmark::ClobberMemory();

            arena.ReadPayload(*offset_opt, kPayloadLen, &out);
            benchmark::DoNotOptimize(out);

            // Pause timing for destruction.
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
