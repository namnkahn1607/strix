// Author: namnkahn1607
//
// Orchestrator: initializes all subsystems, injects dependencies,
// and runs the shutdown event loop via epoll + signalfd.

#include <sys/epoll.h>
#include <sys/signalfd.h>

#include <csignal>
#include <thread>

#include "aligned_vec.h"
#include "avx2_kernel.h"
#include "cache_service.h"
#include "constants.h"
#include "inference_model.h"
#include "level1_ivf.h"
#include "memory_arena.h"
#include "vector_index.h"

namespace {

// File descriptor index of the Death Pipe reader end.
// The pipe is created by the Control plane and inherited by this process
// on spawn. EOF on this signals that this process must shut down.
inline constexpr uint32_t kPipeReaderFD = 3;

// Vector Engine's graceful shutdown timeout in second(s).
inline constexpr uint32_t kShutdownTimeout = 5;

// `WarmupEngine()` drives the ONNX runtime and AVX2 dispatch through one full
// execution path before the server starts accepting requests, eliminating JIT
// initialization and cold-cache latency from the first real request.
void WarmupEngine(const Embedder& embedder) {
    std::cout << "[Vector Engine] Warming up ONNX runtime...\n";

    constexpr uint32_t kWarmupRounds = 3;
    const std::string  dummy_prompt  = "Hello, World!";

    AlignedVec dummy_vec;
    for (uint32_t i = 0; i < kWarmupRounds; ++i) {
        auto result = embedder.Encode(dummy_prompt);
        if (result.ok()) {
            dummy_vec = std::move(result.value());
        }
    }

    // Warm-up AVX2 dispatch with a zero-initialized batch.
    auto dummy_batch = CreateAlignedVector(4 * kVectorDim);
    std::memset(dummy_batch.get(), 0, 4 * kVectorMemsize);

    float scores[kBatchSize] = {};
    if (dummy_vec) {
        DotProductBatch(dummy_vec.get(), dummy_batch.get(), scores);
    }

    std::cout << "[Vector Engine] Warm-up completed.\n";
}

// `RunServer()` starts the gRPC server and GC background thread, then blocks on
// the epoll event loop until a shutdown signal arrives (Death Pipe EOF or
// SIGINT/SIGTERM).
// `fd_sig` must be a `signalfd` created by `main()` after the signal mask is
// set.
void RunServer(const Embedder& embedder, VectorIndex& index, MemoryArena& arena,
               std::atomic<bool>& g_shutdown_req, int fd_sig) {
    const std::string server_address{"unix:///tmp/strix.sock"};
    unlink("/tmp/strix.sock");

    CacheServiceImpl service(embedder, index);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    const std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
        throw std::runtime_error("Failed to start gRPC server");
    }

    std::cout << "[Vector Engine] Listening on " << server_address << "\n";

    std::thread gc_thread(&MemoryArena::RunGarbageCollector, &arena,
                          std::ref(g_shutdown_req));
    std::thread coord_thread(&VectorIndex::RunCoordinator, &index,
                             std::ref(g_shutdown_req));
    std::thread grpc_thread([&]() {
        try {
            server->Wait();
        } catch (const std::exception& e) {
            std::cerr << "[Vector Engine] FATAL: gRPC crashed: " << e.what()
                      << "\n";
            g_shutdown_req.store(true, std::memory_order_release);
        } catch (...) {
            std::cerr
                << "[Vector Engine] FATAL: gRPC crashed with unknown error.\n";
            g_shutdown_req.store(true, std::memory_order_release);
        }
    });

    // Create epoll FD on Death pipe and signal FD.
    const int epfd = epoll_create1(EPOLL_CLOEXEC);
    if (epfd == -1) {
        close(fd_sig);
        throw std::runtime_error("Failed invoking epoll_create1()");
    }

    auto epoll_add = [&](const int fd) {
        epoll_event ev{};
        ev.events  = EPOLLIN;
        ev.data.fd = fd;
        if (epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev) == -1) {
            close(fd_sig);
            close(epfd);
            throw std::runtime_error("Failed invoking epoll_ctl()");
        }
    };

    epoll_add(kPipeReaderFD);
    epoll_add(fd_sig);

    // Block until one of the two shutdown sources fires.
    epoll_event fired{};
    while (true) {
        const int n = epoll_wait(epfd, &fired, 1, -1);
        if (n == -1) {
            // Spurious wakeup, retry.
            if (errno == EINTR) {
                continue;
            }

            break;
        }

        if (fired.data.fd == kPipeReaderFD) {
            std::cout
                << "[Vector Engine] Death Pipe EOF. Initiating shutdown...\n";
        } else {
            signalfd_siginfo si{};
            read(fd_sig, &si, sizeof(si));
            std::cout << "[Vector Engine] Signal " << si.ssi_signo
                      << " received. Initiating shutdown...\n";
        }

        break;
    }

    close(epfd);
    close(fd_sig);

    // Graceful shutdown: close all subsystems, then join threads.
    g_shutdown_req.store(true, std::memory_order_release);

    const auto deadline = std::chrono::system_clock::now() +
                          std::chrono::seconds(kShutdownTimeout);
    server->Shutdown(deadline);

    if (grpc_thread.joinable()) {
        grpc_thread.join();
    }

    if (gc_thread.joinable()) {
        gc_thread.join();
    }

    if (coord_thread.joinable()) {
        coord_thread.join();
    }
}

}  // namespace

int main() {
    const char* tok_path  = std::getenv("TOKENIZER_PATH");
    const char* bert_path = std::getenv("TRANSFORMER_PATH");

    if (tok_path == nullptr || bert_path == nullptr) {
        std::cerr
            << "[Vector Engine] FATAL: TOKENIZER_PATH or TRANSFORMER_PATH "
               "environment variable is not set.\n";
        return 1;
    }

    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);

    // pthread_sigmask affects only the calling thread; every thread spawned
    // afterwards inherits the mask. Masking here ensures that gRPC's internal
    // thread pool, ONNX execution-provider threads, and our own threads never
    // receive SIGINT or SIGTERM - only the signalfd delivers them.
    pthread_sigmask(SIG_BLOCK, &mask, nullptr);

    // signalfd must be created on the same thread that set the mask and
    // before any other thread exists.
    const int fd_sig = signalfd(-1, &mask, SFD_NONBLOCK | SFD_CLOEXEC);
    if (fd_sig == -1) {
        std::cerr << "[Vector Engine] FATAL: Failed invoking signalfd().\n";
        return 1;
    }

    try {
        const Embedder embedder(tok_path, bert_path);
        std::cout << "[Vector Engine] Initialized Inference Model.\n";

        const auto arena =
            std::make_unique<MemoryArena>(ArenaConfig::Production());
        std::cout << "[Vector Engine] Initialized Memory Arena.\n";

        VectorIndex indexer(*arena, kL0Capacity, IvfConfig::Production());
        std::cout << "[Vector Engine] Initialized Vector Index.\n";

        arena->SetNodeFreedCallback(
            [&indexer](uint32_t node_id) { indexer.ReleaseNode(node_id); });

        WarmupEngine(embedder);

        std::cout << "[Vector Engine] Opening to gRPC...\n";
        std::atomic<bool> g_shutdown_req{false};
        RunServer(embedder, indexer, *arena, g_shutdown_req, fd_sig);
        std::cout << "[Vector Engine] Closing...\n";

    } catch (const std::exception& e) {
        std::cerr << "[Vector Engine] FATAL: " << e.what() << "\n";
        close(fd_sig);
        return 1;
    }

    close(fd_sig);
    return 0;
}
