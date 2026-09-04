// Data plane orchestrator: initializes all subsystems, injects dependencies,
// and runs the shutdown event loop via epoll + signalfd.

#include <grpcpp/resource_quota.h>
#include <grpcpp/server_builder.h>
#include <sys/epoll.h>
#include <sys/signalfd.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <memory>
#include <thread>

#include "collection/collection.h"
#include "collection/config.h"
#include "inference/sentence_encoder.h"
#include "inference/simd_float_buf.h"
#include "memory/arena.h"
#include "memory/config.h"
#include "rpc/cache_service.h"
#include "worker/identity.h"

using namespace strix;

namespace {

const char* kServerAddress = "unix:///tmp/strix.sock";
const char* kSockerPath    = "/tmp/strix.sock";

// File descriptor of the Death Pipe read-end.
// Inherited by this process on spawn; an EOF signals shutdown.
constexpr uint32_t kPipeReaderFD = 3;

// Graceful shutdown timeout.
constexpr std::chrono::seconds kShutdownTimeout{5};

// Drives the ONNX runtime through one full execution path before serving
// prompt requests, eliminating JIT initialization and cold-cache latency.
void WarmupONNX(const inference::SentenceEncoder& encoder) {
    constexpr uint32_t kWarmupRounds = 10;
    const std::string  kDummyPrompt  = "Hello, World!";

    inference::SimdFloatBuf dummy;
    for (uint32_t i = 0; i < kWarmupRounds; ++i) {
        (void)encoder.Encode(kDummyPrompt, dummy);
        asm volatile("" : : "r,m"(dummy) : "memory");
    }
}

void ConfigureServer(grpc::ServerBuilder& builder) {
    grpc::ResourceQuota quota;
    quota.SetMaxThreads(worker::kNumRPCWorkers);

    builder.AddListeningPort(kServerAddress, grpc::InsecureServerCredentials());
    builder.SetResourceQuota(quota);
    builder.SetSyncServerOption(
        grpc::ServerBuilder::SyncServerOption::NUM_CQS, 1
    );
    builder.SetSyncServerOption(
        grpc::ServerBuilder::SyncServerOption::MIN_POLLERS,
        worker::kNumRPCWorkers
    );
    builder.SetSyncServerOption(
        grpc::ServerBuilder::SyncServerOption::MAX_POLLERS,
        worker::kNumRPCWorkers
    );
}

// Starts the gRPC server and GC background thread, then blocks on
// the epoll event loop until a shutdown signal arrives (Death Pipe EOF or
// SIGINT/SIGTERM).
// `fd_sig` must be a `signalfd` created by `main()` after the signal mask is
// set.
void Run(
    const inference::SentenceEncoder& encoder,
    collection::Collection& collector, memory::Arena& arena,
    std::atomic<bool>& shutdown_req, int fd_sig
) {
    unlink(kSockerPath);

    rpc::CacheServiceImpl service(encoder, collector);

    grpc::ServerBuilder builder;
    ConfigureServer(builder);
    builder.RegisterService(&service);

    const std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
        throw std::runtime_error("Failed to start gRPC server");
    }

    std::cout << "[Data-plane] Listening on " << kServerAddress << "\n";

    std::thread gc_thread(
        &memory::Arena::StartGarbageCollector, &arena, std::ref(shutdown_req)
    );
    std::thread coord_thread(
        &collection::Collection::StartCoordinator, &collector,
        std::ref(shutdown_req)
    );
    std::thread grpc_thread([&]() {
        try {
            server->Wait();
        } catch (const std::exception& e) {
            std::cerr << "[Vector Engine] FATAL: gRPC crashed: " << e.what()
                      << "\n";
            shutdown_req.store(true, std::memory_order_release);
        } catch (...) {
            std::cerr
                << "[Vector Engine] FATAL: gRPC crashed with unknown error.\n";
            shutdown_req.store(true, std::memory_order_release);
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
    shutdown_req.store(true, std::memory_order_release);

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
    // NOLINTBEGIN(concurrency-mt-unsafe)
    const char* tok_path  = std::getenv("TOKENIZER_PATH");
    const char* bert_path = std::getenv("TRANSFORMER_PATH");
    // NOLINTEND(concurrency-mt-unsafe)

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
        const inference::SentenceEncoder encoder(tok_path, bert_path);
        std::cout << "[Vector Engine] Initialized Inference Model.\n";

        const auto arena =
            std::make_unique<memory::Arena>(memory::Config::Standard());
        std::cout << "[Vector Engine] Initialized Memory Arena.\n";

        collection::Collection collector(
            collection::Config::Standard(), *arena
        );
        std::cout << "[Vector Engine] Initialized Vector Index.\n";

        arena->SetNodeFreedCallback([&collector](uint32_t node_id) {
            collector.ReleaseSlot(node_id);
        });

        WarmupONNX(encoder);

        std::cout << "[Vector Engine] Opening to gRPC...\n";
        std::atomic<bool> shutdown_req{false};
        Run(encoder, collector, *arena, shutdown_req, fd_sig);
        std::cout << "[Vector Engine] Closing...\n";

    } catch (const std::exception& e) {
        std::cerr << "[Vector Engine] FATAL: " << e.what() << "\n";
        close(fd_sig);
        return 1;
    }

    close(fd_sig);
    return 0;
}
