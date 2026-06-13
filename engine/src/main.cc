//
// main.cc
//
// Orchestrator: initializes all subsystems, injects dependencies,
// and runs the shutdown event loop via epoll + signalfd.
//

#include <sys/epoll.h>
#include <sys/signalfd.h>

#include <csignal>
#include <thread>

#include "aligned_vec.hh"
#include "arena.hh"
#include "avx2_math.hh"
#include "constants.hh"
#include "inference.hh"
#include "service.hh"

namespace {

// Death Pipe reader File Descriptor's index
inline constexpr int32_t PIPE_READER_FD = 3;

// Vector Engine's shutdown timeout in second(s)
inline constexpr uint32_t SHUTDOWN_TIMEOUT = 5;

// --- WarmUpEngine ---
// Drives ONNX runtime and AVX2 dispatch through one full execution path
// before the server starts accepting requests.
void WarmupEngine(const Embedder& embedder) {
    std::cout << "[Vector Engine] Warming up ONNX runtime...\n";

    constexpr int32_t WARMUP_ROUNDS = 3;
    const std::string dummy_prompt = "Hello, World!";

    AlignedVec dummy_vec;
    for (int32_t i = 0; i < WARMUP_ROUNDS; ++i) {
        auto result = embedder.Encode(dummy_prompt);
        if (result.ok()) {
            dummy_vec = std::move(result.value());
        }
    }

    // Warmup AVX2 dispatch with a zero-initialized batch
    auto dummy_batch = CreateAlignedVector(4 * VECTOR_DIM);
    std::memset(dummy_batch.get(), 0, 4 * VECTOR_MEMSIZE);

    float scores[BATCH_SIZE] = {};
    if (dummy_vec) {
        DotProductL0_Batch4(dummy_vec.get(), dummy_batch.get(), scores);
    }

    std::cout << "[Vector Engine] Warm-up completed.\n";
}

void RunServer(const Embedder& embedder, MemoryArena& arena,
               std::atomic<bool>& g_shutdown_req) {
    // --- gRPC setup ---
    const std::string server_address{"unix:///tmp/strix.sock"};
    unlink("/tmp/strix.sock");

    CacheServiceImpl service(embedder, arena);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    const std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
        throw std::runtime_error("Failed to start gRPC server");
    }

    std::cout << "[Vector Engine] Listening on " << server_address << "\n";

    // --- Background GC Thread ---
    std::thread gc_thread(&MemoryArena::RunGarbageCollector, &arena,
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

    // --- SIGINT / SIGTERM handler ---
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);

    // Block signals to all threads so only Signal FD delivers them
    pthread_sigmask(SIG_BLOCK, &mask, nullptr);

    const int fd_sig = signalfd(-1, &mask, SFD_NONBLOCK | SFD_CLOEXEC);
    if (fd_sig == -1) {
        throw std::runtime_error("Failed invoking signalfd()");
    }

    // --- Use epoll to watch Pipe Reader & Signal FD ---
    const int epfd = epoll_create1(EPOLL_CLOEXEC);
    if (epfd == -1) {
        close(fd_sig);
        throw std::runtime_error("Failed invoking epoll_create1()");
    }

    auto epoll_add = [&](const int fd) {
        epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.fd = fd;
        if (epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev) == -1) {
            close(fd_sig);
            close(epfd);
            throw std::runtime_error("Failed invoking epoll_ctl()");
        }
    };

    epoll_add(PIPE_READER_FD);
    epoll_add(fd_sig);

    // --- Event loop ---
    // Blocks until one of the two shutdown source fires
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

        if (fired.data.fd == PIPE_READER_FD) {
            // Death Pipe Reader fired.
            std::cout
                << "[Vector Engine] Death Pipe EOF. Initiating shutdown...\n";
        } else {
            // Signal FD fired -> Drain it to get signal info.
            signalfd_siginfo si{};
            read(fd_sig, &si, sizeof(si));
            std::cout << "[Vector Engine] Signal " << si.ssi_signo
                      << " received. Initiating shutdown...\n";
        }

        break;
    }

    close(epfd);
    close(fd_sig);

    // --- Graceful shutdown ---
    g_shutdown_req.store(true, std::memory_order_release);

    const auto deadline = std::chrono::system_clock::now() +
                          std::chrono::seconds(SHUTDOWN_TIMEOUT);
    server->Shutdown(deadline);

    if (grpc_thread.joinable()) {
        grpc_thread.join();
    }

    if (gc_thread.joinable()) {
        gc_thread.join();
    }
}

};  // namespace

// ---------------------------------------------------------------------------
// Vector Engine Entry Point
// ---------------------------------------------------------------------------

int main() {
    const char* tok_path = std::getenv("TOKENIZER_PATH");
    const char* bert_path = std::getenv("TRANSFORMER_PATH");

    if (tok_path == nullptr || bert_path == nullptr) {
        std::cerr
            << "[Vector Engine] FATAL: TOKENIZER_PATH or TRANSFORMER_PATH "
               "environment variable is not set.\n";
        return 1;
    }

    try {
        const Embedder embedder(tok_path, bert_path);
        const auto     memory_arena =
            std::make_unique<MemoryArena>(ArenaConfig::Production());
        std::atomic<bool> g_shutdown_req{false};

        WarmupEngine(embedder);

        std::cout << "[Vector Engine] Opening to gRPC...\n";
        RunServer(embedder, *memory_arena, g_shutdown_req);
        std::cout << "[Vector Engine] Closing...\n";

    } catch (const std::exception& e) {
        std::cerr << "[Vector Engine] FATAL: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
