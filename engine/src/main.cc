#include <grpcpp/grpcpp.h>

#include <csignal>
#include <thread>

#include "arena.hh"
#include "avx_math.hh"
#include "constant.hh"
#include "embedder.hh"
#include "service.hh"

constexpr int32_t PIPE_READER_FD = 3;

std::atomic g_shutdown_requested{false};

static void WarmUpEngine(const Embedder& embedder) {
    std::cout << "[Engine] Warming up ONNX runtime..." << std::endl;

    constexpr int32_t WARM_UP_ROUNDS = 3;
    const std::string dummy_prompt = "Hello, World";

    AlignedVector dummy_vec;
    for (int32_t i = 0; i < WARM_UP_ROUNDS; ++i) {
        dummy_vec = embedder.Encode(dummy_prompt);
    }

    const auto* dummy_batch =
        static_cast<float*>(std::aligned_alloc(32, 4 * engine::VECTOR_MEMSIZE));
    float scores[4];

    if (dummy_vec) {
        CosineL0_Batch4(dummy_vec.get(), dummy_batch, scores);
    }

    std::cout << "[Engine] Warm-up completed." << std::endl;
}

void RunServer(const Embedder& embedder, MemoryArena& arena) {
    const std::string server_address{"unix:///tmp/strix.sock"};
    const auto socket_directory{"/tmp/strix.sock"};

    // Clear out old socket file from previous process run
    // before binding into new one.
    unlink(socket_directory);

    // Service is only allowed to reference for reading and writing data.
    CacheServiceImpl service(embedder, arena);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(
        server_address,
        grpc::InsecureServerCredentials());  // no TLS encryption
    builder.RegisterService(&service);

    // Force UNIX to create a physical file (socket) at server_address
    // and bind() C++ process to it.
    const std::unique_ptr server(builder.BuildAndStart());

    // Run Snowplow garbage collector on a background thread.
    std::thread gc_thread(&MemoryArena::RunGarbageCollector, &arena,
                          std::ref(g_shutdown_requested));

    // Call Wait() on another thread to avoid blocking Main Thread.
    std::thread grpc_thread([&]() {
        try {
            server->Wait();
        } catch (const std::exception& e) {
            std::cerr << "[Engine] FATAL: gRPC crashed unexpectedly: "
                      << e.what() << std::endl;
            g_shutdown_requested.store(true, std::memory_order_release);
        } catch (...) {
            std::cerr << "[Engine] FATAL: gRPC crashed with unknown error."
                      << std::endl;
            g_shutdown_requested.store(true, std::memory_order_release);
        }
    });

    char buffer;
    if (const size_t bytes_read = read(PIPE_READER_FD, &buffer, 1);
        bytes_read == 0) {
        std::cout
            << "[Engine] HTTP Gateway detached (EOF). Initiating Shutdown..."
            << std::endl;
    } else {
        std::cout
            << "[Engine] POSIX pipe error/Interrupt. Initiating Shutdown..."
            << std::endl;
    }

    g_shutdown_requested.store(true, std::memory_order_release);

    const auto deadline = std::chrono::system_clock::now() +
                          std::chrono::seconds(engine::G_SHUTDOWN_TIMEOUT);
    // Shutdown() will stop receiving gRPC requests on calling,
    // and close the server once deadline is met.
    server->Shutdown(deadline);

    // Main Thread waits for all workers to finish before closing.
    if (grpc_thread.joinable()) {
        grpc_thread.join();
    }

    if (gc_thread.joinable()) {
        gc_thread.join();
    }
}

int main() {
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &mask, nullptr);

    std::thread sig_thread([&mask]() {
        int sig;
        sigwait(&mask, &sig);
        g_shutdown_requested.store(true, std::memory_order_release);
    });

    const char* tok_path{std::getenv("TOKENIZER_PATH")};
    const char* bert_path{std::getenv("TRANSFORMER_PATH")};
    if (tok_path == nullptr || bert_path == nullptr) {
        throw std::runtime_error(
            "Env-var TOKENIZER_PATH or TRANSFORMER_PATH is not set");
    }

    const Embedder embedder(tok_path, bert_path);

    // Main Thread is responsible for construct & deconstruct Memory Arena.
    const auto memory_arena = std::make_unique<MemoryArena>();

    WarmUpEngine(embedder);

    std::cout << "[Vector Engine] Opening to gRPC..." << std::endl;
    RunServer(embedder, *memory_arena);
    std::cout << "[Vector Engine] Closing..." << std::endl;

    return 0;
}
