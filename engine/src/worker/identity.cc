// Per-thread worker identity implementation.

#include "worker/identity.h"

#include <cstdlib>
#include <iostream>

#include "common/constants.h"
#include "common/tagged_treiber.h"

#define ABORT(msg)                                                      \
    do {                                                                \
        std::cerr << "[FATAL] " << __FILE__ << ":" << __LINE__ << " - " \
                  << (msg) << '\n';                                     \
        std::abort();                                                   \
    } while (false);

namespace {

// Return the singleton worker ID Freelist.
TreiberStack& Stack() {
    static TreiberStack instance(kNumRPCWorkers);
    return instance;
}

// WorkerSlot represents a RPC worker's identity within its lifecycle.
//
// Ownership: Use with `thread_local` to enable its default automatic behavior
// upon thread construction/destruction. Cannot copy/move/assign.
class WorkerSlot {
public:
    WorkerSlot() = default;
    ~WorkerSlot() {
        if (id != TreiberStack::kEmpty) {
            Stack().Push(id);
        }
    }

    WorkerSlot(const WorkerSlot&)            = delete;
    WorkerSlot& operator=(const WorkerSlot&) = delete;
    WorkerSlot(WorkerSlot&&)                 = delete;
    WorkerSlot& operator=(WorkerSlot&&)      = delete;

    uint32_t id = TreiberStack::kEmpty;
};

}  // namespace

// Single-most `WorkerSlot` instance owned by THIS thread.
thread_local WorkerSlot tls_slot{};

void RegisterWorker() noexcept {
    if (tls_slot.id == TreiberStack::kEmpty) {
        const auto wid = Stack().Pop();
        if (wid == TreiberStack::kEmpty) {
            ABORT("Out of worker ID!");
        }

        tls_slot.id = wid;
    }
}

uint32_t WorkerID() noexcept {
    if (tls_slot.id == TreiberStack::kEmpty) {
        ABORT("Unregisterd worker!");
    }

    return tls_slot.id;
}
