// Per-thread worker identity implementation.

#include "worker/identity.h"

#include <absl/log/check.h>

#include "base/tagged_treiber.h"

namespace strix::worker {

namespace {

TreiberStack& Stack() {
    static TreiberStack instance(kNumRPCWorkers);
    return instance;
}

// RPC worker's identity within its lifecycle.
// Use with `thread_local` to enable its default automatic behavior upon
// thread construction and destruction.
class Slot {
public:
    Slot() = default;
    ~Slot() {
        if (id != TreiberStack::kEmpty) {
            Stack().Push(id);
            id = TreiberStack::kEmpty;
        }
    }

    Slot(const Slot&)            = delete;
    Slot& operator=(const Slot&) = delete;
    Slot(Slot&&)                 = delete;
    Slot& operator=(Slot&&)      = delete;

    uint32_t id = TreiberStack::kEmpty;
};

}  // namespace

// Single-most `WorkerSlot` instance owned by THIS thread.
thread_local Slot tls_slot{};

void Register() noexcept {
    if (tls_slot.id == TreiberStack::kEmpty) {
        const auto wid = Stack().Pop();
        CHECK(wid != TreiberStack::kEmpty)
            << "Out of worker ID. Failed expectations on gRPC thread model.";
        tls_slot.id = wid;
    }
}

uint32_t ThreadID() noexcept {
    CHECK(tls_slot.id != TreiberStack::kEmpty)
        << "Damn! Worker ID was released before thread destruction.";
    return tls_slot.id;
}

}  // namespace strix::worker
