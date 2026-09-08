// Buffer pool. Provides data buffering for payload reader
// without extra allocation.

#pragma once

#include <absl/strings/cord.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "base/tagged_treiber.h"
#include "memory/control_block.h"

namespace strix::memory {

class BufPool {
public:
    explicit BufPool(uint32_t capacity) : stack_{capacity} {
        buffers_.reserve(capacity);
        for (uint32_t i = 0; i < capacity; ++i) {
            buffers_.push_back(
                std::make_unique_for_overwrite<uint8_t[]>(kMaxPayloadLength)
            );
        }
    };

    BufPool(const BufPool&)           = delete;
    BufPool& operator=(const BufPool) = delete;
    BufPool(BufPool&&)                = delete;
    BufPool& operator=(BufPool&&)     = delete;

    // Acquires a raw buffer of `kMaxPayloadLen` bytes. Returns `nullptr` only
    // if the pool is exhausted.
    // On success, writes the slot identifier to `*id` - caller `Wrap()` once
    // after filling data into the buffer.
    uint8_t* Acquire(uint32_t* id) noexcept {
        const auto got = stack_.Pop();
        if (got == TreiberStack::kEmpty) {
            return nullptr;
        }

        *id = got;
        return buffers_[got].get();
    }

    // Wraps a buffer previously returned by `Acquire()` into an `absl::Cord`.
    // The buffer is released back to pool once the Cord's `ref_count` reaches
    // zero. `*buf` and `id` must remain untouched afterwards.
    absl::Cord Wrap(uint8_t* buf, uint32_t id, uint32_t length) noexcept {
        return absl::MakeCordFromExternal(
            absl::string_view{reinterpret_cast<const char*>(buf), length},
            [this, id]() noexcept { this->stack_.Push(id); }
        );
    }

private:
    TreiberStack stack_;

    // Backing array containing discrete buffer pointers.
    std::vector<std::unique_ptr<uint8_t[]>> buffers_;
};

}  // namespace strix::memory
