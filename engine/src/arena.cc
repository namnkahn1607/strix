//
// Created by nlnk on Apr 16, 26.
//

#include "arena.hh"

#include <cstring>
#include <iostream>
#include <thread>

#include "constant.hh"

MemoryArena::MemoryArena() : write_head(0), read_tail(0) {
    // Allocating metadata Node array
    metadata = new MetaNode[engine::TOTAL_MAX_SLOTS];

    // Allocating Vector array
    vectors = static_cast<float*>(std::aligned_alloc(
        32, engine::TOTAL_MAX_SLOTS * engine::VECTOR_MEMSIZE));

    // Allocating Ring Buffer payload array
    buffer_payload = static_cast<uint8_t*>(
        std::aligned_alloc(32, engine::PAYLOAD_BUFFER_SIZE));

    // Warming up by touching all pages to avoid Page Faults at runtime
    std::memset(vectors, 0, engine::TOTAL_MAX_SLOTS * engine::VECTOR_MEMSIZE);
    std::memset(buffer_payload, 0, engine::PAYLOAD_BUFFER_SIZE);

    for (size_t i = 0; i < engine::TOTAL_MAX_SLOTS; ++i) {
        metadata[i].created_at.store(0, std::memory_order_relaxed);
        metadata[i].control_block.store(0, std::memory_order_relaxed);
    }

    std::cout << "[Vector Engine] Initialized Dual Memory Arena" << std::endl;
}

MemoryArena::~MemoryArena() {
    free(vectors);
    free(buffer_payload);
    delete[] metadata;
}

void MemoryArena::RunGarbageCollector(
    const std::atomic<bool>& g_shutdown_request) {
    try {
        while (!g_shutdown_request.load(std::memory_order_relaxed)) {
            const uint64_t head = write_head.load(std::memory_order_relaxed);
            const uint64_t tail = read_tail.load(std::memory_order_relaxed);

            if (const uint64_t used_space = head - tail;
                used_space < engine::LOW_WATERMARK_THRESHOLD) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(engine::LOW_GC_SLEEP_MS));
                continue;
            } else if (used_space < engine::HIGH_WATERMARK_THRESHOLD) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(engine::HIGH_GC_SLEEP_MS));
            }

            /* The Snowplow mechanism */
            const uint64_t tail_index =
                tail & (engine::PAYLOAD_BUFFER_SIZE - 1);

            // Perform leaping in case approaching buffer border
            if (const uint64_t padding =
                    engine::PAYLOAD_BUFFER_SIZE - tail_index;
                padding < sizeof(PayloadHeader)) {
                read_tail.fetch_add(padding, std::memory_order_relaxed);
                continue;
            }

            // Read payload header
            const auto* header =
                reinterpret_cast<PayloadHeader*>(buffer_payload + tail_index);

            // Approach an invalid header identifier, advance tail by 1.
            if (header->identifier != engine::VALID_IDENTIFIER) {
                read_tail.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            const uint32_t node_id = header->node_id;
            const uint32_t text_len = header->length;
            const uint32_t total_size = sizeof(PayloadHeader) + text_len;
            MetaNode& node = metadata[node_id];
            auto [state, ref_bit, length, v_offset] = node.LoadControl();

            if (state == NodeState::DEAD ||
                v_offset != (tail & engine::VIRTUAL_OFFSET_MASK)) {
                read_tail.fetch_add(total_size, std::memory_order_relaxed);
                continue;
            }

            if (ref_bit == EvictState::COLD) {
                // Evict in case encountering cold node
                uint64_t expected_ctrl =
                    PackControl(state, ref_bit, length, v_offset);
                const uint64_t desired_ctrl =
                    PackControl(NodeState::DEAD, ref_bit, length, v_offset);

                node.control_block.compare_exchange_strong(
                    expected_ctrl, desired_ctrl, std::memory_order_release,
                    std::memory_order_relaxed);

                node.created_at.store(0, std::memory_order_relaxed);
            } else {
                // Perform payload rescue & give it a Second chance
                // if the node is still hot.
                const uint64_t rescued_offset = AllocatePayload(text_len);
                const uint64_t rescued_index =
                    rescued_offset & (engine::PAYLOAD_BUFFER_SIZE - 1);

                PayloadHeader new_header{engine::VALID_IDENTIFIER, node_id,
                                         text_len};
                std::memcpy(buffer_payload + rescued_index, &new_header,
                            sizeof(PayloadHeader));

                uint64_t src_idx = (tail + sizeof(PayloadHeader)) &
                                   (engine::PAYLOAD_BUFFER_SIZE - 1);
                uint64_t dst_idx = (rescued_offset + sizeof(PayloadHeader)) &
                                   (engine::PAYLOAD_BUFFER_SIZE - 1);
                uint64_t bytes_left = text_len;

                while (bytes_left > 0) {
                    uint64_t src_continuous =
                        engine::PAYLOAD_BUFFER_SIZE - src_idx;
                    uint64_t dst_continuous =
                        engine::PAYLOAD_BUFFER_SIZE - dst_idx;

                    const uint64_t chunk_size =
                        std::min({bytes_left, src_continuous, dst_continuous});

                    std::memcpy(buffer_payload + dst_idx,
                                buffer_payload + src_idx, chunk_size);

                    bytes_left -= chunk_size;
                    src_idx = (src_idx + chunk_size) &
                              (engine::PAYLOAD_BUFFER_SIZE - 1);
                    dst_idx = (dst_idx + chunk_size) &
                              (engine::PAYLOAD_BUFFER_SIZE - 1);
                }

                uint64_t expected_ctrl =
                    PackControl(state, EvictState::HOT, length, v_offset);
                const uint64_t desired_ctrl = PackControl(
                    state, EvictState::COLD, length, rescued_offset);

                node.control_block.compare_exchange_strong(
                    expected_ctrl, desired_ctrl, std::memory_order_release,
                    std::memory_order_relaxed);
            }

            // Move read tail forward anyway
            read_tail.fetch_add(total_size, std::memory_order_relaxed);
        }
    } catch (const std::exception& e) {
        std::cout << "[Engine] WARNING: " << e.what() << std::endl;
    }
}

uint64_t MemoryArena::AllocatePayload(const uint32_t length) {
    const size_t total_size = sizeof(PayloadHeader) + length;
    uint64_t curr_write = write_head.load(std::memory_order_relaxed);
    uint64_t allocated_offset;

    while (true) {
        // Backpressure mechanism: System memory resource exhausted
        if (curr_write + total_size -
                read_tail.load(std::memory_order_relaxed) >=
            engine::PAYLOAD_BUFFER_SIZE) {
            throw std::runtime_error("Resource Exhausted");
        }

        const uint64_t actual_index =
            curr_write & (engine::PAYLOAD_BUFFER_SIZE - 1);
        uint64_t padding = 0;

        // Wrap-around Payload Header protection
        if (engine::PAYLOAD_BUFFER_SIZE - actual_index <
            sizeof(PayloadHeader)) {
            padding = engine::PAYLOAD_BUFFER_SIZE - actual_index;
        }

        allocated_offset = curr_write + padding;

        if (const uint64_t next_write = allocated_offset + total_size;
            write_head.compare_exchange_weak(curr_write, next_write,
                                             std::memory_order_relaxed)) {
            break;
        }
    }

    return allocated_offset;
}

void MemoryArena::ReadPayload(const uint64_t v_offset, const uint32_t length,
                              std::string* out_payload) const {
    if (length == 0) {
        out_payload->clear();
        return;
    }

    out_payload->resize(length);
    const uint64_t text_index =
        (v_offset + sizeof(PayloadHeader)) & (engine::PAYLOAD_BUFFER_SIZE - 1);
    char* destination = out_payload->data();

    if (engine::PAYLOAD_BUFFER_SIZE - text_index >= length) {
        std::memcpy(destination, buffer_payload + text_index, length);
    } else {
        const size_t chunk1_size = engine::PAYLOAD_BUFFER_SIZE - text_index;
        const size_t chunk2_size = length - chunk1_size;
        std::memcpy(destination, buffer_payload + text_index, chunk1_size);
        std::memcpy(destination + chunk1_size, buffer_payload, chunk2_size);
    }
}

uint64_t MemoryArena::WritePayload(const uint32_t node_id,
                                   const uint8_t* in_payload,
                                   const uint32_t length) {
    const uint64_t header_offset = AllocatePayload(length);
    const uint64_t header_index =
        header_offset & (engine::PAYLOAD_BUFFER_SIZE - 1);

    // Create and write payload header
    const PayloadHeader header{engine::VALID_IDENTIFIER, node_id, length};
    std::memcpy(buffer_payload + header_index, &header, sizeof(PayloadHeader));

    // Now write the payload text
    const uint64_t text_index = (header_index + sizeof(PayloadHeader)) &
                                (engine::PAYLOAD_BUFFER_SIZE - 1);

    if (engine::PAYLOAD_BUFFER_SIZE - text_index >= length) {
        std::memcpy(buffer_payload + text_index, in_payload, length);
    } else {
        const size_t chunk1_size = engine::PAYLOAD_BUFFER_SIZE - text_index;
        const size_t chunk2_size = length - chunk1_size;
        std::memcpy(buffer_payload + text_index, in_payload, chunk1_size);
        std::memcpy(buffer_payload, in_payload + chunk1_size, chunk2_size);
    }

    return header_offset;
}
