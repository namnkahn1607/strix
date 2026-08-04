// A lightweight backport of std::expected<T, E> - a C++23 feature.

#pragma once

#include <utility>
#include <variant>

// Expected<T, E> encodes success `T` or a predictable, non-fatal failure `E`
// without exceptions.
//
// Use `Expected<T, E>` for errors that callers are expected to handle at the
// call site (e.g. validation failures), not unrecoverable ones.
template <typename T, typename E>
struct Expected {
    std::variant<T, E> data;

    bool ok() const noexcept { return std::holds_alternative<T>(data); }

    T&       value() { return std::get<T>(data); }
    const T& value() const { return std::get<T>(data); }

    E&       error() { return std::get<E>(data); }
    const E& error() const { return std::get<E>(data); }

    static Expected Ok(T val) {
        return Expected{
            std::variant<T, E>{std::in_place_index<0>, std::move(val)}
        };
    }

    static Expected Err(E err) {
        return Expected{
            std::variant<T, E>{std::in_place_index<1>, std::move(err)}
        };
    }
};
