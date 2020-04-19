//
// Created by ishwark on 11/04/20.
// Copyright 2020 Ishwar Kulkarni.
// Subject to GPL License(www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
//


#ifndef WORD_EMBEDDING_UTILS_HXX
#define WORD_EMBEDDING_UTILS_HXX

#include <memory>
#include <numeric>
#include <cmath>
#include <chrono>

namespace Utils {
    // Span: [ptr, ptr + len).
    template<typename T>
    struct Span {
        Span(T *data, size_t len) : ptr(data), len(len) {}

        Span(const T *start, T *end) : ptr(start), len(end - start) {}

        explicit Span(size_t len)
                : storage(new T[len]), ptr(storage.get()), len(len) {}

        template<typename U>
        void deepCopy(const Span<U> &other) {
#ifndef NDEBUG
            if (other.size() != this->size())
              throw std::runtime_error("Span lengths different, cannot be copied.");
#endif
            for (size_t i = 0; i < Span<T>::len; ++i)
                Span<T>::ptr[i] = other[i];
        }

        inline T &operator[](size_t i) {
#ifndef NDEBUG
            if (i > len)
              throw std::runtime_error("Invalid access");
#endif
            return ptr[i];
        }

        inline Span<T> &operator=(const Span<T> &other) {
#ifndef NDEBUG
            if (other.size() != this->size())
              throw std::runtime_error("Span lengths different, cannot be copied.");
#endif
            if (other.ptr == this->ptr)
                return *this;
            if (storage.get())
                this->deepCopy(other);
            else {
                ptr = other.ptr;
            }
            return *this;
        }

        Span<T>(const Span<T> &other) :
                ptr(other.ptr),
                len(other.len) {}

        [[nodiscard]] inline size_t size() const { return len; }

        inline const T &operator[](size_t i) const {
#ifndef NDEBUG
            if (i > len)
              throw std::runtime_error("Invalid access");
#endif
            return ptr[i];
        }

        inline T *begin() { return ptr; }

        inline T *end() { return ptr + len; }

        inline const T *begin() const { return ptr; }

        inline const T *end() const { return ptr + len; }

        inline void fill(const T &v) { std::fill(ptr, ptr + len, v); }

        inline T mag() {
            return std::sqrt(std::inner_product(begin(), end(), begin(), T(0)));
        }

    protected:
        std::unique_ptr<T[]> storage = nullptr;
        T *ptr = nullptr;
        const size_t len = 0;
    };

    template<typename T>
    Span<T> &operator+=(Span<T> &a, const Span<T> &b) {
        for (size_t i = 0; i < b.size(); ++i)
            a[i] += b[i];

        return a;
    }

    template<typename T>
    Span<T> &operator-=(Span<T> &a, const Span<T> &b) {
        for (size_t i = 0; i < b.size(); ++i)
            a[i] -= b[i];
        return a;
    }

    template<typename T>
    T operator*(const Span<T> &a, const Span<T> &b) {
        T r = 0;
        return std::inner_product(a.begin(), a.end(), b.begin(), r);
    }

    template<typename T, typename S>
    void operator*=(Span<T> &a, const S &s) {
        for (auto &i : a)
            i *= s;
    }

    template<typename T, typename S>
    void operator/=(Span<T> &a, const S &s) {
        for (auto &i : a)
            i /= s;
    }

    using FloatSpan = Utils::Span<float>;

    inline float Sigmoid(float v) { return 1.f / (1.f + expf(-v)); }

    inline size_t rseed() {
        size_t seed = 42;
#ifdef NDEBUG
        seed = time(nullptr);
#endif
        return seed;
    }
}

#endif // WORD_EMBEDDING_UTILS_HXX
