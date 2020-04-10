//
// Created by ishwark on 08/04/20.
//

#ifndef WORD_EMBEDDING_UTILS_HXX
#define WORD_EMBEDDING_UTILS_HXX

#include <memory>
#include <random>

namespace Utils {
// Span: [ptr, ptr + len).
    template<typename T>
    struct Span {
        Span(T *data, size_t len) : ptr(data), len(len) {}

        Span(const T *start, T *end) : ptr(start), len(end - start) {}

        explicit Span(size_t len)
                : storage(new T[len]), ptr(storage.get()), len(len) {}

        Span() = default;

        void copyFrom(const Span<T> &other)
        {
#ifndef NDEBUG
            if (other.size() != this->size())
              throw std::runtime_error("Span lengths different, cannot be assigned.");
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

        [[nodiscard]] inline size_t size() const { return len; }

        inline const T &operator[](size_t i) const { return ptr[i]; }

        inline T *begin() { return ptr; }

        inline T *end() { return ptr + len; }

        inline const T *begin() const { return ptr; }

        inline const T *end() const { return ptr + len; }

        inline void fill(const T &v) { std::fill(ptr, ptr + len, v); }

        inline T mag() {
            return sqrt(std::inner_product(begin(), end(), begin(), 0));
        }

    protected:
        std::unique_ptr<T[]> storage = nullptr;
        T *ptr = nullptr;
        size_t len = 0;
    };

    template<typename T>
    inline Span<T> make_span(T *data, size_t len) {
        return Span<T>(data, len);
    }

    template<typename T>
    inline Span<T> make_span(T *start, T *end) {
        return Span<T>(start, end);
    }

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
}

#endif // WORD_EMBEDDING_UTILS_HXX
