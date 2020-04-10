//
// Created by ishwark on 11/04/20.
//

#ifndef WORDEMBEDDING_EMBEDDING_HXX
#define WORDEMBEDDING_EMBEDDING_HXX

#include "Corpus.hxx"
#include "Utils.hxx"
#include <algorithm>
#include <cmath>

template<typename T, typename T2 = void>
struct EmbeddingMatrix;

template<typename T>
struct EmbeddingMatrix<
        T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
    EmbeddingMatrix(size_t major, size_t minor)
            : majorSize(major), minorSize(minor), data(new T[majorSize * minorSize])
    {
        for (size_t i = 0; i < majorSize * minorSize; ++i)
            data[i] = getNormalRandom();
    }

    static T getNormalRandom(const size_t *seed = nullptr) {
        static std::default_random_engine generator;

        static std::normal_distribution<T> distribution(0, 1);

        if (seed)
            generator = std::default_random_engine(*seed);

        return distribution(generator);
    }

    size_t findNanColumn()
    {
        auto last = data.get() + majorSize * minorSize;
        auto found = std::find_if(data.get(), last, [](float f) { return std::isnan(f); });
        if (found == last)
            return size_t(-1);
        return (last - data.get()) / majorSize;
    }

    inline Utils::Span<T> operator[](size_t offset) {
#ifndef NDEBUG
        if (offset >= minorSize)
          throw std::runtime_error("Offset too large: " + std::to_string(offset));
#endif
        return Utils::make_span(data.get() + offset * majorSize, majorSize);
    }

    inline const Utils::Span<T>& operator[](size_t offset) const {
        return Utils::make_span(data.get() + offset * majorSize, majorSize);
    }

    const size_t majorSize, minorSize;

private:
    std::unique_ptr<T[]> data;
};

class Embedding {
public:

    using Context = std::vector<size_t>;
    Embedding(Corpus &corpus, size_t K, unsigned pCt, unsigned nCt);

    void train(float eta);

    virtual void updateOutputMatrix(size_t target, const Context &ctx, Context &nctx,float eta) = 0;
    virtual void updateInputMatrix(size_t target, const Context &ctx, Context &nctx, float eta) = 0;

    Utils::FloatSpan operator[](const std::string& s) { return Wo[m_corpus[s]]; }

protected:
    static constexpr unsigned NUM_NEG_SAMPLES = 15;

    Corpus &m_corpus;
    EmbeddingMatrix<float> Wi; // Word matrix
    EmbeddingMatrix<float> Wo; // Context matrix
    Utils::FloatSpan h, dh;    // input vector, and its gradient;
    Utils::FloatSpan v, dv;    // output vector, and its gradient
};

class SkipGram : public Embedding {
public:
    SkipGram(Corpus &corpus, size_t K, unsigned contextSize)
            : Embedding(corpus, K, K % 2 == 0 ? K / 2 : K / 2 + 1, K / 2) {}
};

// The CBOW model predicts the target word, given a context
// This minimizes cross-entropy loss between the probability vector and the
// embedded vector of the output word
class CBoW : public Embedding {
public:
    CBoW(Corpus &corpus, size_t K, unsigned contextSize)
            : Embedding(corpus, K, K % 2 == 0 ? K / 2 : K / 2 + 1, K / 2) {}

    void updateOutputMatrix(size_t target, const Context &ctx, Context &nctx, float eta) override;
    void updateInputMatrix(size_t target, const Context &ctx, Context &nctx, float eta)  override;
};

#endif // WORDEMBEDDING_EMBEDDING_HXX