//
// Created by ishwark on 11/04/20.
// Copyright 2020 Ishwar Kulkarni.
// Subject to GPL License(www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
//

#ifndef WORDEMBEDDING_EMBEDDING_HXX
#define WORDEMBEDDING_EMBEDDING_HXX

#include "Corpus.hxx"
#include "Utils.hxx"
#include <algorithm>

template<typename T, typename T2 = void>
struct EmbeddingMatrix;

template<typename T>
struct EmbeddingMatrix<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
    EmbeddingMatrix(size_t major, size_t minor) :
            majorSize(major),
            minorSize(minor),
            data(new T[majorSize * minorSize]) {
        size_t seed = Utils::rseed();
        static std::default_random_engine generator(seed);
        static std::uniform_real_distribution<T> distribution(0, 1);
        for (size_t i = 0; i < majorSize * minorSize; ++i)
            data[i] = distribution(generator);
    }

    const T *getData() { return data.get(); }

    inline Utils::Span<T> operator[](size_t offset) {
#ifndef NDEBUG
        if (offset >= minorSize)
          throw std::runtime_error("Offset too large: " + std::to_string(offset));
#endif
        return Utils::make_span(data.get() + offset * majorSize, majorSize);
    }

    inline const Utils::Span<T> &operator[](size_t offset) const {
        return Utils::make_span(data.get() + offset * majorSize, majorSize);
    }

    const size_t majorSize, minorSize;

private:
    std::unique_ptr<T[]> data;
};

// These predictive embedding models minimize the loss of predicting a target word,
// given a context in case of CBoW subclass and that of predicting context words, given
// a word in case of SkipGram subClass. In both cases the loss function is Cross Entropy
// between predicted and 'corrected' distribution.
// Bother sub classes use  Negative sampling optimization to reduce compute.
class Embedding {
public:

    using Context = std::vector<size_t>;

    Embedding(Corpus &corpus, size_t K, size_t pCt, size_t nCt);

    void train(float eta, size_t maxSamples = size_t(-1));

    Utils::FloatSpan operator[](const std::string &s) { return Wo[m_corpus[s]]; }

    void serialize(std::ofstream &file);

protected:

    virtual void updateOutputMatrix(size_t target, const Context &ctx, Context &nctx, float eta) = 0;

    virtual void updateInputMatrix(size_t target, const Context &ctx, Context &nctx, float eta) = 0;

    virtual void updateH(Context &ctx, size_t word) = 0;

    void updateV(size_t w, float tj, float eta);

    const Utils::FloatSpan &getdH(size_t w, float tj, float eta);

    static constexpr size_t NUM_NEG_SAMPLES = 20;

    Corpus &m_corpus;
    EmbeddingMatrix<float> Wi; // Word matrix
    EmbeddingMatrix<float> Wo; // Context matrix
    Utils::FloatSpan h, dh;    // input vector, and its gradient;
    Utils::FloatSpan v, dv;    // output vector, and its gradient
};

class SkipGram : public Embedding {
public:
    SkipGram(Corpus &corpus, size_t K, size_t contextSize)
            : Embedding(corpus, K, K % 2 == 0 ? K / 2 : K / 2 + 1, K / 2) {}

    void updateOutputMatrix(size_t target, const Context &ctx, Context &nctx, float eta) override;
    void updateInputMatrix(size_t target, const Context &ctx, Context &nctx, float eta)  override;

    void updateH(Context&, size_t word) override  { h.copyFrom(Wi[word]); }
};

class CBoW : public Embedding {
public:
    CBoW(Corpus &corpus, size_t K, size_t contextSize)
            : Embedding(corpus, K, K % 2 == 0 ? K / 2 : K / 2 + 1, K / 2) {}

    void updateOutputMatrix(size_t target, const Context &ctx, Context &nctx, float eta) override;
    void updateInputMatrix(size_t target, const Context &ctx, Context &nctx, float eta)  override;

    void updateH(Context& ctx, size_t) override;
};

#endif // WORDEMBEDDING_EMBEDDING_HXX
