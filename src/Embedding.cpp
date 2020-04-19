//
// Created by ishwark on 11/04/20.
// Copyright 2020 Ishwar Kulkarni.
// Subject to GPL V2 License(www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
//

#include "Embedding.hxx"
#include <random>
#include <algorithm>

Embedding::Embedding(Corpus &corpus, size_t K, size_t pCt, size_t nCt, size_t seed)
        : m_corpus(corpus),
          Wi(K, corpus.getVocabularySize()),
          Wo(K, corpus.getVocabularySize()),
          h(K), dh(K),
          v(K), dv(K) {
    auto almostSame = [](const std::string &w1, const std::string &w2) -> float {
        size_t min, max;
        std::tie(min, max) = std::minmax(w1.size(), w2.size());
        if (float(min) / max < 0.2)
            return 0.f;
        float same = 0;
        for (size_t i = 0; i < min; ++i)
            same += float(w1[i] == w2[i]);
        return same / min;
    };
    std::mt19937 rng_engine(seed);
    std::uniform_real_distribution<float> dist(-0.1, 0.1);
    auto perturb = [&rng_engine, &dist](Utils::Span<float> &&sp) {
        for (auto &s : sp)
            s += dist(rng_engine);
    };

    m_corpus.initIterators(pCt, nCt);
    size_t sz = m_corpus.getVocabularySize();
    for (size_t i = 1; i < sz; ++i) {
        auto prev = m_corpus[i - 1];
        auto &curr = m_corpus[i];
        float same = almostSame(prev, curr);
        if (same > 0.8) {
            // std::cout << same << ":\t" << prev << " : " << curr << std::endl;

            Wo[i].deepCopy(Wo[i - 1]);
            Wi[i].deepCopy(Wi[i - 1]);

            perturb(Wo[i]);
            perturb(Wi[i]);
        }
    }
}

void Embedding::train(float eta, size_t maxSample)
{
    size_t word, wi;
    Context ctx, nCtx;
    m_corpus.resetIterators();

    for (size_t i = 0; m_corpus.next(word, ctx, wi) and i < maxSample; ++i) {
        if (ctx.empty())
            continue;

        nCtx.clear();
        for (size_t n = 0; n < NUM_NEG_SAMPLES; ++n)
            nCtx.push_back(m_corpus.sampleVocab());

        updateH(ctx, word);
        updateOutputMatrix(word, ctx, nCtx, eta);
        updateInputMatrix(word, ctx, nCtx, eta);
    }
}

void Embedding::serialize(std::ofstream &file) {
    size_t n = Wi.minorSize * Wi.minorSize;
    file << Wi.majorSize << ' ' << Wi.minorSize << ' ';
    file.write(reinterpret_cast<const char *>(Wi.getData()), n * sizeof(float));
    file.write(reinterpret_cast<const char *>(Wo.getData()), n * sizeof(float));
}

void Embedding::updateV(size_t w, float tj, float eta)
{
    float s = Utils::Sigmoid( Wo[w] * h ) - tj;
    dv.deepCopy(h);
    dv *= (s * eta);
    auto v = Wo[w];
    v -= dv;
}

const Utils::FloatSpan& Embedding::getdH(size_t w, float tj, float eta)
{
    float s = Utils::Sigmoid(Wo[w] * h) - tj;
    v.deepCopy(Wo[w]);
    v *= (s * eta);
    return v;
}

void CBoW::updateOutputMatrix(size_t target, const Context &ctx,  Context& nCtx, float eta)
{
    // EQN 58/59 from https://arxiv.org/pdf/1411.2738.pdf
    updateV(target, 1, eta);
    for(auto& nc: nCtx)
        updateV(nc, 0.f, eta);
}

void SkipGram::updateOutputMatrix(size_t target, const Context &ctx, Context &nCtx, float eta)
{
    for(auto& c : ctx)
        updateV(c, 1, eta);

    for(auto&nc : nCtx)
        updateV(nc, 0, eta);
}

void CBoW::updateInputMatrix(size_t target, const Context &ctx, Context& nCtx, float eta)
{
    // EQN 61 from https://arxiv.org/pdf/1411.2738.pdf , dh := EH
    dh.deepCopy(getdH(target, 1, eta));
    for(auto& nc: nCtx)
        dh += getdH(nc, 0.f, eta);

    for(auto& c : ctx)
    {
        auto vi = Wi[c];
        vi -= dh;
    }
}

void SkipGram::updateInputMatrix(size_t target, const Context &ctx, Context &nCtx, float eta)
{

    // EQN 61 from https://arxiv.org/pdf/1411.2738.pdf , dh := EH
    dh.deepCopy(getdH(target, 1, eta));
    for (auto &nc: nCtx)
        dh += getdH(nc, 0.f, eta);

    auto vi = Wi[target];
    vi -= dh;
}

void CBoW::updateH(Context& ctx, size_t word)
{
    h.fill(0.f);
    for (auto &c : ctx)
        h += Wi[c];
    h /= ctx.size();
}
