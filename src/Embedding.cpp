//
// Created by ishwark on 11/04/20.
//

#include "Embedding.hxx"
#include <iostream>

Embedding::Embedding(Corpus &corpus, size_t embeddingSize, unsigned pCt,
                     unsigned nCt)
    : m_corpus(corpus),
      Wi(embeddingSize, corpus.getVocabularySize()),
      Wo(embeddingSize, corpus.getVocabularySize()),
      h(embeddingSize), dh(embeddingSize),
      v(embeddingSize), dv(embeddingSize)
{
    m_corpus.initIterators(pCt, nCt);
}

void Embedding::train(float eta, size_t maxSample) {
    size_t word, wi;
    Context ctx, nCtx;
    m_corpus.resetIterators();

    for (size_t nSample = 0; m_corpus.next(word, ctx, wi) and nSample < maxSample; ++nSample) {
        if (ctx.empty())
            continue;

        nCtx.clear();
        for (unsigned n = 0; n < NUM_NEG_SAMPLES; ++n)
            nCtx.push_back(m_corpus.sampleVocab());

        h.fill(0.f);
        for (auto &c : ctx)
            h += Wi[c];
        h /= ctx.size();

        updateOutputMatrix(word, ctx, nCtx, eta);
        updateInputMatrix(word, ctx, nCtx, eta);
    }
}

void Embedding::serialize(std::ofstream &file) {
    file << Wi.majorSize << ' ' << Wi.minorSize << ' ';
    file.write(reinterpret_cast<const char *>(Wi.getData()), Wi.minorSize * Wi.minorSize * sizeof(float));
    file.write(reinterpret_cast<const char *>(Wo.getData()), Wi.minorSize * Wi.minorSize * sizeof(float));
}

void CBoW::updateOutputMatrix(size_t target, const Context &ctx,  Context& nCtx, float eta)
{
    auto updateV = [this, eta](size_t w, float tj)
    {
        float s = Utils::Sigmoid( Wo[w] * h ) - tj;
        dv.copyFrom(h);
        dv *= (s * eta);
        auto v = Wo[w];
        v -= dv;
    };

    // EQN 58/59 from https://arxiv.org/pdf/1411.2738.pdf
    updateV(target, 1);
    for(auto& nc: nCtx)
        updateV(nc, 0.f);
}

void CBoW::updateInputMatrix(size_t target, const Context &ctx, Context& nCtx, float eta)
{
    auto getdH = [this, eta](size_t w, float tj) -> const Utils::FloatSpan&
    {
        float s = Utils::Sigmoid( Wo[w] * h ) - tj;
        v.copyFrom(Wo[w]);
        v *= (s * eta);
        return v;
    };

    // EQN 61 from https://arxiv.org/pdf/1411.2738.pdf , dh := EH
    dh.copyFrom(getdH(target, 1));
    for(auto& nc: nCtx)
        dh += getdH(nc, 0.f);

    for(auto& c : ctx)
    {
        auto vi = Wi[c];
        vi -= dh;
    }
}
