//
// Created by ishwark on 11/04/20.
//

#include "Embedding.hxx"

Embedding::Embedding(Corpus &corpus, size_t K, unsigned pCt, unsigned nCt)
    : m_corpus(corpus),
      Wi(K, corpus.getVocabularySize()),
      Wo(K, corpus.getVocabularySize()),
      h(K), dh(K),
      v(K), dv(K)
{
    m_corpus.initIterators(pCt, nCt);
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
        for (unsigned n = 0; n < NUM_NEG_SAMPLES; ++n)
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
    dv.copyFrom(h);
    dv *= (s * eta);
    auto v = Wo[w];
    v -= dv;
}

const Utils::FloatSpan& Embedding::getdH(size_t w, float tj, float eta)
{
    float s = Utils::Sigmoid(Wo[w] * h) - tj;
    v.copyFrom(Wo[w]);
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
    dh.copyFrom(getdH(target, 1, eta));
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
    dh.copyFrom(getdH(target, 1, eta));
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
