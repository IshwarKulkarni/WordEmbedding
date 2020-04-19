//
// Created by ishwark on 11/04/20.
// Copyright 2020 Ishwar Kulkarni.
// Subject to GPL V2 License(www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
//

#ifndef WORDEMBEDDING_EMBEDDINGEVALUATOR_HXX
#define WORDEMBEDDING_EMBEDDINGEVALUATOR_HXX

#include "Embedding.hxx"
#include <random>

class EmbeddingEvaluator {
public:

    explicit EmbeddingEvaluator(Embedding &embedding, size_t seed) :
            m_embedding(embedding),
            m_gen(seed) {};

    void addWordGrpFiles(const char *synonymsList, const char *antonymsList);

    void addCountriesFile(const char* countryList);

    void evaluate();

    struct WordGroups {
        std::vector<std::string> words;
        std::vector<Utils::FloatSpan> embeddings;

        float mu = 0;
        float sigma = 0; // mean and std. dev of pair wise distances among word embeddings/
        void updateCosDistMuSigma();
    };

private:

    static constexpr float RANDOMIZE_PROB = 0.15; // evaluate about 15% of the word groups;
    static constexpr float PRINT_PROB = 0.02; // print 2% among evaluated;
    static constexpr size_t RANDOM_LIST_CT = 40;


    struct WordEmbedding {
        std::string word;
        Utils::FloatSpan embedding;
    };

    void doWordGroupTest(std::vector<WordGroups> &grp,
                         bool randomized, bool printAFew,
                         const std::string& grpName);


    void doCountriesTest(bool randomized, bool printAFew);

    Embedding &m_embedding;

    std::vector<WordGroups> m_synonyms;
    std::vector<WordGroups> m_antonyms;
    std::vector<WordGroups> m_randomList;

    std::vector<WordEmbedding> m_countries;

    std::uniform_real_distribution<float> m_realDist;
    std::uniform_int_distribution<size_t> m_intDist;
    std::mt19937 m_gen;
};

std::ostream &operator<<(std::ostream &out, EmbeddingEvaluator::WordGroups &set);


#endif //WORDEMBEDDING_EMBEDDINGEVALUATOR_HXX
