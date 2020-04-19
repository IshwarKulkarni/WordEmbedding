//
// Created by ishwark on 11/04/20.
// Copyright 2020 Ishwar Kulkarni.
// Subject to GPL V2 License(www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
//

#ifndef WORD_EMBEDDING_CORPUS_H
#define WORD_EMBEDDING_CORPUS_H

#include "Utils.hxx"
#include <cmath>
#include <fstream>
#include <map>
#include <unordered_set>
#include <random>
#include <vector>

class Corpus {
public:
    explicit Corpus(const std::vector<std::string> &filenames,
                    const std::vector<std::string> &ignoredWordFiles, size_t seed);

    void initIterators(size_t prevCt, size_t nextCt);

    void resetIterators();

    inline size_t getVocabularySize() const { return m_uniqueWordCount.size(); }

    bool next(size_t &word, std::vector<size_t> &mRetWords, size_t &wrdIdx);

    [[nodiscard]] size_t operator[](const std::string &word) const;

    [[nodiscard]] const std::string &operator[](size_t wc) const;

    [[nodiscard]] bool useWord(size_t w);

    inline size_t sampleVocab() { return m_vocabDistribution(m_generator); }

    void serialize(std::ofstream &file);

    explicit Corpus(std::ifstream &file, size_t seed);

private:

    size_t m_numWords = 0;

    size_t m_prevCt = 0, m_nextCt = 0;

    static constexpr float SAMPLE_THRESH = 5e-4f;

    static constexpr float SAMPLING_POW = 0.4f;

    std::map<std::string, size_t> m_uniqueWordCount;

    std::vector<std::vector<size_t>> m_sentences;

    std::vector<std::vector<size_t>>::iterator m_sentenceIter;
    std::vector<size_t>::iterator m_wordIter;

    std::vector<std::string> m_orderedWords;
    std::vector<float> m_occurenceFrequency;

    void buildWordCount(const char *filename);

    void flattenWordCounts();

    void encodeSource(const char *filename);

    bool clean(std::string &word) const;

    std::unordered_set<std::string> m_ignoreWords;

    std::unordered_set<std::string> m_stopWords =
            {"the", "and", "said", "could", "for", "reuter", "that", "from",
             "will", "its", "with", "was", "has", "would", "not", "which",
             "are", "but", "have", "this", "bank", "inc", "were", "net", "last",
             "had", "they", "one", "also", "about", "loss", "been", "more",
             "may", "their", "first", "than", "other", "all", "our", "some"};

    std::mt19937 m_generator;

    std::discrete_distribution<size_t> m_vocabDistribution;
};

#endif // WORD_EMBEDDING_CORPUS_H
