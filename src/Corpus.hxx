//
// Created by ishwark on 08/04/20.
//

#ifndef WORD_EMBEDDING_CORPUS_H
#define WORD_EMBEDDING_CORPUS_H

#include "Utils.hxx"
#include <cmath>
#include <fstream>
#include <map>
#include <vector>

class Corpus {
public:
    explicit Corpus(const std::vector<std::string> &filenames, size_t seed);

    void initIterators(size_t prevCt, size_t nextCt);

    void resetIterators();

    [[nodiscard]] size_t getVocabularySize() const {
        return m_uniqueWordCount.size();
    }

    bool next(size_t &word, std::vector<size_t> &m_returnedWords,
              size_t &wordIndexs);

    [[nodiscard]] size_t operator[](const std::string &word) const;

    [[nodiscard]] const std::string &operator[](size_t code) const;

    [[nodiscard]] bool useWord(size_t w) const;

    [[nodiscard]] size_t sampleWord(size_t maxAttempts = 100)
    const; // sample a word based on above rejection criterion

private:
    size_t m_numWords = 0;

    size_t m_prevCt = 0, m_nextCt = 0;

    static constexpr float SAMPLE_THRESH = (1e-3f);

    std::map<std::string, size_t> m_uniqueWordCount;

    std::vector<std::vector<size_t>> m_sentences;
    std::vector<std::vector<size_t>>::iterator m_sentenceIter;
    std::vector<size_t>::iterator m_wordIter;

    std::vector<std::string> m_orderedWords;
    std::vector<float> m_occurenceCounts; // z(wi)/0.001

    void buildWordCount(const char *filename);

    void flattenWordCounts();

    void encodeSource(const char *filename);
};

#endif // WORD_EMBEDDING_CORPUS_H
