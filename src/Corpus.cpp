//
// Created by ishwark on 08/04/20.
//

#include "Corpus.hxx"
#include <algorithm>
#include <exception>
#include <iostream>
#include <iterator>
#include <random>

enum WordState {
    EmptyWord, LastWord, NormalWord
};

static std::pair<std::string, WordState> clean(const std::string &word) {
    auto first = word.begin();
    auto last = std::prev(word.end());

    WordState state = NormalWord;
    while (*last == ',' or *last == '.' or *last == '\'' or *last == '\"' or
           *last == '!' or *last == ':' or *last == '-' or *last == ')' or
           *last == '?' or *last == ';') {
        if (*last == '.')
            state = LastWord;
        last = std::prev(last);
    }

    while (*first == '\'' or *first == '\"' or *first == '(' or *first == ' ')
        first = std::next(first);

    if (last - first > 1) {
        std::string s;
        for (auto c = first; c <= last; ++c)
            if (isalpha(*c) or *c == '\'')
                s.push_back(tolower(*c));
        if (state != LastWord)
            state = s.empty() ? EmptyWord : NormalWord;
        return {s, state};
    } else
        return {"", EmptyWord};
}

static std::default_random_engine g_generator;

static std::uniform_int_distribution<size_t> g_randomSentenceDist;

static std::uniform_real_distribution<float> g_realDistribution; // [0,1)

Corpus::Corpus(const std::vector<std::string> &filenames, size_t seed) {
    if (filenames.empty())
        throw std::runtime_error("Sources Empty");

    for (const auto &s : filenames)
        buildWordCount(s.c_str());

    flattenWordCounts();

    for (const auto &s : filenames)
        encodeSource(s.c_str());

    std::cout << "Corpus built with " << m_numWords << " original words, "
              << m_uniqueWordCount.size() << " unique words in "
              << m_sentences.size() << " sentences.\n";

    g_generator.seed(seed);

    g_randomSentenceDist =
            std::uniform_int_distribution<size_t>(0, m_sentences.size() - 1);
}

void Corpus::buildWordCount(const char *filename) {
    size_t existingCt = m_uniqueWordCount.size();
    std::ifstream source(filename, std::ios::binary);
    std::string word;
    while (source >> word) {
        auto ret = clean(word);
        if (ret.second != EmptyWord) {
            m_uniqueWordCount[ret.first]++;
            m_numWords++;
        }
    }

    if (m_uniqueWordCount.size() - existingCt == 0)
        throw std::runtime_error("Word count is zero, file at \"" +
                                 std::string(filename) + "\" could not be read");
}

void Corpus::flattenWordCounts() {
    size_t existingCt = m_uniqueWordCount.size();

    m_orderedWords.reserve(m_uniqueWordCount.size());
    m_occurenceCounts.reserve(m_uniqueWordCount.size());
    auto wordCt = float(m_numWords) * SAMPLE_THRESH;
    for (auto &p : m_uniqueWordCount) {
        m_orderedWords.push_back(p.first);
        m_occurenceCounts.push_back(p.second / wordCt);
    }
}

void Corpus::encodeSource(const char *filename) {
    std::ifstream source(filename, std::ios::binary);

    std::string word;
    std::vector<size_t> sentence;

    size_t existingCt = m_sentences.size();

    while (source >> word) {
        auto ret = clean(word);
        if (ret.second != EmptyWord) {
            size_t enc = (*this)[ret.first];
            if (useWord(enc))
                sentence.push_back(enc);
            if (ret.second == LastWord) {
                if (sentence.size() > 1)
                    m_sentences.push_back(sentence); // no one word sentences.
                sentence.clear();
            }
        }
    }

    if (m_sentences.size() - existingCt == 0)
        throw std::runtime_error("Encoding the file failed");
}

size_t Corpus::operator[](const std::string &word) const {
    auto found =
            std::equal_range(m_orderedWords.begin(), m_orderedWords.end(), word);
    if (found.first == m_orderedWords.end()) {
        throw std::runtime_error("Unknown word \"" + word + "\" queried");
    }

    return std::distance(m_orderedWords.begin(), found.first);
}

const std::string &Corpus::operator[](size_t o) const {
    if (o >= m_orderedWords.size()) {
        throw std::runtime_error("Word not found ");
    }
    return m_orderedWords[o];
}

bool Corpus::useWord(size_t w) const {
    float z = m_occurenceCounts[w];
    float K = (1.f + sqrtf(z)) / z;
    return g_realDistribution(g_generator) < K;
}

size_t Corpus::sampleWord(size_t maxAttempts) const {
    for (size_t i = 0; i < maxAttempts; ++i) {
        size_t r = g_randomSentenceDist(g_generator) % m_uniqueWordCount.size();
        if (useWord(r))
            return r;
    }
    return g_randomSentenceDist(g_generator);
}

void Corpus::initIterators(size_t prevCt, size_t nextCt) {
    m_prevCt = prevCt;
    m_nextCt = nextCt;
    m_sentenceIter = m_sentences.begin();
    m_wordIter = m_sentenceIter->begin()++;
}


void Corpus::resetIterators()
{
    m_sentenceIter = m_sentences.begin();
    m_wordIter = m_sentenceIter->begin()++;
}

bool Corpus::next(size_t &word, std::vector<size_t> &m_returnedWords,
                  size_t &wordIndex) {

    m_returnedWords.clear();
    wordIndex = size_t(-1);

    if (m_wordIter == m_sentenceIter->end()) {
        if (++m_sentenceIter == m_sentences.end())
            return false;
        m_wordIter = m_sentenceIter->begin()++;
    }

    word = *m_wordIter;

    auto first = m_wordIter == m_sentenceIter->begin() ? m_wordIter
                                                       : std::prev(m_wordIter);
    auto last =
            m_wordIter == m_sentenceIter->end() ? m_wordIter : std::next(m_wordIter);

    for (size_t i = 0; i < m_prevCt && first != m_sentenceIter->begin();
         ++i, --first)
        m_returnedWords.push_back(*first);

    wordIndex = m_returnedWords.size();

    for (size_t i = 0; i < m_nextCt && last != m_sentenceIter->end(); ++i, last++)
        m_returnedWords.push_back(*last);

    m_wordIter++;

    return true;
}