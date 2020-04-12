//
// Created by ishwark on 08/04/20.
//

#include "Corpus.hxx"
#include <algorithm>
#include <exception>
#include <iostream>
#include <random>

bool Corpus::clean(std::string &word) const {
    auto first = word.begin();
    auto last = std::prev(word.end());

    bool endOfSentence = false;

    while (*last == ',' or *last == '\'' or *last == '\"' or
           *last == ':' or *last == '-' or *last == ')')
        last = std::prev(last);


    while (*last == '.' or *last == '?' or *last == ';' or *last == '!') {
        last = std::prev(last);
        endOfSentence = true;
    }

    while (*first == '\'' or *first == '\"' or
           *first == '(' or *first == ' ')
        first = std::next(first);

    if (last - first > 1) {
        std::string s;
        for (auto c = first; c <= last; ++c)
            if (isalpha(*c) or *c == '\'')
                s.push_back(tolower(*c));
            else if (isdigit(*c)) {
                s = "";
                break;
            }

        word = s;

        if (m_ignoreWords.find(s) == m_ignoreWords.end())
            return endOfSentence;
    }

    word = "";
    return endOfSentence;
}

static std::default_random_engine g_generator;

static std::uniform_real_distribution<float> g_realDistribution; // [0,1)

Corpus::Corpus(const std::vector<std::string> &filenames,
               const std::vector<std::string> &ignoredWordFiles, size_t seed) {
    if (filenames.empty())
        throw std::runtime_error("Sources Empty");

    for (auto &fn : ignoredWordFiles) {
        std::ifstream file(fn);
        std::string s;
        while (file >> s)
            m_ignoreWords.insert(s);
    }

    for (const auto &s : filenames)
        buildWordCount(s.c_str());

    flattenWordCounts();

    for (const auto &s : filenames)
        encodeSource(s.c_str());

    std::cout << "Corpus built with " << m_numWords << " original words, "
              << m_uniqueWordCount.size() << " unique words in "
              << m_sentences.size() << " sentences.\n";

    g_generator.seed(seed);
}


template<typename F>
void goThroughFile(const char *filename, F function) {
    auto isBodyOpenTag = [](const std::string &word) {
        return word.find("<BODY>") != std::string::npos ||
               word.find("<body>") != std::string::npos;
    };

    auto isBodyCloseTag = [](const std::string &word) {
        return word.find("</BODY>") != std::string::npos ||
               word.find("</body>") != std::string::npos;
    };

    auto isFileLookingLikeMarkUp = [](const char *filename) {
        std::ifstream file(filename);
        if (!file)
            throw std::runtime_error("Couldn't open file at " + std::string(filename));
        std::string s;
        file >> s;
        return s == "<!DOCTYPE";
    };

    bool ismarkup = isFileLookingLikeMarkUp(filename);

    std::ifstream source(std::ifstream(filename, std::ios::binary));
    std::string word;
    bool accumulateOn = !ismarkup;
    while (source >> word) {
        if (ismarkup) {
            if (!accumulateOn and isBodyOpenTag(word))
                accumulateOn = true;

            if (!accumulateOn)
                continue;
            else if (isBodyCloseTag(word))
                accumulateOn = false;
        }
        function(word);
    }
}

void Corpus::buildWordCount(const char *filename) {
    size_t existingCt = m_uniqueWordCount.size();

    goThroughFile(filename, [this](std::string &s) {
        clean(s);
        if (s.empty())
            return;
        m_uniqueWordCount[s]++;
        m_numWords++;
    });

    if (m_uniqueWordCount.size() - existingCt == 0)
        throw std::runtime_error("Word count is zero, file at \"" +
                                 std::string(filename) + "\" could not be read");
}

void Corpus::flattenWordCounts() {

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

    goThroughFile(filename, [this, &sentence](std::string &s) {
        auto eos = clean(s);
        if (s.empty())
            return;
        size_t enc = (*this)[s];
        if (useWord(enc))
            sentence.push_back(enc);
        if (eos) {
            if (sentence.size() > 1) // no one word sentences.
                m_sentences.push_back(sentence);
            sentence.clear();
        }
    });

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

size_t Corpus::sampleWord(size_t maxAttempts) const
{
    static std::uniform_int_distribution<size_t> randomSentenceDist(m_sentences.size());

    for (size_t i = 0; i < maxAttempts; ++i) {
        size_t r = randomSentenceDist(g_generator) % m_uniqueWordCount.size();
        if (useWord(r))
            return r;
    }
    return randomSentenceDist(g_generator);
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

bool Corpus::next(size_t &word, std::vector<size_t> &mRetWords, size_t &wrdIdx) {

    mRetWords.clear();
    wrdIdx = size_t(-1);

    if (m_wordIter == m_sentenceIter->end()) {
        if (++m_sentenceIter == m_sentences.end())
            return false;
        m_wordIter = m_sentenceIter->begin()++;
    }

    word = *m_wordIter;

    auto first = m_wordIter == m_sentenceIter->begin() ? m_wordIter
                                                       : std::prev(m_wordIter);
    auto last = m_wordIter == m_sentenceIter->end() ? m_wordIter
                                                    : std::next(m_wordIter);

    for (size_t i = 0; i < m_prevCt && first != m_sentenceIter->begin(); ++i, --first)
        mRetWords.push_back(*first);

    wrdIdx = mRetWords.size();

    for (size_t i = 0; i < m_nextCt && last != m_sentenceIter->end(); ++i, last++)
        mRetWords.push_back(*last);

    m_wordIter++;

    return true;
}

// Very flaky implem of serializing and deserializing
void Corpus::serialize(std::ofstream &file) {
    file << m_numWords << " "
         << m_prevCt << " "
         << m_nextCt << " "
         << m_uniqueWordCount.size() << '\n';

    for (auto &uw : m_uniqueWordCount)
        file << uw.first << ' ' << uw.second << '\n';

    file << m_sentences.size() << '\n';
    for (auto &s : m_sentences) {
        file << s.size() << '\t';
        for (auto &w : s)
            file << w << ' ';
        file << '\n';
    }
}

Corpus::Corpus(std::ifstream &file, size_t seed) {
    size_t numUniqueWords = 0;
    file >> m_numWords
         >> m_prevCt
         >> m_nextCt
         >> numUniqueWords;

    std::string s;
    size_t count;
    for (size_t i = 0; i < numUniqueWords; ++i) {
        file >> s >> count;
        m_uniqueWordCount[s] = count;
    }

    flattenWordCounts();

    std::cout << "Corpus built with " << m_numWords << " original words, "
              << m_uniqueWordCount.size() << " unique words in "
              << m_sentences.size() << " sentences.\n";

    g_generator.seed(seed);

    size_t numSentences = 0;
    file >> numSentences;
    m_sentences.resize(numSentences);
    for (size_t i = 0; i < numSentences; ++i) {
        size_t numWords;
        file >> numWords;
        std::vector<size_t> sentence(numWords, 0);
        for (auto &w : sentence)
            file >> s;
    }
}

