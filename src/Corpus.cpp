//
// Created by ishwark on 08/04/20.
//

#include "Corpus.hxx"
#include <algorithm>
#include <exception>
#include <iostream>
#include <random>

using namespace std;

bool Corpus::clean(string &word) const {
    auto first = word.begin();
    auto last = prev(word.end());

    bool endOfSentence = false;

    while (*last == ',' or *last == '\'' or *last == '\"' or
           *last == ':' or *last == '-' or *last == ')')
        last = prev(last);


    while (*last == '.' or *last == '?' or *last == ';' or *last == '!') {
        last = prev(last);
        endOfSentence = true;
    }

    while (*first == '\'' or *first == '\"' or
           *first == '(' or *first == ' ')
        first = std::next(first);

    if (last - first > 1) {
        string s;
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

Corpus::Corpus(const vector<string> &filenames,
               const vector<string> &ignoredWordFiles, size_t seed)
        : m_generator(seed) {
    if (filenames.empty())
        throw runtime_error("Sources Empty");

    for (auto &fn : ignoredWordFiles) {
        ifstream file(fn);
        string s;
        while (file >> s)
            m_ignoreWords.insert(s);
    }

    for (const auto &s : filenames)
        buildWordCount(s.c_str());

    flattenWordCounts();

    for (const auto &s : filenames)
        encodeSource(s.c_str());

    cout << "Corpus built with " << m_numWords << " original words, "
              << m_uniqueWordCount.size() << " unique words in "
              << m_sentences.size() << " sentences.\n";
}


template<typename F>
void goThroughFile(const char *filename, F function) {
    auto isBodyOpenTag = [](const string &word) {
        return word.find("<BODY>") != string::npos ||
               word.find("<body>") != string::npos;
    };

    auto isBodyCloseTag = [](const string &word) {
        return word.find("</BODY>") != string::npos ||
               word.find("</body>") != string::npos;
    };

    auto isFileLookingLikeMarkUp = [](const char *filename) {
        ifstream file(filename);
        if (!file)
            throw runtime_error("Couldn't open file at " + string(filename));
        string s;
        file >> s;
        return s == "<!DOCTYPE";
    };

    bool ismarkup = isFileLookingLikeMarkUp(filename);

    ifstream source(ifstream(filename, ios::binary));
    string word;
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

    goThroughFile(filename, [this](string &s) {
        clean(s);
        if (s.empty())
            return;
        m_uniqueWordCount[s]++;
        m_numWords++;
    });

    if (m_uniqueWordCount.size() - existingCt == 0)
        throw runtime_error("Word count is zero, file at \"" +
                            string(filename) + "\" could not be read");
}

void Corpus::flattenWordCounts() {
    m_orderedWords.reserve(m_uniqueWordCount.size());
    m_occurenceFrequency.reserve(m_uniqueWordCount.size());

    vector<float> vocabDistVec;
    vocabDistVec.reserve(m_uniqueWordCount.size());

    auto wordCt = float(m_numWords);
    for (auto &p : m_uniqueWordCount) {
        m_orderedWords.push_back(p.first);
        float f = p.second / wordCt;
        m_occurenceFrequency.push_back(f / SAMPLE_THRESH);
        vocabDistVec.push_back(powf(f, SAMPLING_POW));
    }
    m_vocabDistribution = discrete_distribution<size_t>(vocabDistVec.begin(), vocabDistVec.end());
}

void Corpus::encodeSource(const char *filename) {
    ifstream source(filename, ios::binary);

    string word;
    vector<size_t> sentence;

    size_t existingCt = m_sentences.size();

    goThroughFile(filename, [this, &sentence](string &s) {
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
        throw runtime_error("Encoding the file failed");
}

size_t Corpus::operator[](const string &word) const {
    auto found =
            equal_range(m_orderedWords.begin(), m_orderedWords.end(), word);
    if (found.first == m_orderedWords.end()) {
        throw runtime_error("Unknown word \"" + word + "\" queried");
    }

    return distance(m_orderedWords.begin(), found.first);
}

const string &Corpus::operator[](size_t wc) const {
    if (wc >= m_orderedWords.size()) {
        throw runtime_error("Word not found ");
    }
    return m_orderedWords[wc];
}

bool Corpus::useWord(size_t w) {
    static uniform_real_distribution<float> s_realDistribution; // [0,1)
    float z = m_occurenceFrequency[w];
    float K = (1.f + sqrtf(z)) / z;
    return s_realDistribution(m_generator) < K;
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

bool Corpus::next(size_t &word, vector<size_t> &mRetWords, size_t &wrdIdx) {

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
void Corpus::serialize(ofstream &file) {
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

Corpus::Corpus(ifstream &file, size_t seed) :
        m_generator(seed) {
    size_t numUniqueWords = 0;
    file >> m_numWords
         >> m_prevCt
         >> m_nextCt
         >> numUniqueWords;

    string s;
    size_t count;
    for (size_t i = 0; i < numUniqueWords; ++i) {
        file >> s >> count;
        m_uniqueWordCount[s] = count;
    }

    flattenWordCounts();

    cout << "Corpus built with " << m_numWords << " original words, "
         << m_uniqueWordCount.size() << " unique words in "
         << m_sentences.size() << " sentences.\n";

    size_t numSentences = 0;
    file >> numSentences;
    m_sentences.resize(numSentences);
    for (size_t i = 0; i < numSentences; ++i) {
        size_t numWords;
        file >> numWords;
        vector<size_t> sentence(numWords, 0);
        for (auto &w : sentence)
            file >> s;
    }
}

