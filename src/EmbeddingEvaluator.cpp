//
// Created by ishwark on 11/04/20.
// Copyright 2020 Ishwar Kulkarni.
// Subject to GPL License(www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
//

#include <sstream>
#include "EmbeddingEvaluator.hxx"
#include <iostream>

void EmbeddingEvaluator::WordGroups::updateCosDistMuSigma() {
    std::vector<float> pairWiseDist;
    mu = 0.f;
    for (size_t i = 0; i < embeddings.size(); ++i) {
        for (size_t j = i + 1; j < embeddings.size(); ++j) {
            float cost = embeddings[i] * embeddings[j];
            mu += cost;
            pairWiseDist.push_back(cost);
        }
    }

    mu /= pairWiseDist.size();
    float sumSqDist = 0;
    for (auto &c : pairWiseDist) {
        c = (mu - c);
        sumSqDist += c * c;
    }
    sigma = sqrtf(sumSqDist / pairWiseDist.size());
}


void EmbeddingEvaluator::addWordGrpFiles(const char *synonymsList, const char *antonymsList)
{

    std::vector<std::string> words; // list of random words.
    std::string line; // line to process

    auto lineToWordSet = [this, &words, &line]()
    {
        WordGroups sset;
        std::istringstream ss(line);
        std::string word;
        while (ss >> word) {
            try {
                const auto &emb = m_embedding[word];
                sset.words.push_back(word);
                sset.embeddings.push_back(emb);
                words.push_back(word);
            } catch (std::runtime_error &re) {
                std::cerr << "Word: " << word << " not found in corpus";
                continue;
            }
        }
        return sset;
    };

    std::ifstream file(synonymsList);

    while (std::getline(file, line))
    {
        const auto& sset = lineToWordSet();
        if (!sset.words.empty())
            m_synonyms.push_back(sset);
    }

    file.close();
    file.open(antonymsList);
    while (std::getline(file, line))
    {
        const auto& sset = lineToWordSet();
        if (sset.words.size() == 2)
            m_antonyms.push_back(sset);
    }

    std::shuffle(words.begin(), words.end(), m_gen);
    line = "";
    for(size_t  i= 0; i < words.size() && m_randomList.size() <= RANDOM_LIST_CT; ++i)
    {
        if(i and i % 5 == 0)
        {
            m_randomList.push_back(lineToWordSet());
            line = "";
        }
        line += (" " + words[i]);
    }
}

void EmbeddingEvaluator::evaluate()
{
    bool randomized = false;
    bool printAFew = false;
    doWordGroupTest(m_synonyms, randomized, printAFew,   "\tSynonyms   ");
    doWordGroupTest(m_antonyms, randomized, printAFew,   "\tAntonyms   ");
    doWordGroupTest(m_randomList, randomized, printAFew, "\tRandomized ");
}

void EmbeddingEvaluator::doWordGroupTest(std::vector<WordGroups> &grp,
                                         bool randomized, bool printAFew,
                                         const std::string& grpName)
{
    float meanMu = 0, meanSig = 0;
    for (auto &sset : grp) {
        if (randomized && m_realDist(m_gen) > RANDOMIZE_PROB)
            continue;

        sset.updateCosDistMuSigma();
        meanMu += sset.mu;
        meanSig += sset.sigma;
        if (printAFew && m_realDist(m_gen) < PRINT_PROB)
            std::cout << sset;
    }
    std::cout << grpName
              << " : Mean mean-dist: " << meanMu / grp.size()
              << " Mean Sigma: " << meanSig / grp.size() << std::endl;
}

void EmbeddingEvaluator::doCountriesTest(bool randomized, bool printAFew)
{
    for (size_t i = 0; i < 10; ++i) {
        size_t r = m_intDist(m_gen) % m_countries.size();
        auto &country = m_countries[r];
        std::partial_sort(m_countries.begin(), m_countries.begin() + 5, m_countries.end(),
                          [&country](const auto &w1, const auto &w2) {
                              return w1.embedding * country.embedding < w2.embedding * country.embedding;
                          });

        std::cout << "\n" << country.word << "|\t closest 5: ";
        for (size_t k = 0; k < 5; ++k)
            std::cout << m_countries[k].word << ", ";
    }
}

std::ostream &operator<<(std::ostream &out, EmbeddingEvaluator::WordGroups &set) {
    out << "\t|";
    for (auto &w : set.words)
        out << w << '\t';
    out << "| Mean Cos Dist: " << set.mu << " Sigma: " << set.sigma << "\n";
    return out;
}
