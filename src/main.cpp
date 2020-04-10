#include "Corpus.hxx"
#include "Utils.hxx"
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include "Embedding.hxx"

using namespace Utils;
using namespace std;

struct ProgramArgs {

    ProgramArgs(int argc, char **argv) {
        for (auto &arg : Utils::make_span(argv + 1, argc - 1)) {
            const string src_arg = "--sources=";
            if (strstr(arg, src_arg.c_str()) == arg) {
                for (char *s = strtok(arg + src_arg.length(), ","); s;
                     s = strtok(nullptr, ","))
                    sources.emplace_back(s);
            }
            getValue(arg, "--embeddingSize=", embeddingSize);
            getValue(arg, "--eta=", learningRate);
            getValue(arg, "--prevCt=", prevCt);
            getValue(arg, "--nextCt=", nextCt);
            getValue(arg, "--negSampled=", negSamples);
        }
    }

    vector<string> sources = {};
    size_t embeddingSize = 70;
    float learningRate = 0.01;
    size_t prevCt = 3;
    size_t nextCt = 2;
    size_t numEpochs = 100;
    size_t negSamples = 20;

private:
    stringstream ss;

    template<typename T>
    void getValue(char *arg, const string &name, T &val) {
        ss.clear();
        if (strstr(arg, name.c_str()) == arg)
            ss << (arg + name.length()), ss >> val;
    }
};

int main(int argc, char **argv) {
    const ProgramArgs args(argc, argv);

    size_t seed = time(nullptr);
#ifndef NDEBUG
    seed = 42;
#endif
    Corpus corpus(args.sources, seed);
    CBoW model(corpus, args.embeddingSize, 5);

    auto nation = model["nation"];
    auto state = model["state"];
    auto country = model["country"];

    for(unsigned i = 0; i < args.numEpochs; ++i)
    {
        model.train(args.learningRate * powf(0.9, float(i)));
        std::cout << i << ": \t(nation * state): " << nation * state
                  << "\t(nation * country): " << nation * country
                  << "\t(country * state): " << country * state << endl;
    }

    return 0;
}
