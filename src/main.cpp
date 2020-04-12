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

        auto argsToFileList = [](char *arg, const std::string &argname, std::vector<std::string> &files)
        {
            if (strstr(arg, argname.c_str()) != arg)
                return;
            for (char *s = strtok(arg + argname.length(), ","); s;) {
                files.emplace_back(s);
                s = strtok(nullptr, ",");
            }
        };

        for (auto &arg : Utils::make_span(argv + 1, argc - 1)) {
            argsToFileList(arg, "--sources=", sources);
            argsToFileList(arg, "--ignore=", ignore);
            getValue(arg, "--embeddingSize=", embeddingSize);
            getValue(arg, "--eta=", learningRate);
            getValue(arg, "--prevCt=", prevCt);
            getValue(arg, "--nextCt=", nextCt);
        }
    }

    vector<string> sources = {}, ignore = {};
    size_t embeddingSize = 64;
    float learningRate = 0.01;
    size_t prevCt = 3;
    size_t nextCt = 2;
    size_t numEpochs = 5;

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
    Corpus corpus(args.sources, args.ignore, seed);
    std::ofstream corpusFile("corpus.txt");
    corpus.serialize(corpusFile);

    CBoW model(corpus, args.embeddingSize, args.prevCt + args.nextCt);

    auto test = [&model](size_t i, float eta, float time) {
        static auto nation = model["nation"];
        static auto state = model["state"];
        static auto country = model["country"];

        std::cout << i << ":\t\t(nation * state): " << nation * state
                  << "\t(nation * country): " << nation * country
                  << "\t(country * state): " << country * state
                  << "\t\titeration in " << time << "s."
                  << "\t\t eta: " << eta << '.' << endl;
    };

    test(0, args.learningRate, 0.0f);

    for (unsigned i = 0; i < args.numEpochs; ++i) {
        auto start = clock();
        float eta = args.learningRate * powf(0.95, float(i));
        model.train(eta);
        float time = float(clock() - start) / CLOCKS_PER_SEC;

        test(i + 1, eta, time);
        if (i % 10 == 0) {
            std::ofstream file(std::to_string(i) + "-iter.model");
            model.serialize(file);
        }
    }
    return 0;
}
