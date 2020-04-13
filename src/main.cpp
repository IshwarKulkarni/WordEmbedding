//
// Created by ishwark on 11/04/20.
// Copyright 2020 Ishwar Kulkarni.
// Subject to GPL V2 License(www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
//

#include "Corpus.hxx"
#include "Utils.hxx"
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include "Embedding.hxx"
#include "EmbeddingEvaluator.hxx"

using namespace Utils;
using namespace std;

struct ProgramArgs {

    ProgramArgs(int argc, char **argv) {

        auto argsToFileList = [](char *arg, const std::string &argname, std::vector<std::string> &files) {
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
    float learningRate = 0.002;
    size_t prevCt = 3;
    size_t nextCt = 2;
    size_t numEpochs = 25;

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

    size_t seed = rseed();

    std::cout << seed << " <-- seed\n";
    Corpus corpus(args.sources, args.ignore, seed);
    std::ofstream corpusFile("corpus.txt");
    corpus.serialize(corpusFile);

    SkipGram model(corpus, args.embeddingSize, args.prevCt + args.nextCt);

    EmbeddingEvaluator evaluator(model, seed);
    evaluator.addWordGrpFiles("data/synonyms.txt", "data/antonyms.txt");

    for (unsigned i = 0; i < args.numEpochs; ++i) {
        auto start = clock();
        float eta = args.learningRate * powf(0.9, float(i));
        model.train(eta);
        float time = float(clock() - start) / CLOCKS_PER_SEC;

        std::cout << "\nIter " << i << " in " << time << "s." << endl;
        evaluator.evaluate();
        if (i % 5 == 0) {
            std::ofstream file(std::to_string(i) + "-iter.model");
            model.serialize(file);
        }
    }
    return 0;
}
