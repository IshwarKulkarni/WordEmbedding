cmake_minimum_required(VERSION 3.15)
project(WordEmbedding)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-march=native")

set(CMAKE_VERBOSE_MAKEFILE on)

add_executable(WordEmbedding
        src/Corpus.cpp
        src/Corpus.hxx

        src/Embedding.cpp
        src/Embedding.hxx

        src/EmbeddingEvaluator.cpp
        src/EmbeddingEvaluator.hxx

        src/Utils.hxx

        src/main.cpp
        )
