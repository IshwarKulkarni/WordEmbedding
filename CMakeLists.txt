cmake_minimum_required(VERSION 3.15)
project(WordEmbedding)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-march=native")

set(CMAKE_VERBOSE_MAKEFILE on)

add_executable(WordEmbedding src/main.cpp src/Corpus.cpp src/Corpus.hxx src/Utils.hxx src/Embedding.cpp src/Embedding.hxx)
