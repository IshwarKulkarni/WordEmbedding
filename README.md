A simple word embedding application.

What's so special about this: Well, nothing other than this is written in pure C++, no external libraries
Written mostly as an exercise and learn word embedding in detail.

Reads documents into a 'coprus', that is then fed to a 'Embedding Model'
This model can be trained any number of times and 'serialized' to be loaded later.


Corpus:

    Uses comma separated --sources argument to load up the words and setnece structure.
    Also allows for stop words and other ignored words to instantiated. Can be serialized and has a 
    non-deterministic behaviour. 
   

Model Class:

    Has two subclasses:
        CBoW : Continuos Bag of words, given multiple context words predict a target word. 
        SkipGram: A single root word predicts a set of 'context words'
    Both use negative sampling technique with about 20 negative sample per positive sample to 
    improve perf. 
   
Embedding Matrix:
    
    The "Matrix", is actually implemented as simply an orderer collection of Word Vector 
    (or embeddings) that is interfaced using a 'Span' structure (as the math here only needs 
    simple dot products between rows on the matrix. 
    
Evaluators:
    
    Can evaluate a model based on 'closes-ness' of synonym words, 'far-ness' of antonym words and 
    'meh-ness' of random words. Printing these for each training epoch gives a good view of the 
    process
    
Misc Notes:
    
    Uses GCC and C++17 std. using the micro arch option to GCC, we can use AVX2 on modern Intel 
    CPU's. Datasets inclued one-corpus of Tolstoy's 'War and Peace', Multi-style document of POTUS 
    speeches and finally the Reuters dataset.   
    
 
 TODO:
 
    1. AVX2 vectorization. - Completed with '-march=native' cmake option.
    2. Get better datasets. - Added Retuers dataset.
    3. Better 'sampleWord()' in corpus - using Freq based sampling/
    4. Adding evaluation scripts - EmbeddingEvaluator.
    5. CUDA!! -- May be not. 
    