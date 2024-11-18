# Wiki-Search-Engine
A 200k-articles-based Wikipedia Search Engine using multiple NLP models, from basic SVM models to high-level techniques. 

## The Current Best Performing Model in this project
### Streamline
### Performance

| Dataset | MAP | NDCG |
|---------|-----|------|
| Trai|     |      |
| Test    |     |      |


## Pipeline Overview
![alt text](image.png)

## Pipeline Details
### `document_preprocessor.py`
####  `Tokenizer`  class
A generic class for objects that turn strings into sequences of tokens. It is defaulted to lowcase all words and skip mulwiword expression match. 

Child classes include: `SplitTokenizer`, `RegexTokenizer` , `SpacyTokenizer`.

 <em> **Note that:** `RegexTokenizer` with pattern `\w+` is the defaulted tokenizer for this project. </em>

### `indexing.py`

#### `IndexType` class

Currently two types are supported by the system, `PositionalIndex` and `BasicInveredIndex` 

#### `InvertedIndex` class
attributes include:
- `statistics`: `vocab` corpus vocabulary tokens count, `total_token_count` # of total tokens including those filtered , `number_of_documents` , `unique_token_count`, `stored_total_token_count` # of total tokens excluding those filtered
- `vocabulary`: corpus vocabulary
- `document_metadata`: {docid:dict}, where dict includes `length`,`unique_tokens`,`tokens_count`
- `index`:{token:[(docid1,freq),(docid2,freq)]}

main functions include:
- `.remove_doc(*)`
- `.add_doc(*)`
- `.get_postings(*)`
- `.get_doc_metadata(*)`
- `.get_term_metadata(*)`
- `.get_statistics(*)`
- `.save()`
- `.load()`

 <em> **Note that:** `BasicInveredIndex` is the defaulted indexing method for this project. </em>

#### `Indexer` class
main functions include:
- `.create_index(*)`

### `network_features.py`

A class to generate network features, such as PageRank scores