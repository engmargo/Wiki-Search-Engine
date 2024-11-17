import lightgbm

from document_preprocessor import Tokenizer
from indexing import InvertedIndex, BasicInvertedIndex
from ranker import *
from tqdm import tqdm
import os
import pandas as pd
import time


class L2RRanker:
    def __init__(
        self,
        document_index: InvertedIndex,
        title_index: InvertedIndex,
        document_preprocessor: Tokenizer,
        stopwords: set[str],
        ranker: Ranker,
        feature_extractor: "L2RFeatureExtractor",
    ) -> None:
        """
        Initializes a L2RRanker model.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object ** hw3 modified **
            feature_extractor: The L2RFeatureExtractor object
        """
        self.dindex = document_index
        self.tindex = title_index
        self.tokenizer = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.ftextractor = feature_extractor
        self.model = LambdaMART()

    def prepare_training_data(
        self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]
    ):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores (dict): A dictionary of queries mapped to a list of
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            tuple: A tuple containing the training data in the form of three lists: x, y, and qgroups
                X (list): A list of feature vectors for each query-document pair
                y (list): A list of relevance scores for each query-document pair
                qgroups (list): A list of the number of documents retrieved for each query
        """
        # This is for LightGBM to know how many relevance scores we have per query.
        X = []
        y = []
        qgroups = []

        # for each query and the documents that have been rated for relevance to that query,
        # process these query-document pairs into features

        # Accumulate the token counts for each document's title and content

        # For each of the documents, generate its features, then append
        # the features and relevance score to the lists to be returned
        for query, value in query_to_document_relevance_scores.items():
            qgroups.append(len(value))
            query_parts = self.tokenizer.tokenize(query)
            query_parts = [
                None if token in self.stopwords else token for token in query_parts
            ]

            body_counts = self.accumulate_doc_term_counts(self.dindex, query_parts)
            title_counts = self.accumulate_doc_term_counts(self.tindex, query_parts)

            for doc in value:
                y.append(doc[1])
                doc_body_counts = body_counts.get(doc[0], {})
                doc_title_counts = title_counts.get(doc[0], {})
                X.append(
                    self.ftextractor.generate_features(
                        doc[0], doc_body_counts, doc_title_counts, query_parts, query
                    )
                )

        # Make sure to keep track of how many scores we have for this query in qrels
        print(f"len_X:{len(X)},len_y:{len(y)},sum_qgroups:{np.sum(qgroups)}")

        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(
        index: InvertedIndex, query_parts: list[str]
    ) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # Retrieve the set of documents that have each query word (i.e., the postings) and
        # create a dictionary that keeps track of their counts for the query word
        doc_term_counts = defaultdict(lambda: defaultdict(int))
        for term in query_parts:
            if term in index.vocabulary:
                for docid, freq in index.index[term]:
                    doc_term_counts[docid][term] = index.document_metadata[docid][
                        "tokens_count"
                    ][term]

        return doc_term_counts

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        #  Convert the relevance data into the right format for training data preparation
        path = os.path.join(os.path.dirname(__file__), training_data_filename)
        dataset = pd.read_csv(path, encoding="unicode_escape")[
            ["query", "docid", "rel"]
        ]
        dataset["rel"] = np.round(dataset["rel"], 0).astype(int)
        rows = dataset.shape[0]
        q2drel = defaultdict(list)
        for row in range(rows):
            q2drel[dataset.iloc[row, 0]].append(
                (dataset.iloc[row, 1], dataset.iloc[row, 2])
            )

        print("training data loaded")
        # prepare the training data by featurizing the query-doc pairs and
        # getting the necessary datastructures
        X, y, qgroups = self.prepare_training_data(q2drel)

        # Train the model
        # print("---X---")
        # print(X)
        # print("--y--")
        # print(y)
        # print("--qgroups--")
        # print(qgroups)
        self.model = self.model.fit(X, y, qgroups)

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        try:
            predict_rank = self.model.predict(X)
        except:
            raise ValueError("Model has not been trained yet.")

        #  Return a prediction made using the LambdaMART model
        return predict_rank

    def query(
        self,
        query: str,
        pseudofeedback_num_docs=0,
        pseudofeedback_alpha=0.8,
        pseudofeedback_beta=0.2,
        user_id=None,
    ) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # Retrieve potentially-relevant documents

        # Fetch a list of possible documents from the index and create a mapping from
        # a document ID to a dictionary of the counts of the query terms in that document.
        # Pass the dictionary to the RelevanceScorer as input.

        # Accumulate the documents word frequencies for the title and the main body

        # Score and sort the documents by the provided scrorer for just the document's main text (not the title)
        # This ordering determines which documents we will try to *re-rank* using our L2R model

        # Filter to just the top 100 documents for the L2R part for re-ranking
        if len(query) == 0:
            return []
        else:
            query_parts = [
                None if term in self.stopwords else term
                for term in self.tokenizer.tokenize(query)
            ]

            flag = False
            for term in query_parts:
                if (term in self.dindex.vocabulary) or (term in self.tindex.vocabulary):
                    flag = True
                    break

            if flag:
                dcandidates = self.ranker.query(
                    query,
                    pseudofeedback_num_docs,
                    pseudofeedback_alpha,
                    pseudofeedback_beta,
                )
                x_predict = []

                # Construct the feature vectors for each query-document pair in the top 100
                for idx, doc in enumerate(dcandidates):
                    if idx < 100:
                        dword_counts = self.get_body_title_tokens(
                            self.dindex, doc[0], query_parts
                        )
                        tword_counts = self.get_body_title_tokens(
                            self.tindex, doc[0], query_parts
                        )
                        x_predict.append(
                            self.ftextractor.generate_features(
                                doc[0], dword_counts, tword_counts, query_parts, query
                            )
                        )

                    else:
                        break
                # Use L2R model t/Users/leilei/Desktop/courses/si650/HW2/starter-code/base_l2r.json /Users/leilei/Desktop/courses/si650/HW2/starter-code/bm25.jsono rank these top 100 documents
                predict_rank = self.model.predict(x_predict)
                # Sort posting_lists based on scores
                new_partial_rank = sorted(
                    [
                        (dcandidates[i][0], predict_rank[i])
                        for i in range(len(predict_rank))
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
                # Make sure to add back the other non-top-100 documents that weren't re-ranked
                start = len(predict_rank)
                end = len(dcandidates)
                if start < end - 1:
                    new_partial_rank.extend(dcandidates[start:end])
                # Return the ranked documents
                return new_partial_rank
            else:
                return []

    def get_body_title_tokens(
        self, index: InvertedIndex, docid: int, query: list[str]
    ) -> tuple:
        tokens_count = {
            token: count
            for token, count in index.document_metadata[docid]["tokens_count"].items()
            if token in query
        }
        return tokens_count


class L2RFeatureExtractor:
    def __init__(
        self,
        document_index: InvertedIndex,
        title_index: InvertedIndex,
        doc_category_info: dict[int, list[str]],
        document_preprocessor: Tokenizer,
        stopwords: set[str],
        recognized_categories: set[str],
        docid_to_network_features: dict[int, dict[str, float]],
        ce_scorer: CrossEncoderScorer,
    ) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        self.dindex = document_index
        self.tindex = title_index
        self.tokenizer = document_preprocessor
        self.stopwords = stopwords
        self.ntfeatures = docid_to_network_features
        self.doc_category_info = doc_category_info
        self.docid_to_network_features = docid_to_network_features

        self.categories = recognized_categories

        self.dindex_bm25 = BM25(document_index)
        self.dindex_piv = PivotedNormalization(document_index)
        self.ce_scorer = ce_scorer

    # Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.dindex.document_metadata[docid]["length"]

    # Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.tindex.document_metadata[docid]["length"]

    # pass after-stopwords-filtered queryparts
    # TF
    def get_tf(
        self,
        index: InvertedIndex,
        docid: int,
        word_counts: dict[str, int],
        query_parts: list[str],
    ) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        score = 0
        for term in query_parts:
            if term not in self.stopwords and term in word_counts:
                score += np.log(word_counts[term] + 1)
        return score

    # TF-IDF
    def get_tf_idf(
        self,
        index: InvertedIndex,
        docid: int,
        word_counts: dict[str, int],
        query_parts: list[str],
    ) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """

        return TF_IDF(index).score(docid, word_counts, Counter(query_parts))

    def get_BM25_score(
        self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]
    ) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # Calculate the BM25 score and return it
        return self.dindex_bm25.score(docid, doc_word_counts, Counter(query_parts))

    # Pivoted Normalization
    def get_pivoted_normalization_score(
        self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]
    ) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score

        """
        #  Calculate the pivoted normalization score and return it
        return self.dindex_piv.score(docid, doc_word_counts, Counter(query_parts))

    #  Document Categories
    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has.
        """
        binary_categories = [0] * len(self.categories)
        doc_categories = self.doc_category_info[docid]
        for idx, cate in enumerate(self.categories):
            if cate in doc_categories:
                binary_categories[idx] = 1

        return binary_categories

    #  Pagerank score
    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score
        """
        if docid in self.docid_to_network_features:
            return self.docid_to_network_features[docid]["pagerank"]
        else:
            return 0

    #  HITS Hub score
    def get_hits_hub_score(self, docid: int) -> float:
        """
        Gets the HITS hub score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS hub score
        """
        if docid in self.docid_to_network_features:
            return self.docid_to_network_features[docid]["hub_score"]
        else:
            return 0

    #  HITS Authority score
    def get_hits_authority_score(self, docid: int) -> float:
        """
        Gets the HITS authority score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS authority score
        """
        if docid in self.docid_to_network_features:
            return self.docid_to_network_features[docid]["authority_score"]
        else:
            return 0

    #  Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """
        return self.ce_scorer.score(docid, query)

    #  Add at least one new feature to be used with your L2R model
    def get_query_categories_socre(self, docid: int, query_parts: list[str]) -> float:
        # caculate the tf score of query versus each categories
        qcscore = dict()
        for cate in self.categories:
            score = 0
            cate_count = Counter(cate)
            for term in query_parts:
                if term in cate_count:
                    score += np.log(cate_count[term] + 1)
            qcscore[cate] = score
        # map the category similarity score to corresponding documents
        score = 0
        for cate in self.doc_category_info[docid]:
            if cate in self.categories:
                score += qcscore[cate]
        return score

    def generate_features(
        self,
        docid: int,
        doc_word_counts: dict[str, int],
        title_word_counts: dict[str, int],
        query_parts: list[str],
        query: str,
    ) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """

        feature_vector = []
        #  Document Length
        feature_vector.append(self.get_article_length(docid))
        #  Title Length
        feature_vector.append(self.get_title_length(docid))
        #  Query Length
        feature_vector.append(len(query_parts))
        #  TF (document)
        feature_vector.append(
            self.get_tf(self.dindex, docid, doc_word_counts, query_parts)
        )
        #  TF-IDF (document)
        feature_vector.append(
            self.get_tf_idf(self.dindex, docid, doc_word_counts, query_parts)
        )
        #  TF (title)
        feature_vector.append(
            self.get_tf(self.tindex, docid, title_word_counts, query_parts)
        )
        #  TF-IDF (title)
        feature_vector.append(
            self.get_tf_idf(self.tindex, docid, title_word_counts, query_parts)
        )
        #  BM25
        feature_vector.append(self.get_BM25_score(docid, doc_word_counts, query_parts))
        #  Pivoted Normalization
        feature_vector.append(
            self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts)
        )
        #  Pagerank
        feature_vector.append(self.get_pagerank_score(docid))
        #  HITS Hub
        feature_vector.append(self.get_hits_hub_score(docid))
        #  HITS Authority
        feature_vector.append(self.get_hits_authority_score(docid))
        #  Cross-Encoder Score
        # feature_vector.append(1)
        feature_vector.append(
            self.get_cross_encoder_score(docid, query)
        )  # only applicable when query is tokenized by blank space
        #  Add at least one new feature to be used with your L2R model.
        feature_vector.append(self.get_query_categories_socre(docid, query_parts))
        #  Calculate the Document Categories features.
        feature_vector.extend(self.get_document_categories(docid))
        # This should be a list of binary values indicating which categories are present.

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            "objective": "lambdarank",
            "boosting_type": "gbdt",
            "n_estimators": 10,
            "importance_type": "gain",
            "metric": "ndcg",
            "num_leaves": 20,
            "learning_rate": 0.005,
            "max_depth": -1,
            "n_jobs": 4,
            # "verbosity": 1,
        }

        if params:
            default_params.update(params)

        #  initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.model = lightgbm.LGBMRanker(**default_params)

    def fit(self, X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples.
            y_train (array-like): Target values.
            qgroups_train (array-like): Query group sizes for training data.

        Returns:
            self: Returns the instance itself.
        """

        #  fit the LGBMRanker's parameters using the provided features and labels
        self.model = self.model.fit(X_train, y_train, group=qgroups_train)

        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like):
                A list of featurized documents where each document is a list of its features
                All documents should have the same length.

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """

        #  Generating the predicted values using the LGBMRanke
        return self.model.predict(featurized_docs)
