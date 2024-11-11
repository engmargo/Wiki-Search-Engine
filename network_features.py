from pandas import DataFrame
from sknetwork.data import from_edge_list
from sknetwork.ranking import PageRank, HITS
import pandas as pd
import os


class NetworkFeatures:
    """
    A class to help generate network features such as PageRank scores, HITS hub score and HITS authority scores.
    This class uses the scikit-network library https://scikit-network.readthedocs.io to calculate node ranking values.

    Ref:
    [1]PageRank: https://towardsdatascience.com/pagerank-algorithm-fully-explained-dc794184b4af
    [2]HITS: https://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture4/lecture4.html
    """

    def load_network(self, network_filename: str, total_edges: int):
        """
        Loads the network from the specified file and returns the network. A network file
        can be listed using a .csv or a .csv.gz file.

        Args:
            network_filename: The name of a .csv or .csv.gz file containing an edge list
            total_edges: The total number of edges in an edge list

        Returns:
            The loaded network from sknetwork
        """
        #   load the network edgelist dataset and return the scikit-network graph
        path = os.path.join(os.path.dirname(__file__), network_filename)
        if "gz" in network_filename:
            data = pd.read_csv(path, compression="gzip")
        else:
            data = pd.read_csv(path)

        return from_edge_list(
            data.values, directed=True, matrix_only=False, reindex=True
        )

    def calculate_page_rank(
        self, graph, damping_factor=0.85, iterations=100
    ) -> list[float]:
        """
        Calculates the PageRank scores for the provided network and
        returns the PageRank values for all nodes.

        Args:
            graph: A graph from sknetwork
            damping_factor: The complement of the teleport probability for the random walker
                For example, a damping factor of .8 has a .2 probability of jumping after each step.
            iterations: The maximum number of iterations to run when computing PageRank

        Returns:
            The PageRank scores for all nodes in the network (array-like)
        """
        #  Use scikit-network to run Pagerank and return Pagerank scores
        pgrank = PageRank(damping_factor=damping_factor, n_iter=iterations)
        adjacency = graph.adjacency
        scores = pgrank.fit_predict(adjacency)

        return scores

    def calculate_hits(self, graph) -> tuple[list[float], list[float]]:
        """
        Calculates the hub scores and authority scores using the HITS algorithm
        for the provided network and returns the two lists of scores as a tuple.

        Args:
            graph: A graph from sknetwork

        Returns:
            The hub scores and authority scores (in that order) for all nodes in the network
        """
        #  Use scikit-network to run HITS and return HITS hub scores and authority scores

        adjacency = graph.adjacency
        hits_predict = HITS().fit(adjacency)

        return (list(hits_predict.scores_row_), list(hits_predict.scores_col_))

    def get_all_network_statistics(self, graph) -> DataFrame:
        """
        Calculates the PageRank and the hub scores and authority scores using the HITS algorithm
        for the provided network and returns a pandas DataFrame with columns:
        'docid', 'pagerank', 'authority_score', and 'hub_score' containing the relevant values
        for all nodes in the network.

        Args:
            graph: A graph from sknetwork

        Returns:
            A pandas DataFrame with columns 'docid', 'pagerank', 'authority_score', and 'hub_score'
            containing the relevant values for all nodes in the network
        """

        #  Calculate all the Pagerank and HITS scores for the network graph and store it in a dataframe
        pgrank_scores = self.calculate_page_rank(graph)
        hits_scores = self.calculate_hits(graph)
        names = graph.names

        results = pd.DataFrame()
        results["docid"] = names
        results["pagerank"] = pgrank_scores
        results["authority_score"] = hits_scores[1]
        results["hub_score"] = hits_scores[0]

        # results.set_index(['docid']).to_csv(os.path.join(os.path.dirname(__file__),'network_statistics1011.csv'))

        return results
