"""
Core function for auto-keyword-ranker.

Parameters
----------
texts : str or list of str
    Input document(s).
top_n : int, default=10
    Number of keywords to return per document.
ngram_range : tuple, default=(1,2)
    N-gram range for TF-IDF.
stop_words : bool, default=True
    If True, use English stop words.
use_embeddings : bool, default=False
    If True, re-rank TF-IDF candidates using sentence-transformer embeddings.
embedding_model : optional
    Preloaded sentence-transformers model instance. If None and
    use_embeddings=True, the function will attempt to load
    'sentence-transformers/all-MiniLM-L6-v2'.
combine_score_alpha : float, default=0.6
    Weight for TF-IDF in final combined score (0.0-1.0).
    0.6 means 60% TF-IDF, 40% embedding similarity.

Returns
-------
list of (keyword, score) pairs for a single document,
or list of lists for multiple documents.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _ensure_list(texts):
    """Ensure input is a list of strings."""
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def rank_keywords(
    texts,
    top_n: int = 10,
    ngram_range: tuple = (1, 2),
    stop_words: bool = True,
    use_embeddings: bool = False,
    embedding_model=None,
    combine_score_alpha: float = 0.6,
):
    """Extract and (optionally) re-rank keywords from text(s)."""
    texts = _ensure_list(texts)

    stop_words_arg = "english" if stop_words else None
    vect = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words_arg,
        lowercase=True
    )
    X = vect.fit_transform(texts)
    feature_names = vect.get_feature_names_out()

    results = []

    # Optional: prepare embedding model
    embed_model = None
    if use_embeddings:
        try:
            if embedding_model is not None:
                embed_model = embedding_model
            else:
                # lazy import so package stays lightweight
                from sentence_transformers import SentenceTransformer
                embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            raise ImportError(
                "sentence-transformers is required for use_embeddings=True. "
                "Install with `pip install autokeyword[embed]` or "
                "`pip install sentence-transformers`."
            ) from e

    for doc_idx in range(X.shape[0]):
        row = X[doc_idx].toarray().ravel()
        if row.sum() == 0:
            results.append([])
            continue

        # top candidate indices by TF-IDF
        top_indices = np.argsort(row)[::-1][:min(len(row), top_n * 5)]
        candidates = [(feature_names[i], row[i])
                      for i in top_indices if row[i] > 0]

        # Optional embedding re-ranking
        if use_embeddings and embed_model is not None and candidates:
            doc_text = texts[doc_idx]

            # embed document and candidate phrases
            doc_emb = embed_model.encode(doc_text)
            phrases = [p for p, _ in candidates]
            phrase_embs = embed_model.encode(phrases)

            def cosine(a, b):
                a = np.array(a, dtype=float)
                b = np.array(b, dtype=float)
                an = np.linalg.norm(a)
                bn = np.linalg.norm(b)
                if an == 0 or bn == 0:
                    return 0.0
                return float(np.dot(a, b) / (an * bn))

            sims = np.array([cosine(doc_emb, pe) for pe in phrase_embs])
            tfidf_scores = np.array([s for _, s in candidates], dtype=float)

            combined = (
                combine_score_alpha * tfidf_scores
                + (1 - combine_score_alpha) * sims
            )

            ranked = sorted(
                zip([p for p, _ in candidates], combined),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
        else:
            ranked = sorted(
                candidates,
                key=lambda x: x[1],
                reverse=True
            )[:top_n]

        results.append(ranked)

    return results[0] if len(results) == 1 else results
