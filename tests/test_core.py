from autokeyword.core import rank_keywords

def test_rank_keywords_simple():
    text = "Cats and dogs are common pets. Dogs bark. Cats meow."
    kws = rank_keywords(text, top_n=3)
    # should return words like 'dogs' or 'cats'
    tokens = [k for k, _ in kws]
    assert any(t in tokens for t in ["dogs", "cats"])