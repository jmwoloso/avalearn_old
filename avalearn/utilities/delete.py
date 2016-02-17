"""Count words."""
from collections import Counter
def count_words(s, n):
    """Return the n most frequently occuring words in s."""

    # TODO: Count the number of occurences of each word in s
    words = s.split()

    # get list of 2-tuples of (word, count)
    counts = Counter(words)
    words_counts = [(k, counts[k]) for k in counts]


    # TODO: Sort the occurences in descending order (alphabetically in case of ties)
    sorted_words = sorted(words_counts, key=lambda x: (-x[1], x[0]))
    print(sorted_words)
    #sorted_words = sorted(sorted_words, key=lambda x: -x[1], reverse=True)
    # TODO: Return the top n words as a list of tuples (<word>, <count>)
    top_n = words_counts[0:n]s = "cat bat mat cat bat cat" \
                                 ""
    return top_n


def test_run():
    """Test count_words() with some inputs."""
    print(count_words("cat bat mat cat bat cat", 3))
    print( count_words("betty bought a bit of butter but the butter "
                       "was bitter", 3))


if __name__ == '__main__':
    test_run()
