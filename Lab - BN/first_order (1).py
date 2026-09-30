from collections import defaultdict, Counter
import random

DATA = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

START = "<START>"
END = "<END>"
tokenized = [[START] + s.lower().split() + [END] for s in DATA]

# Count consecutive-token transitions.
counts = defaultdict(Counter)
for sentence in tokenized:
    for current, nxt in zip(sentence, sentence[1:]):
        counts[current][nxt] += 1

# Convert counts to P(next | current).
probabilities = {}
for current, next_counts in counts.items():
    total = sum(next_counts.values())
    probabilities[current] = {
        nxt: count / total for nxt, count in next_counts.items()
    }

def next_distribution(current):
    return probabilities.get(current, {})

def predict_next(current):
    distribution = next_distribution(current)
    if not distribution:
        return None
    return max(distribution, key=distribution.get)

def sample_next(current):
    distribution = next_distribution(current)
    if not distribution:
        return None
    words = list(distribution)
    weights = [distribution[w] for w in words]
    return random.choices(words, weights=weights, k=1)[0]

def generate_sentence(greedy=False, max_length=30):
    current = START
    output = []
    for _ in range(max_length):
        distribution = next_distribution(current)
        if not distribution:
            break
        if greedy:
            nxt = max(distribution, key=distribution.get)
        else:
            nxt = sample_next(current)
        if nxt == END:
            break
        output.append(nxt)
        current = nxt
    return " ".join(output)

if __name__ == "__main__":
    for word in ["the", "cat", "dog", "sat", "ran"]:
        print(word, next_distribution(word))
        print("most probable:", predict_next(word))

    print("\n20 sampled sentences:")
    for i in range(20):
        print(f"{i+1:2d}. {generate_sentence()}")
