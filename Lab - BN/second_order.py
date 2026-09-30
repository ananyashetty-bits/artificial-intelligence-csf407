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

# Count observed triples: (x[t-2], x[t-1]) -> x[t].
counts = defaultdict(Counter)
for sentence in tokenized:
    for a, b, c in zip(sentence, sentence[1:], sentence[2:]):
        counts[(a, b)][c] += 1

# Convert counts to P(next | previous two tokens).
probabilities = {}
for context, next_counts in counts.items():
    total = sum(next_counts.values())
    probabilities[context] = {
        nxt: count / total for nxt, count in next_counts.items()
    }

def next_distribution(context):
    return probabilities.get(tuple(context), {})

def predict_next(context):
    distribution = next_distribution(context)
    if not distribution:
        return None
    return max(distribution, key=distribution.get)

def sample_next(context):
    distribution = next_distribution(context)
    if not distribution:
        return None
    words = list(distribution)
    weights = [distribution[w] for w in words]
    return random.choices(words, weights=weights, k=1)[0]

def generate_sentence(greedy=False, max_length=30):
    # Generate the first token from P(X1 | START), then use two-token contexts.
    first_counts = defaultdict(int)
    for sentence in tokenized:
        if len(sentence) > 1:
            first_counts[sentence[1]] += 1
    total = sum(first_counts.values())
    first_probs = {w: c / total for w, c in first_counts.items()}

    if greedy:
        current = max(first_probs, key=first_probs.get)
    else:
        words = list(first_probs)
        weights = [first_probs[w] for w in words]
        current = random.choices(words, weights=weights, k=1)[0]

    if current == END:
        return ""

    output = [current]
    previous = START

    for _ in range(max_length - 1):
        distribution = next_distribution((previous, current))
        if not distribution:
            break
        if greedy:
            nxt = max(distribution, key=distribution.get)
        else:
            nxt = sample_next((previous, current))
        if nxt == END:
            break
        output.append(nxt)
        previous, current = current, nxt

    return " ".join(output)

if __name__ == "__main__":
    print("Selected second-order conditional distributions:")
    for context in [(START, "the"), ("the", "cat"), ("the", "dog"),
                    ("cat", "sat"), ("dog", "sat"), ("dog", "ran")]:
        print(context, next_distribution(context))

    print("\n20 sampled sentences:")
    for i in range(20):
        print(f"{i+1:2d}. {generate_sentence()}")
