# AI Laboratory — Bayesian Networks and Autoregressive Language Models

## Files
- `first_order.py` — first-order model P(next | current)
- `second_order.py` — second-order model P(next | previous two tokens)
- `report.pdf` — completed laboratory report
- `prompts.txt` — LLM prompts and validation/correction notes
- `results.txt` — compact experimental output
- `requirements.txt` — no third-party packages

## Run
```bash
python first_order.py
python second_order.py
```

## Dataset
The six sentences are exactly the starting dataset supplied by the laboratory.

## Main measured checks
- First-order conditional distributions normalize to 1 (maximum error: 0).
- Second-order conditional distributions normalize to 1 (maximum error: 0).
- First-order observed contexts: 11
- Second-order observed contexts: 14
- First-order nonzero conditional entries: 17
- Second-order nonzero conditional entries: 18

The report uses the terminology and tasks from the supplied laboratory handout.
