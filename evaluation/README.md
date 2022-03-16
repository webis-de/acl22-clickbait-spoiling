# Evaluation of Approaches for Clickbait Spoiling

The code in this directory is used to evaluate the predictions of spoiling approaches.
The focus of evaluation is mainly on the semantical similarity of two sections of text, but syntax shouldn't be neglected too much.
The metrics we chose are BLEU-4, METEOR 1.5 and BERTScore.

- Evaluate predictions with all three metrics with [eval_abs_with_thresholds.sh](eval_abs_with_thresholds.sh)

