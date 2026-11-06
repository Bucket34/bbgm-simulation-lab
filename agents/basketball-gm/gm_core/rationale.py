"""Narrow heuristic checks for language-model explanations.

These pattern checks do NOT constitute full factual verification.
"""
import re


def rationale_flags(reason, scores, cap):
    """Flag two known misleading explanation patterns from V5.1."""
    flags = []
    text = reason.lower()
    if re.search(r'(all|every)\s+(?:legal\s+)?(?:offers?|players?|options?)\b.{0,75}'
                 r'(?:negative|no\s+positive|not\s+worth|low\s+value)', text):
        if any(v > 0 for v in scores.values()):
            flags.append('Reason suggests all offers have low/negative value, but at least one policy score is positive')
    if re.search(r'(?:cap|salary\s+cap)\s+(?:is\s+)?(?:fully\s+used|exhausted|at\s+zero)', text):
        if cap > 0:
            flags.append(f'Reason claims cap is exhausted, but current available cap is {cap}')
    return flags
