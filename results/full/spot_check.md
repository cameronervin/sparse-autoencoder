# Label spot-check

12 of the 200 model labels, picked at random (seed 7), checked against each feature's top 5 contexts.

| Feature | Model label | Verdict |
|---|---|---|
| 2004 | "in the" phrase completion | Correct. Fires on " the" after "in", mostly in economic reporting. |
| 2043 | "ev" word prefixes | Correct. evade, evoke, evaced. |
| 2361 | Words ending in "-al" | Too broad. Fires on "-al" after department, function, development. |
| 2948 | "Comes down to" phrase | Correct. |
| 3927 | Unicode replacement character | Correct. First byte of a split multi-byte character. |
| 9434 | "In response to" phrase | Correct. |
| 10665 | Accession word forms | Correct. "accession", often EU accession. |
| 11721 | Sentence-initial "Until" | Correct. |
| 16163 | Neural terminology | Correct. |
| 17473 | Duration in years | Too narrow. Also fires on "hours". Replaced with a hand label. |
| 18621 | Expected event prediction | Correct. "is expected to". |
| 20104 | Free or no-cost offer | Correct. |

10 of 12 are correct and 2 are close but off in scope. The model's self-reported confidence is not useful: the median is 0.99 and 193 of 200 are 0.8 or higher.
