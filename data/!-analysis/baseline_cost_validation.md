# Baseline cost + wall-time validation
Sampled 200 fresh calls per cell on MedQA dev. Measured tokens + wall time, then extrapolated to the full dev sweep and compared against the tiktoken-estimated costs in [data/8-verifier/results.md](../8-verifier/results.md).

| cell | n | tokens in (cached) | tokens out | ¢/call (measured) | full dev $ (measured-extrap) | full dev $ (tiktoken-est) | wall p95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `snippet/full-context/gpt-5.4-high` | 200 | 151,120 (26,624) | 72,777 | 0.70¢ | $3.48 | $6.65 | 31.13 |
| `snippet/claim-only/gpt-5.4-high` | 200 | 17,501 (0) | 56,536 | 0.45¢ | $2.20 | $6.02 | 15.87 |
| `atom/full-context/gpt-5.4-high` | 200 | 130,401 (17,408) | 43,907 | 0.47¢ | $5.10 | $14.43 | 16.48 |
| `atom/claim-only/gpt-5.4-high` | 200 | 14,216 (0) | 34,250 | 0.27¢ | $2.96 | $13.10 | 10.71 |
