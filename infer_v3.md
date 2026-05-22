# Scenario Results

| Context | Result | Why |
|---|---|---|
| BMW M3 | Works well | BMW content is well-represented in corpus; BM25 exact-matched "M3 Touring" and lifted it from rank ~18 to top-10 |
| Honda Civic | Works well | Honda Civic Facelift 2025 jumps to rank 1; other Honda content surfaces. Good brand density + exact BM25 keyword hits |
| Mercedes-Benz | Works well | Top 7 results are all Mercedes-specific. Large, well-represented brand with strong semantic signal |
| Drag Race | Weak | Rank 1 is "intense dramatic" content (not drag racing). "Drag Race" is semantically ambiguous to the backbone — the embedding space conflates it with general high-intensity automotive content |
| Jetour | Weak | Only 2 Jetour videos appear at ranks 12 and 14. Jetour is a niche brand; BM25 found few keyword matches because very few videos contain "Jetour" in their metadata |
| Kia Carnival | Weak | No Kia Carnival-specific video surfaces in top 20. Generic car content dominates — Kia has sparse corpus representation for this brand |

---

# Root Cause Analysis

Three systemic bottlenecks explain all failures:

## 1. ANN candidate pool is the hard ceiling

BM25 re-ranks within the 200 ANN candidates. If niche-brand videos (Jetour, Kia) are not semantically close enough to the `0.7×user_emb + 0.3×ctx_emb` query to land in those 200 candidates, BM25 cannot rescue them.

The videos simply are not in the pool to be re-ranked.

## 2. `CTX_ALPHA=0.3` is too conservative for brand-specific queries

The blend is:

```text
0.7×user_emb + 0.3×ctx_emb