# Run Scripts

```bash
# 1) Search + deduplication (outputs to generated/)
python3 build_review_lists.py

# Optional: restart from scratch
python3 build_review_lists.py --reset

# 2) Build shortlist for manual review
python3 build_shortlist.py
```

After `build_shortlist.py`:
- `review.md` is created next to `generated/`
- `generated/shortlist_500.json` is updated
