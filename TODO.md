# TODO reminders for myself

## TODOs

1. Refactor global/local replacement & build_attribution graph, prune, viz to make the structure cleaner. Especially replacement and building graphs. We then do this using the refactored code: Show the comparison of raw/global/local model when building the attribution graph.
2. Draw a Venn Map -- each circle is a pair of operand, and use this to show the actual overlap between many pairs. Should be super helpful!
3. Comparison for addition in raw LLMs.
4. Add more information to the graph (e.g. incoming/outgoing edges of nodes, histograms).

## Notes

1. Lazy decoder loading is slow for *repeated* forwards — each `decode` re-reads `W_dec` from disk. Fix: load with `lazy_decoder=False` to keep decoders resident; Qwen3-4B fits on the 80 GB A100 (~68/85 GB) and forwards get ~500x faster (the progressive/steering jobs dropped 43 min → 5 min). The intervention/steering example scripts auto-try eager loading with a lazy fallback on OOM.
