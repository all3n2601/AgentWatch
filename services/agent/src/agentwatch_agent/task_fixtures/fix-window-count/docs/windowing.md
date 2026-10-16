# Windowing contract

## Window count

A sequence of n values has n - size + 1 windows of a given size when size is at most n, and no
windows otherwise. The first window starts at position 0 and the final window ends at the last
value, so no reading is ever skipped.

## Invalid sizes

A size of zero or less raises ValueError.
