# Statistics contract

## Totals

`total` returns the sum of its values. The total of an empty input is 0, because adding
nothing is a real, measured result.

## Mean of an empty input

`mean` must return None for an empty input. An empty input has no mean: returning 0 would
report a measured average of zero that never happened, and raising would crash reports that
legitimately contain no data. Callers display None as "no data".

## Rounding

Neither function rounds. Callers format values for display.
