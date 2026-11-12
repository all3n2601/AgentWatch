# Median contract

## Odd number of values

`median` sorts its values and returns the middle one.

## Even number of values

`median` must return the mean of the two middle values for an even number of values. Taking
either middle value alone would favor one half of the readings, so the reported median would
depend on which half happened to be larger.

## Input

`median` never modifies its input; it sorts a copy.
