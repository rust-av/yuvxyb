## Version 0.0.5

- Bump `v_frame` crate

## Version 0.0.4

- Use the `v_frame` crate instead of rolling our own
- Add `data_mut` method for `Xyb`, useful for some metrics and
  does not violate invariants of frame size
- Minor optimizations

## Version 0.0.3

- Add tests
- Fix as many bugs as possible so that this crate is usable
  - The current state is that _this crate is usable_ but there are slight discrepancies between the expected test values and some of the results. The discrepancies are small enough that this crate should be safe to use for things that do not critically require accuracy (i.e. if you just want some butteraugli metrics), but if you're using this for things that _must_ be color exact, it's better to wait until these discrepancies are resolved.

## Version 0.0.2

- Initial release
- There may be bugs, but the API is setup
