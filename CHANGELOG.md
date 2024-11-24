## Version 0.4.1

- Improve speed of FP math on platforms without FMA at the cost of precision
- Update v_frame dependency to 0.3.8 in order to remove unsound code
- Increase MSRV to 1.64.0
- Factor out math abstractions into new subcrate [`yuvxyb-math`](https://crates.io/crates/yuvxyb-math)
- Refactor code to avoid some proc-macro dependencies

## Version 0.4.0

- Optimize HLG transfer function
- Improve error types (breaking change)
- Improve struct documentation
- Remove unnecessary dependencies

## Version 0.3.1

- Fix some clippy lints, including some which may improve performace a bit
- Bump nalgebra dependency version

## Version 0.3.0

- More math optimizations
- Increase speed and reduce memory usage by converting in place
  - [Breaking] This required that some of the implementations of `From<&T>` be removed.
    The recommendation is that crates which were using these and require the input to
    not be moved should use `from(input.clone())`.
- Add "fastmath" cargo feature, which is on by default, and can be turned off to disable fastmath optimizations
  - This is intended primarily for development and should not be disabled in the real world,
    unless you just like making your crate 5x slower for no good reason.

## Version 0.2.3

- More optimizations to linear/gamma conversion functions
- More optimizations to everything basically

## Version 0.2.2

- Optimizations to linear/gamma conversion functions

## Version 0.2.1

- Add more exports from the crate root

## Version 0.2.0

- Add HSL color space, which can be converted to/from Linear RGB

## Version 0.1.5

- Speed up cube roots by allowing autovectorization

## Version 0.1.4

- Improve performance of conversions between linear RGB and XYB

## Version 0.1.3

- Fix conversions to and from XYB. All previous versions should not be used, as this conversion was very wrong prior to this.

## Version 0.1.2

- Add intermediate structs for `Rgb` and `LinearRgb` to make this crate more versatile

## Version 0.1.1

- Improvements to XYB conversion accuracy

## Version 0.1.0

- Make things faster
- Unofficially declare that the interface is not going to change much after this

## Version 0.0.6

- Fix more conversion bugs

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
