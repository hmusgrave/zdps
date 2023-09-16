# zdps

dps calculations accounting for overkill

## Purpose

In popular games (MMOs, D&D, ...), there's often a notion of chance-to-hit followed by rolling some sort of fair die to determine damage. DPS (damage per second) estimates usually don't take the enemy health into account, but that drastically overestimates effectiveness when max hits are within a small factor of enemy health due to overkill (e.g., if you're rolling 1d100 attacking an enemy with 1 health, you'll only ever do 1 damage, not an average of 50.5).

This library uses basic dynamic programming techniques to account for overkill. In particular, for each posible roll_count it gives you the probability it'll take exactly that many rolls to meet or exceed the enemy health.

## Examples

The following demonstrates the entire API surface, including appropriate allocation/deallocation to indicate who owns what when. The main interface is `PDynamic`, which allows you to iterate over a probability mass function.

Most callers don't actually need the full `with_zeros` probability mass function. Consider something like a D&D encounter where you have a 15% chance to hit and then deal 1d6 damage. If (ignoring the chance to hit, just using the result of `PDynamic`) you take 8 rolls on average to kill the monster then when counting the chance to hit you instead take 8 / 0.15 = 53.3 rolls on average. That summary statistic didn't require the full distribution.

```zig
test "readme" {
    const allocator = std.testing.allocator;

    const F = f64;
    const die_sides: usize = 6;
    const target_total: usize = 10;

    // Initialize the dynamic programming problem allowing you
    // to iterate over the probability mass function
    var iter = try PDynamic(F).init(allocator, die_sides, target_total);
    defer iter.deinit();

    // Iterate over all roll_counts which might not be
    // zero and fill a newly created array with them
    const pma = try iter.to_owned_array();
    defer allocator.free(pma);

    // You can also iterate lazily and find the
    // probability that the target will be reached
    // or exceeed in exactly a given roll_count
    // (iterating further results in probs of 0).
    //
    // for (0..target_total + 1) |roll_count| {
    //     const prob = iter.next();
    //     // then do something with prob, roll_count
    // }

    // If there is a non-zero chance of rolling a 0, the
    // resulting probability mass function has infinitely
    // many terms, and some of the intermediate computations
    // convolve over infinitely many terms. This truncates
    // the output to roll_counts maxing at 30 (31 elements
    // in the returned array, since we provide the probability
    // for reaching the target in zero rolls in the 0th entry).
    const max_roll_count: usize = 30;

    // The resulting probability mass array will be as if instead
    // of just rolling a die, you first flipped a biased coin.
    // With probability prob_zero you generate a 0 rather than
    // rolling the die.
    const prob_zero: F = 0.3;

    // Convert the probability mass array we computed earlier for a 6-sided
    // die to what we would expect if there were a 30% chance of outputting
    // 0 rather than always rolling 1d6.
    const zero_pma = try with_zeros(F, allocator, max_roll_count, prob_zero, pma);
    defer allocator.free(zero_pma);

    // If your probability of zero is not too high, a small constant
    // factor of the original pma length should suffice to capture
    // almost all the total probability available in the infinite
    // sequence described above.
    var total_prob: F = 0;
    for (zero_pma) |p|
        total_prob += p;
    try std.testing.expectApproxEqAbs(@as(F, 1), total_prob, 1e-9);
}
```

## Status

This works and is reasonably efficient, but the `with_zeros` computations are numerically unstable and prone to giving `nan` values when projected out to 500+ turns, even with `f128` intermediate values.
