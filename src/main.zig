const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.rand.Random;

fn p_rand(comptime F: type, rand: Random, d: usize, t: usize, r: usize, monte_iterations: usize) F {
    if (d == 0)
        return if (t == 0 and r == 0) 1 else 0;

    var n_correct_roll_count: usize = 0;
    for (0..monte_iterations) |_| {
        var running_total: usize = 0;
        var running_roll_count: usize = 0;
        while (running_total < t) : (running_roll_count += 1) {
            const roll = rand.intRangeAtMostBiased(usize, 1, d);
            running_total += roll;
        }
        n_correct_roll_count += @intFromBool(running_roll_count == r);
    }
    const denominator: F = @floatFromInt(monte_iterations);
    const numerator: F = @floatFromInt(n_correct_roll_count);
    return numerator / denominator;
}

fn p_recursive(comptime F: type, d: usize, t: usize, r: usize) F {
    if (d == 0 or r == 0 or t == 0)
        return if (t == 0 and r == 0) 1 else 0;

    const df: F = @floatFromInt(d);
    const d_inv = 1 / df;

    if (r == 1) {
        const numerator: F = @floatFromInt(d - @min(t - 1, d));
        return d_inv * numerator;
    }

    var total: F = 0;
    for (1..@min(t, d) + 1) |i|
        total += p_recursive(F, d, t - i, r - 1) * d_inv;

    return total;
}

test "recursive matches monte carlo" {
    var prng = std.rand.DefaultPrng.init(0);
    const rand = prng.random();

    const F = f32;
    for (0..5) |t| {
        for (0..4) |d| {
            for (0..t + 2) |r| {
                const monte = p_rand(F, rand, d, t, r, 1000);
                const recursive = p_recursive(F, d, t, r);
                try std.testing.expectApproxEqAbs(monte, recursive, 3e-2);
            }
        }
    }
}

// Basic dynamic programming to compute the probability that
// it takes exactly r rolls of a d-sided die to meet or exceed
// a total t
pub fn PDynamic(comptime F: type) type {
    return struct {
        buf: []F,
        cumsum_buf: []F,
        d: usize,
        t: usize,
        d_inv: F,
        next_r: usize,
        allocator: Allocator,

        // d-sided die, trying to meet or exceed total t
        pub fn init(allocator: Allocator, d: usize, t: usize) !@This() {
            var buf = try allocator.alloc(F, t + 1);
            errdefer allocator.free(buf);

            var cumsum_buf = try allocator.alloc(F, t + 2);
            errdefer allocator.free(cumsum_buf);

            const df: F = @floatFromInt(d);
            const d_inv = 1 / df;

            return @This(){
                .buf = buf,
                .cumsum_buf = cumsum_buf,
                .d = d,
                .t = t,
                .d_inv = d_inv,
                .next_r = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *const @This()) void {
            defer self.allocator.free(self.buf);
            defer self.allocator.free(self.cumsum_buf);
        }

        // returns probability for 0 rolls, then 1 roll, then 2 rolls, ...
        //
        // this is a pmf (probability mass function) -- probability that it
        // took _exactly_ that number of rolls to meet or exceed the total
        pub fn next(self: *@This()) F {
            defer self.next_r += 1;

            if (self.d == 0 or self.next_r == 0 or self.t == 0)
                return if (self.t == 0 and self.next_r == 0) 1 else 0;

            if (self.next_r == 1) {
                self.buf[0] = 0;
                for (self.buf[1..], 1..) |*b, t| {
                    const numerator: F = @floatFromInt(self.d - @min(t - 1, self.d));
                    b.* = self.d_inv * numerator;
                }
                return self.buf[self.buf.len - 1];
            }

            self.cumsum_buf[0] = 0;
            for (1..self.cumsum_buf.len) |i|
                self.cumsum_buf[i] = self.cumsum_buf[i - 1] + self.buf[i - 1];

            self.buf[0] = 0;
            for (self.buf[1..], 1..) |*b, t|
                b.* = self.d_inv * (self.cumsum_buf[t] - self.cumsum_buf[t - @min(t, self.d)]);

            return self.buf[self.buf.len - 1];
        }

        pub fn to_owned_array(self: *@This()) ![]F {
            if (self.next_r != 0)
                return error.AlreadyStartedIteration;
            var rtn = try self.allocator.alloc(F, self.t + 1);
            errdefer self.allocator.free(rtn);
            for (rtn) |*r|
                r.* = self.next();
            return rtn;
        }
    };
}

test "recursive matches actual" {
    const allocator = std.testing.allocator;
    const F = f32;
    for (0..5) |t| {
        for (0..4) |d| {
            var dyn = try PDynamic(F).init(allocator, d, t);
            defer dyn.deinit();
            for (0..t + 2) |r| {
                const recursive = p_recursive(F, d, t, r);
                const dynamic = dyn.next();
                std.testing.expectApproxEqAbs(recursive, dynamic, 1e-5) catch |err| {
                    std.debug.print("error with (d,t,r) == ({}, {}, {})\n", .{ d, t, r });
                    return err;
                };
            }
        }
    }
}

fn binarr(comptime F: type, allocator: Allocator, k: usize) ![]F {
    var rtn = try allocator.alloc(F, k);
    errdefer allocator.free(rtn);

    if (k == 0)
        return rtn;
    rtn[0] = 1;

    if (k == 1)
        return rtn;
    const fk: F = @floatFromInt(k);
    rtn[1] = (fk - 1);

    for (2..rtn.len) |i| {
        const fi: F = @floatFromInt(i);
        rtn[i] = rtn[i - 1] * (fk - fi) / fi;
    }

    return rtn;
}

fn powers(comptime F: type, allocator: Allocator, x: F, n: usize) ![]F {
    // powers 1 .. 1+n
    var rtn = try allocator.alloc(F, n);
    errdefer allocator.free(rtn);

    if (n == 0)
        return rtn;
    rtn[0] = x;

    if (n == 1)
        return rtn;
    rtn[1] = x * x;

    if (n == 2)
        return rtn;

    for (2..rtn.len) |i| {
        const a = (i - 1) >> 1;
        const b = (i - 1) - a;
        rtn[i] = rtn[a] * rtn[b];
    }

    return rtn;
}

pub fn with_zeros(comptime F: type, allocator: Allocator, k: usize, p: F, pma: []const F) ![]F {
    // converts a probability mass array `pma` to what would have been produced
    // if the same process had a `p` chance of including a zero before sampling
    // from `pma`.
    //
    // Note: this is necessarily an infinite array, and an intermediate computation has
    // a convolution over an infinitely sized buffer. The parameter `k` truncates the
    // array to include at most `k` turns (note the 0th array element representing the pma
    // before any turns have executed is always included).
    var rtn = try allocator.alloc(F, k + 1);
    errdefer allocator.free(rtn);

    if (pma.len < 1)
        return error.MissingZerothTurnProbability;

    if (p < 0 or p > 1)
        return error.ZeroProbNotAProbability;

    {
        var total: F = 0;
        for (pma) |x| {
            if (x < 0 or x > 1)
                return error.PMANotAProbability;
            total += x;
        }
        const diff = total - 1;
        const adiff = @max(diff, -diff);
        if (adiff > 1e-5)
            return error.PMADoesNotSumToOne;
    }

    if (k == 0) {
        rtn[0] = pma[0];
        return rtn;
    }

    for (rtn) |*r|
        r.* = 0;

    if (pma[0] != 0) {
        rtn[0] = 1;
        return rtn;
    }

    if (p == 0) {
        const i = @min(pma.len, rtn.len);
        for (rtn[0..i], pma[0..i]) |*r, x|
            r.* = x;
        return rtn;
    }

    const pow = try powers(F, allocator, (1 - p) / p, k);
    defer allocator.free(pow);

    var buf_sum: F = 0;
    {
        const bin = try binarr(F, allocator, k);
        defer allocator.free(bin);

        const fk: F = @floatFromInt(k);
        const pk = @exp(@log(p) * fk);
        for (pow, bin, 1..) |*buf, bc, i| {
            const pma_val = if (i < pma.len) pma[i] else 0;
            buf.* = buf.* * bc * pk * pma_val;
            buf_sum += buf.*;
        }
        rtn[rtn.len - 1] = buf_sum;
    }

    for (2..k) |i| {
        const a: F = @floatFromInt((k + 1) - i);
        buf_sum = 0;
        for (0..(pow.len + 1) - i) |j| {
            const jf: F = @floatFromInt(j);
            const z: F = (a - jf) / a / p * pow[j];
            pow[j] = z;
            buf_sum += z;
        }
        rtn[rtn.len - i] = buf_sum;
    }

    rtn[1] = pma[1] * (1 - p);

    return rtn;
}

test "zero mixin works on a small example" {
    const F = f32;
    const probs = [_]F{ 0, 0.5, 0.5, 0 };
    const allocator = std.testing.allocator;
    var z = try with_zeros(F, allocator, 10, 0.5, probs[0..]);
    defer allocator.free(z);
    var expected_arr = [_]F{ 0, 0.25, 0.25, 0.1875, 0.125, 0.078125, 0.046875, 0.02734375, 0.015625, 0.0087890625, 0.0048828125 };
    var expected: []F = expected_arr[0..];
    try std.testing.expectEqualDeep(expected, z);
}

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
