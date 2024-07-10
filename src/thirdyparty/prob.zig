// Code from Cephes,
// translated to zig by tiehuis, member of the zig core team

// SPDX-License-Identifier: MIT
// Copyright (c) 2015-2021 Zig Contributors
// This file is part of [zig](https://ziglang.org/), which is MIT licensed.
// The MIT license requires this copyright notice to be included in all copies
// and substantial portions of the software.
const std = @import("std");
const testing = std.testing;
const math = std.math;

pub const normalDist = @import("prob/normal_dist.zig").normalDist;
pub const inverseNormalDist = @import("prob/normal_dist.zig").inverseNormalDist;
pub const erfc = @import("prob/normal_dist.zig").erfc;
pub const erf = @import("prob/normal_dist.zig").erf;

pub const incompleteGamma = @import("prob/incomplete_gamma.zig").incompleteGamma;
pub const complementedIncompleteGamma = @import("prob/incomplete_gamma.zig").complementedIncompleteGamma;
pub const inverseComplementedIncompleteGamma = @import("prob/incomplete_gamma.zig").inverseComplementedIncompleteGamma;

pub const gamma = @import("prob/gamma.zig").gamma;
pub const lnGamma = @import("prob/gamma.zig").lnGamma;

pub const incompleteBeta = @import("prob/incomplete_beta.zig").incompleteBeta;
pub const inverseIncompleteBeta = @import("prob/incomplete_beta.zig").inverseIncompleteBeta;

pub const polevl = @import("prob/polevl.zig").polevl;
pub const p1evl = @import("prob/polevl.zig").p1evl;

test "math.prob" {
    _ = @import("prob/expx2.zig");
    _ = @import("prob/polevl.zig");

    _ = @import("prob/gamma.zig");
    _ = @import("prob/incomplete_gamma.zig");
    _ = @import("prob/incomplete_beta.zig");
    _ = @import("prob/normal_dist.zig");
}
