const std = @import("std");

const exp_underflow = -745.1332191019412076235;
const expm2 = @exp(-2.0);
const sqrt2pi = @sqrt(2 * std.math.pi);

pub fn probability(x: f64) f64 {
    const z = x * std.math.sqrt1_2;
    const t = @abs(z);
    if (t < 1) {
        return 0.5 + 0.5 * erf(z);
    } else {
        const y = 0.5 * erfc(t);
        return if (z <= 0) y else 1 - y;
    }
}

pub fn survival(x: f64) f64 {
    const z = x * std.math.sqrt1_2;
    const t = @abs(z);
    if (t < 1) {
        return 0.5 - 0.5 * erf(z);
    } else {
        const y = 0.5 * erfc(t);
        return if (z > 0) y else 1 - y;
    }
}

pub fn quantile(p: f64) f64 {
    if (p <= 0) {
        return -std.math.inf(f64);
    }
    if (p >= 1) {
        return std.math.inf(f64);
    }

    const flip = p > 1 - expm2;
    const y = if (flip) 1 - p else p;

    if (y > expm2) {
        const w = y - 0.5;
        const w2 = w * w;
        const num = polynomial(w2, &inverse_num);
        const den = polynomial(w2, &inverse_den);
        const x = w + w * w2 * num / den;
        return sqrt2pi * x;
    }

    const z = @sqrt(2 * -@log(y));
    const invz = 1 / z;

    const num, const den = if (z < 8) .{
        polynomial(invz, &inverse_high_num),
        polynomial(invz, &inverse_high_den),
    } else .{
        polynomial(invz, &inverse_low_num),
        polynomial(invz, &inverse_low_den),
    };

    const x0 = z - @log(z) / z;
    const x1 = invz * num / den;
    return if (flip) x0 - x1 else x1 - x0;
}

// |x| < 1 only
fn erf(x: f64) f64 {
    const z = x * x;
    const num = polynomial(z, &erf_num);
    const den = polynomial(z, &erf_den);
    return x * num / den;
}

// x >= 1 only
fn erfc(x: f64) f64 {
    const t = @abs(x);
    const z = -x * x;

    if (z < exp_underflow) {
        return if (x < 0) 2 else 0;
    }

    const num, const den = if (t < 8) .{
        polynomial(t, &erfc_low_num),
        polynomial(t, &erfc_low_den),
    } else .{
        polynomial(t, &erfc_high_num),
        polynomial(t, &erfc_high_den),
    };

    return @exp(z) * num / den;
}

// c_0 x^n + c_1 x^n-1 + ... + c_n-1 x + c_n
fn polynomial(x: f64, coeffs: []const f64) f64 {
    @setRuntimeSafety(false);
    var acc = coeffs[0];
    for (coeffs[1..]) |c| {
        acc *= x;
        acc += c;
    }
    return acc;
}

const erf_num = [_]f64{
    9.60497373987051638749e0,
    9.00260197203842689217e1,
    2.23200534594684319226e3,
    7.00332514112805075473e3,
    5.55923013010394962768e4,
};

const erf_den = [_]f64{
    1.00000000000000000000e0,
    3.35617141647503099647e1,
    5.21357949780152679795e2,
    4.59432382970980127987e3,
    2.26290000613890934246e4,
    4.92673942608635921086e4,
};

const erfc_low_num = [_]f64{
    2.46196981473530512524e-10,
    5.64189564831068821977e-1,
    7.46321056442269912687e0,
    4.86371970985681366614e1,
    1.96520832956077098242e2,
    5.26445194995477358631e2,
    9.34528527171957607540e2,
    1.02755188689515710272e3,
    5.57535335369399327526e2,
};

const erfc_low_den = [_]f64{
    1.00000000000000000000e0,
    1.32281951154744992508e1,
    8.67072140885989742329e1,
    3.54937778887819891062e2,
    9.75708501743205489753e2,
    1.82390916687909736289e3,
    2.24633760818710981792e3,
    1.65666309194161350182e3,
    5.57535340817727675546e2,
};

const erfc_high_num = [_]f64{
    5.64189583547755073984e-1,
    1.27536670759978104416e0,
    5.01905042251180477414e0,
    6.16021097993053585195e0,
    7.40974269950448939160e0,
    2.97886665372100240670e0,
};

const erfc_high_den = [_]f64{
    1.00000000000000000000e0,
    2.26052863220117276590e0,
    9.39603524938001434673e0,
    1.20489539808096656605e1,
    1.70814450747565897222e1,
    9.60896809063285878198e0,
    3.36907645100081516050e0,
};

const inverse_num = [_]f64{
    -5.99633501014107895267e1,
    9.80010754185999661536e1,
    -5.66762857469070293439e1,
    1.39312609387279679503e1,
    -1.23916583867381258016e0,
};

const inverse_den = [_]f64{
    1.00000000000000000000e0,
    1.95448858338141759834e0,
    4.67627912898881538453e0,
    8.63602421390890590575e1,
    -2.25462687854119370527e2,
    2.00260212380060660359e2,
    -8.20372256168333339912e1,
    1.59056225126211695515e1,
    -1.18331621121330003142e0,
};

const inverse_high_num = [_]f64{
    4.05544892305962419923e0,
    3.15251094599893866154e1,
    5.71628192246421288162e1,
    4.40805073893200834700e1,
    1.46849561928858024014e1,
    2.18663306850790267539e0,
    -1.40256079171354495875e-1,
    -3.50424626827848203418e-2,
    -8.57456785154685413611e-4,
};

const inverse_high_den = [_]f64{
    1.00000000000000000000e0,
    1.57799883256466749731e1,
    4.53907635128879210584e1,
    4.13172038254672030440e1,
    1.50425385692907503408e1,
    2.50464946208309415979e0,
    -1.42182922854787788574e-1,
    -3.80806407691578277194e-2,
    -9.33259480895457427372e-4,
};

const inverse_low_num = [_]f64{
    3.23774891776946035970e0,
    6.91522889068984211695e0,
    3.93881025292474443415e0,
    1.33303460815807542389e0,
    2.01485389549179081538e-1,
    1.23716634817820021358e-2,
    3.01581553508235416007e-4,
    2.65806974686737550832e-6,
    6.23974539184983293730e-9,
};

const inverse_low_den = [_]f64{
    1.00000000000000000000e0,
    6.02427039364742014255e0,
    3.67983563856160859403e0,
    1.37702099489081330271e0,
    2.16236993594496635890e-1,
    1.34204006088543189037e-2,
    3.28014464682127739104e-4,
    2.89247864745380683936e-6,
    6.79019408009981274425e-9,
};
