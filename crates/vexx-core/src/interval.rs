//! Interval arithmetic with mandatory outward rounding.
//!
//! Mirrors `python/vexx/interval.py`. Every operation widens its result by one ULP in
//! each direction, so intersection can never reject a value that the real arithmetic
//! admits. `from_numeric_prefix` is the incremental mask used while a number is being
//! decoded: it maps a partial decimal literal to the interval of values it can still grow
//! into (an empty result means the prefix is unreachable).

use crate::num::{next_down, next_up};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

#[allow(clippy::should_implement_trait)] // names mirror `python/vexx/interval.py`
impl Interval {
    pub const EMPTY: Interval = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    pub const ENTIRE: Interval = Interval {
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
    };

    pub fn new(lo: f64, hi: f64) -> Interval {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            Interval::EMPTY
        } else {
            Interval { lo, hi }
        }
    }

    pub fn point(value: f64) -> Interval {
        Interval::new(value, value)
    }

    pub fn is_empty(&self) -> bool {
        self.lo.is_nan() || self.hi.is_nan() || self.lo > self.hi
    }

    pub fn width(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.hi - self.lo
        }
    }

    pub fn mid(&self) -> f64 {
        if self.is_empty() {
            f64::NAN
        } else if self.lo.is_infinite() && self.hi.is_infinite() {
            0.0
        } else if self.lo.is_infinite() {
            self.hi - 1.0
        } else if self.hi.is_infinite() {
            self.lo + 1.0
        } else {
            self.lo + (self.hi - self.lo) / 2.0
        }
    }

    pub fn contains(&self, value: f64) -> bool {
        !self.is_empty() && value >= self.lo && value <= self.hi
    }

    pub fn intersect(self, other: Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return Interval::EMPTY;
        }
        Interval::new(self.lo.max(other.lo), self.hi.min(other.hi))
    }

    pub fn hull(self, other: Interval) -> Interval {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Interval::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }

    pub fn add(self, other: Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return Interval::EMPTY;
        }
        Interval::new(next_down(self.lo + other.lo), next_up(self.hi + other.hi))
    }

    pub fn sub(self, other: Interval) -> Interval {
        self.add(other.neg())
    }

    pub fn neg(self) -> Interval {
        if self.is_empty() {
            return Interval::EMPTY;
        }
        Interval::new(-self.hi, -self.lo)
    }

    pub fn mul(self, other: Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return Interval::EMPTY;
        }
        let products = [
            mul_point(self.lo, other.lo),
            mul_point(self.lo, other.hi),
            mul_point(self.hi, other.lo),
            mul_point(self.hi, other.hi),
        ];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for value in products.iter() {
            if value.is_nan() {
                return Interval::ENTIRE;
            }
            if *value < lo {
                lo = *value;
            }
            if *value > hi {
                hi = *value;
            }
        }
        Interval::new(next_down(lo), next_up(hi))
    }

    pub fn div(self, other: Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return Interval::EMPTY;
        }
        if other.lo <= 0.0 && other.hi >= 0.0 {
            // Division by an interval containing zero is not a single interval; the caller
            // must not use this for contraction (returning ENTIRE keeps it sound).
            return Interval::ENTIRE;
        }
        let quotients = [
            self.lo / other.lo,
            self.lo / other.hi,
            self.hi / other.lo,
            self.hi / other.hi,
        ];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for value in quotients.iter() {
            if *value < lo {
                lo = *value;
            }
            if *value > hi {
                hi = *value;
            }
        }
        Interval::new(next_down(lo), next_up(hi))
    }

    pub fn scale(self, factor: f64) -> Interval {
        self.mul(Interval::point(factor))
    }

    pub fn sqr(self) -> Interval {
        if self.is_empty() {
            return Interval::EMPTY;
        }
        if self.lo >= 0.0 {
            return Interval::new(
                next_down(self.lo * self.lo),
                next_up(self.hi * self.hi),
            );
        }
        if self.hi <= 0.0 {
            return Interval::new(
                next_down(self.hi * self.hi),
                next_up(self.lo * self.lo),
            );
        }
        let peak = self.lo.abs().max(self.hi.abs());
        Interval::new(0.0, next_up(peak * peak))
    }

    pub fn sqrt(self) -> Interval {
        if self.is_empty() || self.hi < 0.0 {
            return Interval::EMPTY;
        }
        let lo = if self.lo <= 0.0 {
            0.0
        } else {
            next_down(self.lo.sqrt())
        };
        Interval::new(lo, next_up(self.hi.sqrt()))
    }

    /// Log of the (finite) volume, used for the SHIM volume twist.
    pub fn log_volume(&self) -> Option<f64> {
        if self.is_empty() || !self.lo.is_finite() || !self.hi.is_finite() {
            return None;
        }
        let width = self.hi - self.lo;
        if width <= 0.0 {
            return Some(f64::NEG_INFINITY);
        }
        Some(width.ln())
    }

    /// Interval of values a partially typed decimal literal can still become.
    ///
    /// `closed` (a delimiter was seen) narrows to the exact parsed value; otherwise the
    /// literal can always grow by one more digit, one more decimal place or a wider integer
    /// magnitude. Returns [`Interval::EMPTY`] when the prefix cannot reach `bounds`.
    pub fn from_numeric_prefix(text: &str, bounds: Interval, allow_exp: bool) -> Interval {
        let trimmed = text.trim_start();
        if trimmed.is_empty() {
            return bounds;
        }
        let mut cut: Option<usize> = None;
        for (index, ch) in trimmed.char_indices() {
            // ``\r`` is JSON whitespace and terminates a number, exactly like ``\n``.
            if ch == ','
                || ch == '}'
                || ch == ']'
                || ch == ' '
                || ch == '\n'
                || ch == '\t'
                || ch == '\r'
            {
                cut = Some(index);
                break;
            }
        }
        let (number_part, closed) = match cut {
            Some(index) => (&trimmed[..index], true),
            None => (trimmed, false),
        };
        if number_part == "-" {
            return bounds.intersect(Interval::new(f64::NEG_INFINITY, 0.0));
        }
        if number_part == "+" {
            return bounds.intersect(Interval::new(0.0, f64::INFINITY));
        }
        let negative = number_part.starts_with('-');
        // Reject leading-zero integer prefixes ("00", "-00", "01", "01.5") and non-JSON
        // spellings ("inf", "nan").  The Python reference rejects them, and the two engines
        // must agree exactly on which prefixes are dead: accepting "00" when the native
        // extension is loaded would let the decoder emit invalid JSON.
        let digits = number_part.trim_start_matches(['-', '+']);
        let int_part = digits.split(['.', 'e', 'E']).next().unwrap_or("");
        if int_part.len() > 1
            && int_part.starts_with('0')
            && int_part.bytes().all(|b| b.is_ascii_digit())
        {
            return Interval::EMPTY;
        }
        if digits
            .bytes()
            .any(|b| b.is_ascii_alphabetic() && b != b'e' && b != b'E')
        {
            return Interval::EMPTY;
        }
        if allow_exp && number_part.to_ascii_lowercase().contains('e') {
            // A scientific-notation prefix is not a point: "1e1" can still become "1e10".
            // ``float("1e")`` fails in Python, so handling this before the parse keeps the
            // two engines in step and lets exponent tokens decode at all.
            let mut parts = number_part.split(['e', 'E']);
            let mantissa_text = parts.next().unwrap_or("");
            let exponent_text = parts.next().unwrap_or("");
            if !valid_mantissa(mantissa_text) || !valid_exponent(exponent_text) {
                return Interval::EMPTY;
            }
            let mantissa = match mantissa_text.parse::<f64>() {
                Ok(value) if value.is_finite() => value,
                _ => return Interval::EMPTY,
            };
            if closed && !exponent_text.is_empty() {
                return match number_part.parse::<f64>() {
                    Ok(value) if value.is_finite() => bounds.intersect(Interval::point(value)),
                    _ => Interval::EMPTY,
                };
            }
            return exp_prefix_span(mantissa, exponent_text, bounds);
        }
        let clean = number_part.strip_suffix('.').unwrap_or(number_part);
        let base = match clean.parse::<f64>() {
            Ok(value) if value.is_finite() => value,
            _ => return Interval::EMPTY,
        };
        if closed {
            return bounds.intersect(Interval::point(base));
        }
        if number_part.contains('.') {
            let dot = number_part.find('.').unwrap_or(number_part.len());
            let fraction_digits = number_part.len().saturating_sub(dot + 1);
            let ulp = 10f64.powi(-(fraction_digits as i32));
            let span = if negative {
                let lo = if ulp > 0.0 {
                    next_up(base - ulp)
                } else {
                    base
                };
                Interval::new(lo, base)
            } else {
                Interval::new(base, next_down(base + ulp))
            };
            return bounds.intersect(span);
        }
        if number_part == "0" || number_part == "-0" {
            let span = if negative {
                Interval::new(next_up(-1.0), 0.0)
            } else {
                Interval::new(0.0, next_down(1.0))
            };
            return bounds.intersect(span);
        }
        // Non-zero integer prefix: every wider magnitude is reachable, so union the
        // half-open decades the literal can still turn into.
        let magnitude = base.abs();
        let mut union = Interval::EMPTY;
        let mut scale = 1.0f64;
        for _ in 0..16 {
            let span = if negative {
                let lo = -((magnitude + 1.0) * scale);
                let hi = -(magnitude * scale);
                let lo = if lo == f64::NEG_INFINITY { lo } else { next_up(lo) };
                Interval::new(lo, hi)
            } else {
                let lo = magnitude * scale;
                let hi = (magnitude + 1.0) * scale;
                let hi = if hi == f64::INFINITY { hi } else { next_down(hi) };
                Interval::new(lo, hi)
            };
            let sub = bounds.intersect(span);
            if !sub.is_empty() {
                union = union.hull(sub);
            }
            scale *= 10.0;
            if !negative && bounds.hi.is_finite() && magnitude * scale > bounds.hi {
                break;
            }
            if negative && bounds.lo.is_finite() && -magnitude * scale < bounds.lo {
                break;
            }
        }
        union
    }
}

/// JSON mantissa syntax: optional sign, no leading zero, optional fraction.
fn valid_mantissa(text: &str) -> bool {
    let body = text.trim_start_matches(['-', '+']);
    if body.is_empty() {
        return false;
    }
    let mut parts = body.split('.');
    let integer = parts.next().unwrap_or("");
    if integer.is_empty() || !integer.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    if integer.len() > 1 && integer.starts_with('0') {
        return false;
    }
    match parts.next() {
        None => true,
        Some(fraction) => {
            !fraction.is_empty()
                && fraction.bytes().all(|b| b.is_ascii_digit())
                && parts.next().is_none()
        }
    }
}

/// Exponent syntax: empty, an optional single sign, then digits only.
fn valid_exponent(text: &str) -> bool {
    let mut chars = text.chars();
    match chars.next() {
        None => true,
        Some(first) if first == '+' || first == '-' => chars.all(|c| c.is_ascii_digit()),
        Some(first) => first.is_ascii_digit() && chars.all(|c| c.is_ascii_digit()),
    }
}

/// Values reachable by completing a scientific-notation prefix (Python parity).
fn exp_prefix_span(mantissa: f64, exponent_text: &str, bounds: Interval) -> Interval {
    if mantissa == 0.0 {
        return bounds.intersect(Interval::point(0.0));
    }
    let mut prefix = exponent_text;
    let mut negative_exponent = false;
    if let Some(rest) = prefix.strip_prefix('-') {
        negative_exponent = true;
        prefix = rest;
    } else if let Some(rest) = prefix.strip_prefix('+') {
        prefix = rest;
    }
    let mut matched = false;
    for exponent in -323i32..=308 {
        let spelling = if exponent < 0 {
            if !negative_exponent {
                continue;
            }
            (-exponent).to_string()
        } else {
            if negative_exponent {
                continue;
            }
            exponent.to_string()
        };
        if !prefix.is_empty() && !spelling.starts_with(prefix) {
            continue;
        }
        let value = mantissa * 10f64.powi(exponent);
        if !value.is_finite() {
            continue;
        }
        if bounds.lo <= value && value <= bounds.hi {
            matched = true;
            break;
        }
    }
    if !matched {
        // An open exponent can still gain a minus sign and enough digits to underflow to
        // 0.0 (``"1e"`` -> ``"1e-999"``); that completion is representable, so a domain
        // containing zero is viable even when the sampled exponents all missed it.
        let zero_reachable = exponent_text.is_empty() || exponent_text.starts_with('-');
        if zero_reachable && bounds.lo <= 0.0 && 0.0 <= bounds.hi {
            matched = true;
        }
    }
    if !matched {
        return Interval::EMPTY;
    }
    if mantissa > 0.0 {
        bounds.intersect(Interval::new(0.0, f64::INFINITY))
    } else {
        bounds.intersect(Interval::new(f64::NEG_INFINITY, 0.0))
    }
}

fn mul_point(a: f64, b: f64) -> f64 {
    if (a == 0.0 && b.is_infinite()) || (b == 0.0 && a.is_infinite()) {
        0.0
    } else {
        a * b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outward_rounding_keeps_exact_endpoints() {
        let half = Interval::new(0.5, 0.5);
        let sum = Interval::new(0.5, 0.5).add(half);
        assert!(sum.contains(1.0));
        assert!(sum.width() < 1e-15);
    }

    #[test]
    fn numeric_prefix_rejects_the_same_dead_prefixes_as_python() {
        let bounds = Interval::new(0.0, 10.0);
        for dead in ["01.5", "00", "-00", "01", "inf", "NaN", "1e400"] {
            assert!(
                Interval::from_numeric_prefix(dead, bounds, true).is_empty(),
                "{dead} should be dead"
            );
        }
        assert!(!Interval::from_numeric_prefix("1e", Interval::new(0.0, 1000.0), true).is_empty());
        let big = Interval::new(1e9, 1e11);
        assert!(!Interval::from_numeric_prefix("1e1", big, true).is_empty());
        assert!(Interval::from_numeric_prefix("1e2", big, true).is_empty());
    }

    #[test]
    fn numeric_prefix_terminators_and_underflow_agree_with_python() {
        // ``\r`` is JSON whitespace: the literal is closed, so it pins exactly.
        let closed = Interval::from_numeric_prefix("342\r", Interval::ENTIRE, false);
        assert_eq!(closed, Interval::point(342.0));
        // An open negative exponent can underflow to zero, which is representable.
        let zero = Interval::from_numeric_prefix("1e-", Interval::new(-0.5, 0.0), true);
        assert!(zero.contains(0.0), "1e- can still become 1e-999 -> 0.0");
        // An exponent whose sign is already positive cannot reach zero.
        assert!(Interval::from_numeric_prefix("1e+", Interval::new(-0.5, 0.0), true).is_empty());
    }

    #[test]
    fn arithmetic_is_sound_at_the_edges() {
        let a = Interval::new(-2.0, 3.0);
        let b = Interval::new(4.0, 5.0);
        let product = a.mul(b);
        assert!(product.contains(-8.0));
        assert!(product.contains(15.0));
        assert!(Interval::new(0.0, 4.0).sqrt().contains(2.0));
        assert!(Interval::new(-3.0, -1.0).sqr().contains(9.0));
        assert!(Interval::new(-3.0, 1.0).sqr().contains(9.0));
        assert!(Interval::new(-3.0, 1.0).sqr().contains(0.0));
        assert!(Interval::new(1.0, 2.0).div(Interval::new(2.0, 4.0)).contains(0.5));
        assert!(Interval::new(1.0, 2.0).div(Interval::new(2.0, 4.0)).contains(1.0));
        assert!(Interval::new(1.0, 2.0).div(Interval::new(-1.0, 1.0)) == Interval::ENTIRE);
    }

    #[test]
    fn decimal_prefixes_narrow() {
        let bounds = Interval::new(0.0, 1.0);
        assert!(Interval::from_numeric_prefix("0", bounds, false).contains(0.5));
        assert!(Interval::from_numeric_prefix("0.3", bounds, false).contains(0.35));
        assert!(!Interval::from_numeric_prefix("0.3", bounds, false).contains(0.4));
        assert!(Interval::from_numeric_prefix("0.3", Interval::new(0.4, 0.5), false).is_empty());
        // A closed literal pins the value.
        let closed = Interval::from_numeric_prefix("0.25,", bounds, false);
        assert!(closed.contains(0.25) && closed.width() == 0.0);
        assert!(Interval::from_numeric_prefix("2", Interval::new(0.0, 1.0), false).is_empty());
        let unbounded = Interval::from_numeric_prefix("-", Interval::ENTIRE, false);
        assert!(unbounded.contains(-0.5) && unbounded.contains(0.0));
        assert!(!unbounded.contains(0.5));
        assert_eq!(Interval::from_numeric_prefix("abc", bounds, false), Interval::EMPTY);
    }

    #[test]
    fn log_volume_matches_width() {
        let volume = Interval::new(1.0, 3.0).log_volume().unwrap();
        assert!((volume - 2f64.ln()).abs() < 1e-12);
        assert!(Interval::ENTIRE.log_volume().is_none());
    }
}
