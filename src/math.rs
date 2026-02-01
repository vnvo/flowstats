//! Math function wrappers for std/no_std compatibility
//!
//! Uses standard library math when available, falls back to libm for no_std.

#[cfg(feature = "std")]
#[inline]
pub fn ln(x: f64) -> f64 {
    x.ln()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn log2(x: f64) -> f64 {
    x.log2()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn log2(x: f64) -> f64 {
    libm::log2(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn exp2(x: f64) -> f64 {
    x.exp2()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn exp2(x: f64) -> f64 {
    libm::exp2(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn asin(x: f64) -> f64 {
    x.asin()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn asin(x: f64) -> f64 {
    libm::asin(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn ceil(x: f64) -> f64 {
    libm::ceil(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn floor(x: f64) -> f64 {
    libm::floor(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn powi(x: f64, n: i32) -> f64 {
    x.powi(n)
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn powi(x: f64, n: i32) -> f64 {
    libm::pow(x, n as f64)
}
