//! defer.rs

/// Defer function (or "drop guard") will be run on the end of scope (the Drop destructor).
#[derive(Debug)]
pub struct Defer<F: FnOnce()> {
    f: Option<F>,
}

impl<F: FnOnce()> Defer<F> {
    /// create new Defer function.
    #[inline(always)]
    pub const fn new(f: F) -> Self {
        Self {
            f: Some(f),
        }
    }

    /// cancel the defer. it will not to run anyway.
    ///
    /// do nothing if the function already called.
    #[inline(always)]
    pub fn cancel(&mut self) {
        self.f.take();
    }

    /// calls the defer function early. and it will not be run if dropped later.
    ///
    /// do nothing if it has been canceled or already called the defer function.
    #[inline(always)]
    pub fn run(&mut self) -> bool {
        if let Some(f) = self.f.take() {
            f();
            true
        } else {
            false
        }
    }
}

impl<F: FnOnce()> Drop for Defer<F> {
    #[inline(always)]
    fn drop(&mut self) {
        self.run();
    }
}
