//! defer.rs

use core::mem::ManuallyDrop;

/// Defer function (or "drop guard") will be run on the end of scope (the Drop destructor).
#[derive(Debug)]
pub struct Defer<F: FnMut()> {
    f: Option<ManuallyDrop<F>>,
}

impl<F: FnMut()> Defer<F> {
    /// create new Defer function.
    #[inline(always)]
    pub const fn new(f: F) -> Self {
        Self {
            f: Some(ManuallyDrop::new(f)),
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
        if let Some(mut f) = self.f.take() {
            f();
            true
        } else {
            false
        }
    }
}

impl<F: FnMut()> Drop for Defer<F> {
    #[inline(always)]
    fn drop(&mut self) {
        self.run();
    }
}
