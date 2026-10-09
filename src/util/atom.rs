//! Atom is a wrapper of [`sdd::AtomicShared`] for easy to use it.

#![allow(unsafe_code)]

use crate::Arc;

use sdd::{AtomicShared, Guard, Shared, Tag};

use core::sync::atomic::Ordering::{AcqRel, Acquire};

/// Atom is just a wrapper of [`sdd::AtomicShared`] for easy to use it.
/// * this does not implements Clone due to AtomicShared::clone is copied just pointer but not the concurrent container itself. if you need to share it between multi threads, you can use sdd::Shared to wrapping it, or use `Box::leak` (static lifetime) to get a Clone-able reference.
/// * for making this type can be constructed in const context, so it does not use Arc to implements Clone.
pub struct Atom<T> {
    /// the wrapped inner AtomicShared.
    pub inner: AtomicShared<T>,
}

/// Atom wrapped by sdd::Shared.
pub type SharedAtom<T> = Shared<Atom<T>>;

unsafe impl<T: Send> Send for Atom<T> {}
unsafe impl<T: Sync> Sync for Atom<T> {}

impl<T: core::fmt::Debug> core::fmt::Debug for Atom<T> {
    #[inline(always)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        let g = Guard::new();
        let val = self.get_ref(&g);
        f.debug_tuple("Atom").field(&val).finish()
    }
}
impl<T: core::fmt::Display> core::fmt::Display for Atom<T> {
    #[inline(always)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        let g = Guard::new();
        if let Some(val) = self.get_ref(&g) {
            write!(f, "{}", val)
        } else {
            Err(core::fmt::Error)
        }
    }
}

/*
#[cfg(feature="serde")]
mod _serde_impl {
    use super::*;

    use serde::{Serialize, Serializer};
    use serde::{Deserialize, Deserializer};

    impl<T: Serialize> Serialize for Atom<T> {
        #[inline(always)]
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let g = Guard::new();
            if let Some(val) = self.get_ref(&g) {
                val.serialize(serializer)
            } else {
                use serde::ser::Error;
                Err(S::Error::custom("Atom<T> does not have the value!"))
            }
        }
    }

    impl<'d, T: Deserialize<'d>> Deserialize<'d> for Atom<T> {
        #[inline(always)]
        fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Self, D::Error> {
            let val = T::deserialize(deserializer)?;
            Ok(Self::new(val))
        }
    }
}
*/

impl<T> From<&Atom<T>> for Option<Shared<T>> {
    #[inline(always)]
    fn from(val: &Atom<T>) -> Option<Shared<T>> {
        val.get()
    }
}

impl<T: PartialEq> PartialEq for Atom<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        let g = Guard::new();
        match (self.get_ref(&g), other.get_ref(&g)) {
            // both has value
            (Some(x), Some(y)) => {
                x.eq(y)
            },

            // both has no value
            (None, None) => true,

            // (Some, None) or (None, Some)
            // Null != NonNull
            _ => false,
        }
    }
}
impl<T: Eq> Eq for Atom<T> {}

impl<T: PartialOrd> PartialOrd for Atom<T> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        let g = Guard::new();
        match (self.get_ref(&g), other.get_ref(&g)) {
            // both has value
            (Some(x), Some(y)) => {
                x.partial_cmp(y)
            },

            // both has no value
            (None, None) => Some(core::cmp::Ordering::Equal),

            // (Some, None) or (None, Some)
            // NonNull > Null
            (Some(_), None) => Some(core::cmp::Ordering::Greater),

            // (None, Some)
            // Null < NonNull
            _ => Some(core::cmp::Ordering::Less)
        }
    }
}
impl<T: Ord> Ord for Atom<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        let g = Guard::new();
        match (self.get_ref(&g), other.get_ref(&g)) {
            // both has value
            (Some(x), Some(y)) => {
                x.cmp(y)
            },

            // both has no value
            (None, None) => core::cmp::Ordering::Equal,

            // Sime > None
            (Some(_), None) => core::cmp::Ordering::Greater,

            // (None, Some)
            // Null < NonNull
            _ => core::cmp::Ordering::Less
        }
    }
}

impl<T: core::hash::Hash> core::hash::Hash for Atom<T> {
    #[inline(always)]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        if let Some(shared) = self.get() {
            Some(&*shared).hash(state)
        } else {
            None::<T>.hash(state)
        }
    }
}

impl<T: 'static> From<Option<T>> for Atom<T> {
    #[inline(always)]
    fn from(maybe_val: Option<T>) -> Self {
        match maybe_val {
            Some(val) => Self::new(val),
            _ => Self::init(),
        }
    }
}

impl<T: 'static> From<T> for Atom<T> {
    #[inline(always)]
    fn from(val: T) -> Self {
        Self::new(val)
    }
}
impl<T: 'static + Clone> From<&T> for Atom<T> {
    #[inline(always)]
    fn from(val: &T) -> Self {
        val.clone().into()
    }
}

impl<T> From<Shared<T>> for Atom<T> {
    #[inline(always)]
    fn from(val: Shared<T>) -> Self {
        Self::new_shared(val)
    }
}
impl<T> From<&Shared<T>> for Atom<T> {
    #[inline(always)]
    fn from(val: &Shared<T>) -> Self {
        val.clone().into()
    }
}

impl<T: 'static> From<Box<T>> for Atom<T> {
    #[inline(always)]
    fn from(val: Box<T>) -> Self {
        let val: T = *val;
        Self::new(val)
    }
}
impl<T: 'static + Clone> From<&Box<T>> for Atom<T> {
    #[inline(always)]
    fn from(box_val: &Box<T>) -> Self {
        box_val.clone().into()
    }
}

impl<T> Default for Atom<T> {
    #[inline(always)]
    fn default() -> Self {
        Self::init()
    }
}

impl<T> From<Atom<T>> for SharedAtom<T> {
    #[inline(always)]
    fn from(val: Atom<T>) -> SharedAtom<T> {
        val.clonable()
    }
}

impl<T> Atom<T> {
    /// initialize Atom with no value.
    #[inline(always)]
    pub const fn init() -> Self {
        Self {
            inner: AtomicShared::null(),
        }
    }

    /// convert this Atom to SharedAtom for cheap clone it.
    #[inline(always)]
    pub fn clonable(self) -> SharedAtom<T> {
        unsafe { Shared::new_unchecked(self) }
    }

    /// Checks whether this Atom no value.
    #[inline(always)]
    pub fn is_none(&self) -> bool {
        self.inner.is_null(Acquire)
    }

    /// Checks whether this Atom has value.
    #[inline(always)]
    pub fn is_some(&self) -> bool {
        ! self.is_none()
    }

    /// creates new instance of Atom from AtomicShared.
    #[inline(always)]
    pub const fn from(inner: AtomicShared<T>) -> Self {
        Self { inner }
    }

    /// creates new instance of Atom for provided Shared-wrapped value
    #[inline(always)]
    pub const fn new_shared(val: Shared<T>) -> Self {
        Self::from(AtomicShared::from(val))
    }
}

impl<T: 'static> Atom<T> {
    /// creates new instance of Atom for provided value
    #[inline(always)]
    pub fn new(val: T) -> Self {
        unsafe { Self::new_unchecked(val) }
    }
}

impl<T> Atom<T> {
    /// creates new instance of Atom for provided value
    ///
    /// ## <big>however, this is not checking the lifetime of T.</big>
    ///
    /// # <big>Safety</big>
    /// `T::drop` can be run after the [`Atom`] is dropped,
    /// therefore it is safe only if `T::drop` does not access short-lived data,
    /// or [`core::mem::needs_drop`] is `false` for `T`.
    #[inline(always)]
    pub unsafe fn new_unchecked(val: T) -> Self {
        let shared = unsafe { Shared::new_unchecked(val) };

        let inner = AtomicShared::from(shared);
        Self::from(inner)
    }
}

impl<T: 'static> Atom<T> {
    /// set Atom to provided value. returns the Shared-wrapped-value of you provided value.
    #[inline(always)]
    pub fn set(&self, val: T) -> Shared<T> {
        unsafe { self.set_unchecked(val) }
    }

    /// set Atom to provided value and returns the old value.
    #[inline(always)]
    pub fn swap(&self, val: T) -> Option<Shared<T>> {
        unsafe { self.swap_unchecked(val) }
    }
}

impl<T> Atom<T> {
    /// set Atom to provided value. returns the Shared-wrapped-value of you provided value.
    ///
    /// ## <big>however, this is not checking the lifetime of T.</big>
    ///
    /// # <big>Safety</big>
    /// `T::drop` can be run after the [`Atom`] is dropped,
    /// therefore it is safe only if `T::drop` does not access short-lived data,
    /// or [`core::mem::needs_drop`] is `false` for `T`.
    #[inline(always)]
    pub unsafe fn set_unchecked(&self, val: T) -> Shared<T> {
        let shared = unsafe { Shared::new_unchecked(val) };
        self.set_shared(shared.clone());
        shared
    }

    /// set Atom to provided value and returns the old value.
    ///
    /// ## <big>however, this is not checking the lifetime of T.</big>
    ///
    /// # <big>Safety</big>
    /// `T::drop` can be run after the [`Atom`] is dropped,
    /// therefore it is safe only if `T::drop` does not access short-lived data,
    /// or [`core::mem::needs_drop`] is `false` for `T`.
    #[inline(always)]
    pub unsafe fn swap_unchecked(&self, val: T) -> Option<Shared<T>> {
        let shared = unsafe { Shared::new_unchecked(val) };
        self.swap_shared(shared)
    }
}

impl<T> Atom<T> {
    /// set Atom to provided Shared-wrapped value.
    #[inline(always)]
    pub fn set_shared(&self, shared: Shared<T>) {
        self.swap_shared(shared);
    }

    /// set Atom to provided Shared-wrapped value and returns the old value.
    ///
    /// ## <big>however, this is not checking the lifetime of T.</big>
    ///
    /// # <big>Safety</big>
    /// `T::drop` can be run after the [`Atom`] is dropped,
    /// therefore it is safe only if `T::drop` does not access short-lived data,
    /// or [`core::mem::needs_drop`] is `false` for `T`.
    #[inline(always)]
    pub fn swap_shared(&self, shared: Shared<T>) -> Option<Shared<T>> {
        self.inner.swap((Some(shared), Tag::None), AcqRel).0
    }

    /// get value.
    #[inline(always)]
    pub fn get(&self) -> Option<Shared<T>> {
        let guard = Guard::new();
        self.inner.get_shared(Acquire, &guard)
    }

    /// get value by reference.
    #[inline(always)]
    pub fn get_ref<'g>(&self, g: &'g Guard) -> Option<&'g T> {
        self.inner.load(Acquire, g).as_ref()
    }

    /// take and return the value and leave it to None.
    #[inline(always)]
    pub fn take(&self) -> Option<Shared<T>> {
        self.inner.swap((None, Tag::None), AcqRel).0
    }

    /// take the value but dropping it instead of return it.
    #[inline(always)]
    pub fn del(&self) -> bool {
        self.take().is_some()
    }
}

impl<T: 'static> Atom<T> {
    /// update the provided value if any.
    /// * provided callback will be called for each iteration of CAS loop. if callback returns Err, it means cancels this update operation.
    /// * doing CAS loop until success or cancelled, so it's expected multi-calls to provided closure or function.
    /// * if CAS successfully, return submitted update (old_value, new_value) pair.
    /// WARNING: the execute time of this operation can took long time due to there is no timeout.
    #[inline(always)]
    pub fn update<E>(
        &self,
        mut f: impl FnMut(Option<Shared<T>>)->Result<Option<Shared<T>>, E>
    ) -> Result<(Option<Shared<T>>, Option<Shared<T>>), E> {
        let guard = Guard::new();
        let mut old_ptr = self.inner.load(Acquire, &guard);
        let mut old;
        let mut new;
        loop {
            old = old_ptr.get_shared();
            new = f(old.clone())?;
            match self.inner.compare_exchange(
                old_ptr,
                (new.clone(), Tag::None),
                AcqRel,
                Acquire,
                &guard,
            ) {
                Ok((_old, _new_ptr)) => {
                    return Ok((old, new));
                }
                Err((_new, op)) => {
                    old_ptr = op;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn debug() {
        dbg!(Atom::<u8>::init());
        eprintln!("{:?}", Atom::new(2*8*16));
    }

    #[test]
    fn concurrent() {
        let a = Atom::new(19440u128).clonable();

        let mut thrs = Vec::new();
        for i in 0..10 {
            let a = a.clone();
            let thr = std::thread::spawn(move || {
                let mut n: u128 = 1000;
                if (i % 2) == 0 {
                    n = 15;
                }
                for _ in 0..n {
                    if (i % 2) == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    a.update(|n| {
                        eprintln!("n={:?}", n.as_deref());
                        if false {
                            return Err(());
                        }
                        Ok(n.map(|n| { (*n) + i + 1 }).map(sdd::Shared::new))
                    });
                }
            });
            thrs.push(thr);
        }

        for thr in thrs.into_iter() {
            thr.join().unwrap();
        }
    }
}

