//! id.rs

use crate::*;

use std::{
    time::{SystemTime, Instant, Duration},
    hash::{BuildHasher, Hasher, Hash},
};

use portable_atomic::AtomicU64;

/// extension for u128 ID.
pub trait Id {
    /// Get Name Space (NS) of ID.
    fn namespace(self) -> u64;

    /// Get Count (sequence position) of ID.
    fn count(self) -> u64;
}

impl Id for u128 {
    // high 64-bit value
    fn namespace(self) -> u64 {
        (self >> 64) as u64
    }

    // low 64-bit value
    fn count(self) -> u64 {
        self as u64
    }
}

impl Id for (u64, u64) {
    fn namespace(self) -> u64 {
        self.0
    }

    fn count(self) -> u64 {
        self.1
    }
}

/// the namespaced ID generator.
pub struct ID {
    // the 64-bit prefix (name space) of u128 id.
    prefix: u128,

    counter: AtomicU64,
}

impl ID {
    /// check whether provided name space is special.
    #[inline(always)]
    pub const fn is_special_ns(ns: u64) -> bool {
        const STARTS_WITH_BIT_1: u64 = 1 << 63;

        ns < STARTS_WITH_BIT_1
    }

    /// new ID with provided name space (8 bytes for NS only, or 16 bytes for full ID)
    #[inline(always)]
    pub const fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if let Some(this) = Self::_from_bytes(bytes) {
            if ! this.is_special() {
                return Some(this);
            }
        }

        None
    }

    /// unrestricted version of `from_bytes` that allow special NS.
    #[inline(always)]
    const fn _from_bytes(bytes: &[u8]) -> Option<Self> {
        match bytes.len() {
            8 => {
                let ns = u64::from_be_bytes(option_try!(slice_to_array(bytes, 0)));
                Some(Self::_new(ns))
            },
            16 => {
                let ns = u64::from_be_bytes(option_try!(slice_to_array(bytes, 0)));
                let count = u64::from_be_bytes(option_try!(slice_to_array(bytes, 8)));
                Some(Self::_init(ns, count))
            },
            _ => None
        }
    }

    /// new ID with provided name space (unsigned 64-bit integer)
    #[inline(always)]
    pub const fn new(ns: u64) -> Option<Self> {
        if Self::is_special_ns(ns) {
            return None;
        }

        Some(Self::_new(ns))
    }

    /// unrestricted version of `new` that allow special NS.
    #[inline(always)]
    const fn _new(ns: u64) -> Self {
        Self::_init(ns, 1)
    }

    /// initial ID with provided name space and count.
    #[inline(always)]
    pub const fn init(ns: u64, count: u64) -> Option<Self> {
        let this = Self::_init(ns, count);
        if this.is_special() {
            return None;
        }
        Some(this)
    }

    /// unrestricted version of `init` that allow special NS.
    #[inline(always)]
    const fn _init(ns: u64, count: u64) -> Self {
        Self {
            prefix: (ns as u128) << 64,
            counter: AtomicU64::new(count),
        }
    }

    /// check whether this instance of ID is special.
    #[inline(always)]
    pub const fn is_special(&self) -> bool {
        const STARTS_WITH_BIT_1: u128 = 1 << 127;

        self.prefix < STARTS_WITH_BIT_1
    }

    /// runtime version of `new` that generate NS from all of there information:
    /// * ahash RandomState.
    /// * PID of current process.
    /// * ThreadId of current Thread (this is assigned by rust std, not OS-specific TID).
    /// * memory address of some types defined in static, runtime, or alloc.
    /// * current UNIX timestamp and monotonic timestamp.
    /// * sleep jitter.
    /// * pre-defined seed.
    #[inline(always)]
    pub fn auto_ns() -> Self {
        // NonClone is not derived Copy or Clone trait, so it's unclonable.
        #[allow(dead_code)]
        struct NonClone(u8);

        static SEED: u128 = 36737317109501180927066999484774518054;
        const ZERO_DUR: Duration = Duration::new(0, 0);

        let mut t;
        let mut ns;

        let mut h = ahash::RandomState::new().build_hasher();
        std::process::id().hash(&mut h);
        std::thread::current().id().hash(&mut h);
        SEED.hash(&mut h); // seed
        core::ptr::addr_of!(SEED).hash(&mut h); // static address
        core::ptr::addr_of!(h).hash(&mut h); // runtime address
        loop {
            #[allow(unused_allocation)]
            (Box::new(NonClone(123)).as_ref() as *const NonClone).hash(&mut h); // heap alloc address

            SystemTime::now().hash(&mut h);

            t = Instant::now();
            t.hash(&mut h);

            std::thread::sleep(ZERO_DUR);
            std::thread::park_timeout(ZERO_DUR);

            t.elapsed().hash(&mut h);

            ns = h.finish();

            if Self::is_special_ns(ns) {
                continue;
            }

            return Self::_new(ns);
        }
    }

    /// try to generate new ID. returns None if exhausted the 64-bit counter.
    #[inline(always)]
    pub fn generate(&self) -> Option<u128> {
        let count = self.counter.checked_add(1)?;

        Some(
            self.prefix | (count as u128)
        )
    }
}

/// the "global" name space.
pub static GLOBAL_ID: ID = option_unwrap!(ID::_from_bytes(b"__GLOBAL"));

/// the "executor" name space.
pub(crate) static EXECUTOR_ID: ID = option_unwrap!(ID::_from_bytes(b"EXECUTOR"));

/// the "task" name space.
pub(crate) static TASK_ID: ID = option_unwrap!(ID::_from_bytes(b"TASK ID."));

/// shortcut for `GLOBAL_ID.generate()`
#[inline(always)]
pub fn gen_global_id() -> Option<u128> {
    GLOBAL_ID.generate()
}

/// shortcut for `EXECUTOR_ID.generate()`
#[inline(always)]
pub(crate) fn gen_executor_id() -> Option<u128> {
    EXECUTOR_ID.generate()
}

/// shortcut for `TASK_ID.generate()`
#[inline(always)]
pub(crate) fn gen_task_id() -> Option<u128> {
    TASK_ID.generate()
}

