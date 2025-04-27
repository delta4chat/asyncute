pub use core::marker::{PhantomData, PhantomPinned};

use core::{
    ops::{Range, Deref, DerefMut},
    fmt::Debug,
};

use std::{
    net::{
        IpAddr, Ipv4Addr, Ipv6Addr,
        SocketAddr, SocketAddrV4, SocketAddrV6,
    },
    time::{Instant, Duration},
    sync::Arc,
};

use once_cell::sync::Lazy;

use portable_atomic::{
    AtomicBool,
    AtomicU8,
    AtomicU16,
    AtomicU32,
    AtomicUsize,
    AtomicU64,
    AtomicU128,
    Ordering::Relaxed,
};

use num_traits::*;

pub type PhantomNonSend = PhantomData<std::sync::MutexGuard<'static, ()>>;
pub type PhantomNonSync = PhantomData<core::cell::Cell<()>>;
pub type PhantomNonSendSync = PhantomData<(*mut (), PhantomNonSync)>;

pub type PhantomNonSendUnpin = PhantomData<(PhantomNonSend, PhantomPinned)>;
pub type PhantomNonSyncUnpin = PhantomData<(PhantomNonSync, PhantomPinned)>;
pub type PhantomNonSendSyncUnpin = PhantomData<(PhantomNonSendSync, PhantomPinned)>;

#[derive(Debug)]
pub struct AtomicDuration {
    total_ns: AtomicU128,
    changed: Option<AtomicBool>,
}

impl Default for AtomicDuration {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl AtomicDuration {
    const NANOS_PER_SEC: u128 = 1_000_000_000;

    #[inline(always)]
    const fn secs_to_nanos(s: u64) -> u128 {
        // this is safe due to (18446744073709551615_000_000_000 < u128::MAX)
        (s as u128) * Self::NANOS_PER_SEC
    }

    /// u128 nanoseconds version of [core::time::Duration::from_nanos](https://doc.rust-lang.org/1.84.1/src/core/time.rs.html#318-326)
    #[inline(always)]
    const fn duration_from_nanos(nanos: u128) -> Duration {
        let secs = (nanos / Self::NANOS_PER_SEC) as u64;
        let subsec_nanos = (nanos % Self::NANOS_PER_SEC) as u32;
        Duration::new(secs, subsec_nanos)
    }

    #[inline(always)]
    pub const fn zero() -> Self {
        Self::new(0, 0)
    }

    #[inline(always)]
    pub const fn new(s: u64, n: u32) -> Self {
        let ns = Self::secs_to_nanos(s) + (n as u128);
        Self {
            total_ns: AtomicU128::new(ns),
            changed: None,
        }
    }

    #[inline(always)]
    pub const fn new_with_trace(s: u64, n: u32) -> Self {
        let mut this = Self::new(s, n);
        this.trace_change();
        this
    }

    #[inline(always)]
    pub const fn trace_change(&mut self) -> &mut Self {
        self.changed = Some(AtomicBool::new(false));
        self
    }

    #[inline(always)]
    pub fn set_secs(&self, s: u64) -> &Self {
        self.set(Duration::new(s, self.subsec_nanos()))
    }

    #[inline(always)]
    pub fn set_subsec_nanos(&self, n: u32) -> &Self {
        self.set(Duration::new(self.as_secs(), n))
    }

    #[inline(always)]
    pub fn set(&self, d: Duration) -> &Self {
        self.total_ns.store(d.as_nanos(), Relaxed);
        if let Some(ref changed) = self.changed {
            changed.store(true, Relaxed);
        }
        self
    }

    #[inline(always)]
    pub fn add_secs(&self, s: u64) -> &Self {
        self.total_ns.fetch_add(Self::secs_to_nanos(s), Relaxed);
        if let Some(ref changed) = self.changed {
            changed.store(true, Relaxed);
        }
        self
    }

    #[inline(always)]
    pub fn add_nanos(&self, n: u128) -> &Self {
        self.total_ns.fetch_add(n, Relaxed);
        if let Some(ref changed) = self.changed {
            changed.store(true, Relaxed);
        }
        self
    }

    #[inline(always)]
    pub fn add(&self, d: Duration) -> &Self {
        self.total_ns.fetch_add(d.as_nanos(), Relaxed);
        if let Some(ref changed) = self.changed {
            changed.store(true, Relaxed);
        }
        self
    }

    #[inline(always)]
    pub fn as_secs(&self) -> u64 {
        self.get().as_secs()
    }

    #[inline(always)]
    pub fn subsec_nanos(&self) -> u32 {
        self.get().subsec_nanos()
    }

    #[inline(always)]
    pub fn get(&self) -> Duration {
        Self::duration_from_nanos(self.total_ns.load(Relaxed))
    }

    /// checks whether the inner value changed since last call this.
    #[inline(always)]
    pub fn changed(&self) -> bool {
        if let Some(ref changed) = self.changed {
            let is_changed = changed.load(Relaxed);
            if is_changed {
                changed.store(false, Relaxed);
            }
            is_changed
        } else {
            panic!("trace change is not enabled!");
        }
    }
}

#[derive(Debug)]
pub struct AtomicIpv4Addr {
    bits: AtomicU32,
}
impl AtomicIpv4Addr {
    #[inline(always)]
    pub const fn default() -> Self {
        Self::new(Ipv4Addr::UNSPECIFIED)
    }

    #[inline(always)]
    pub const fn new(ip4: Ipv4Addr) -> Self {
        Self {
            bits:
                AtomicU32::new(
                    u32::from_be_bytes(ip4.octets())
                )
        }
    }

    #[inline(always)]
    pub fn get_bits(&self) -> u32 {
        self.bits.load(Relaxed)
    }

    #[inline(always)]
    pub fn set_bits(&self, bits: u32) -> &Self {
        self.bits.store(bits, Relaxed);
        self
    }

    #[inline(always)]
    pub fn get(&self) -> Ipv4Addr {
        Ipv4Addr::from(self.get_bits().to_be_bytes())
    }

    #[inline(always)]
    pub fn set(&self, ip4: Ipv4Addr) -> &Self {
        self.set_bits(u32::from_be_bytes(ip4.octets()))
    }
}

#[derive(Debug)]
pub struct AtomicIpv6Addr {
    bits: AtomicU128,
}
impl AtomicIpv6Addr {
    #[inline(always)]
    pub const fn default() -> Self {
        Self::new(Ipv6Addr::UNSPECIFIED)
    }

    #[inline(always)]
    pub const fn new(ip6: Ipv6Addr) -> Self {
        Self {
            bits:
                AtomicU128::new(
                    u128::from_be_bytes(ip6.octets())
                )
        }
    }

    #[inline(always)]
    pub fn get_bits(&self) -> u128 {
        self.bits.load(Relaxed)
    }

    #[inline(always)]
    pub fn set_bits(&self, bits: u128) -> &Self {
        self.bits.store(bits, Relaxed);
        self
    }

    #[inline(always)]
    pub fn get(&self) -> Ipv6Addr {
        Ipv6Addr::from(self.get_bits().to_be_bytes())
    }

    #[inline(always)]
    pub fn set(&self, ip6: Ipv6Addr) -> &Self {
        self.set_bits(u128::from_be_bytes(ip6.octets()))
    }
}

#[derive(Debug)]
pub struct AtomicIpAddr {
    kind: AtomicU8,
    data: AtomicU128,
}

impl AtomicIpAddr {
    pub const KIND_IPV4:    u8 = 4;
    pub const KIND_IPV6:    u8 = 6;
    pub const KIND_PENDING: u8 = 0xfe;

    #[inline(always)]
    pub const fn default() -> Self {
        Self::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED))
    }

    #[inline(always)]
    pub const fn new(ip: IpAddr) -> Self {
        let (kind, data) = Self::ip2tuple(ip);
        Self {
            kind: AtomicU8::new(kind),
            data: AtomicU128::new(data),
        }
    }

    #[inline(always)]
    const fn ip2tuple(ip: IpAddr) -> (u8, u128) {
        let mut kind;
        let mut data;
        match ip {
            IpAddr::V4(ip4) => {
                kind = Self::KIND_IPV4;
                data = [0u8; 16];
                let octets = ip4.octets();
                let mut i = 0;
                while i < 4 {
                    data[i] = octets[i];
                    i += 1;
                }
            },
            IpAddr::V6(ip6) => {
                kind = Self::KIND_IPV6;
                data = ip6.octets();
            }
        }
        (kind, u128::from_be_bytes(data))
    }

    #[inline(always)]
    pub fn kind(&self) -> u8 {
        self.kind.load(Relaxed)
    }

    #[inline(always)]
    pub fn is_ipv4(&self) -> bool {
        self.kind() == Self::KIND_IPV4
    }

    #[inline(always)]
    pub fn is_ipv6(&self) -> bool {
        self.kind() == Self::KIND_IPV6
    }

    #[inline(always)]
    fn data(&self) -> u128 {
        self.data.load(Relaxed)
    }

    #[inline(always)]
    pub fn get(&self) -> IpAddr {
        let mut k;
        let mut d;
        loop {
            k = self.kind();
            if k == Self::KIND_PENDING {
                continue;
            }
            d = self.data().to_be_bytes();
            match k {
                Self::KIND_IPV4 => {
                    return IpAddr::V4(Ipv4Addr::new(d[0], d[1], d[2], d[3]));
                },
                Self::KIND_IPV6 => {
                    return IpAddr::V6(Ipv6Addr::from(d));
                },
                Self::KIND_PENDING => {
                    continue;
                },
                _ => {
                    panic!("unknown kind!");
                }
            }
        }
    }

    #[inline(always)]
    pub fn set(&self, ip: IpAddr) -> &Self {
        self.kind.store(Self::KIND_PENDING, Relaxed);

        let (kind, data) = Self::ip2tuple(ip);

        self.data.store(data, Relaxed);
        self.kind.store(kind, Relaxed);

        self
    }
}

#[derive(Debug)]
pub struct AtomicSocketAddr {
    ready: AtomicBool,
    ip: AtomicIpAddr,
    port: AtomicU16,
}

impl AtomicSocketAddr {
    #[inline(always)]
    pub const fn default() -> Self {
        Self::new(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0))
    }

    #[inline(always)]
    pub const fn new(addr: SocketAddr) -> Self {
        Self {
            ready: AtomicBool::new(true),
            ip: AtomicIpAddr::new(addr.ip()),
            port: AtomicU16::new(addr.port()),
        }
    }

    #[inline(always)]
    pub fn ip(&self) -> IpAddr {
        self.ip.get()
    }

    #[inline(always)]
    pub fn port(&self) -> u16 {
        self.port.load(Relaxed)
    }

    #[inline(always)]
    pub fn get(&self) -> SocketAddr {
        while ! self.ready.load(Relaxed) {
            // waiting...
        }
        SocketAddr::new(self.ip(), self.port())
    }

    #[inline(always)]
    pub fn set(&self, addr: SocketAddr) -> &Self {
        self.ready.store(false, Relaxed);
        self.ip.set(addr.ip());
        self.port.store(addr.port(), Relaxed);
        self.ready.store(true, Relaxed);

        self
    }
}

#[derive(Debug)]
pub struct AtomicRange<AT: Debug> {
    start: AT,
    end: AT,

    strict: bool,
}

#[derive(Debug)]
pub struct AtomicRangeStrict<AT: Debug>(AtomicRange<AT>);

#[derive(Debug, Clone)]
pub struct NumIter<T>
where
    T: Debug + Clone + PartialOrd + ConstZero + ConstOne + Bounded + CheckedAdd + CheckedSub,
{
    stopped: bool,

    current: T,
    until: T,

    step: T,
    incr: bool,
}


impl<T> NumIter<T>
where
    T: Debug + Clone + PartialOrd + ConstZero + ConstOne + Bounded + CheckedAdd + CheckedSub + Euclid,
{
    #[inline(always)]
    pub fn new(current: T, until: T, step: T) -> Self {
        Self {
            stopped: current == until,
            incr: current < until,

            current,
            until,
            step,
        }
    }

    #[inline(always)]
    pub fn size(&self) -> T {
        let space =
            self.current
                .checked_sub(&self.until)
                .or_else(|| { self.until.checked_sub(&self.current) })
                .unwrap();

        let (mut size, rem) = space.div_rem_euclid(&self.step);
        if rem != T::ZERO {
            size = size.checked_add(&T::ONE).expect("size exceeded the maximum value!");
        }
        size
    }
}

impl<T> Iterator for NumIter<T>
where
    T:  Debug
        + Clone
        + PartialOrd
        + ConstZero
        + ConstOne
        + Bounded
        + CheckedAdd
        + CheckedSub
        + Euclid
        + ToPrimitive,
{
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<T> {
        if self.stopped {
            return None;
        }

        let n = self.current.clone();
        if self.incr {
            if n >= self.until {
                self.stopped = true;
                return None;
            }

            if let Some(next) = self.current.checked_add(&self.step) {
                self.current = next;
            } else {
                self.stopped = true;
            }
        } else {
            if n <= self.until {
                self.stopped = true;
                return None;
            }

            if let Some(next) = self.current.checked_sub(&self.step) {
                self.current = next;
            } else {
                self.stopped = true;
            }
        }
                            
        Some(n)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(size) = self.size().to_usize() {
            (size, Some(size))
        } else {
            (0, None)
        }
    }
}

macro_rules! atomic_range_impl {
    ($($atom:ty = $num:ident,)*) => {
        impl<AT: Debug> AtomicRange<AT> {
            $(
                #[inline(always)]
                pub const fn $num() -> AtomicRange<$atom> {
                    AtomicRange::<$atom>::default()
                }
            )*
        }
        impl<AT: Debug> AtomicRangeStrict<AT> {
            $(
                #[inline(always)]
                pub const fn $num() -> AtomicRangeStrict<$atom> {
                    AtomicRangeStrict::<$atom>::default()
                }
            )*
        }

        $(
            impl From<Range<$num>> for AtomicRange<$atom> {
                #[inline(always)]
                fn from(val: Range<$num>) -> AtomicRange<$atom> {
                    AtomicRange::<$atom>::from_range(val)
                }
            }
            impl From<AtomicRange<$atom>> for Range<$num> {
                #[inline(always)]
                fn from(val: AtomicRange<$atom>) -> Range<$num> {
                    val.range()
                }
            }

            impl AtomicRange<$atom> {
                #[inline(always)]
                pub const fn default() -> Self {
                    const ZERO: $num = 0u8 as $num;
                    Self {
                        start: <$atom>::new(ZERO),
                        end: <$atom>::new(ZERO),

                        strict: false,
                    }
                }

                #[inline(always)]
                pub const fn new(start: $num, end: $num) -> Self {
                    Self {
                        start: <$atom>::new(start),
                        end: <$atom>::new(end),

                        strict: false,
                    }
                }

                #[inline(always)]
                pub const fn from_range(val: Range<$num>) -> Self {
                    Self::new(val.start, val.end)
                }

                #[inline(always)]
                pub const fn strict(&mut self) -> &mut Self {
                    self.strict = true;
                    self
                }

                #[inline(always)]
                pub fn start(&self) -> $num {
                    self.start.load(Relaxed)
                }

                #[inline(always)]
                pub fn end(&self) -> $num {
                    self.end.load(Relaxed)
                }

                #[inline(always)]
                pub fn range(&self) -> Range<$num> {
                    Range {
                        start: self.start(),
                        end: self.end(),
                    }
                }

                #[inline(always)]
                pub fn set_start(&self, val: $num) -> bool {
                    if self.strict {
                        let end = self.end();
                        if val >= end {
                            return false;
                        }
                    }
                    self.start.store(val, Relaxed);
                    true
                }

                #[inline(always)]
                pub fn set_end(&self, val: $num) -> bool {
                    if self.strict {
                        let start = self.start();
                        if val <= start {
                            return false;
                        }
                    }
                    self.end.store(val, Relaxed);
                    true
                }

                #[inline(always)]
                pub fn set_range(&self, val: Range<$num>) -> bool {
                    if self.strict {
                        if val.start >= val.end {
                            return false;
                        }
                    }
                    self.start.store(val.start, Relaxed);
                    self.end.store(val.end, Relaxed);
                    true
                }

                /// due to this is half-open, so range does not contains "end" itself.
                /// so "start" must not equals "end", because it's paradox: "start" requires to be included, but "end" requires to be excluded.
                #[inline(always)]
                pub fn is_valid(&self) -> bool {
                    self.start() < self.end()
                }

                /// "start" must not equals "end", because it's paradox: "start" requires to be included, but "end" requires to be excluded.
                ///
                /// so if "start" is equals to "end", it's invalid value (not a range), so it is not empty, but also not have something.
                #[inline(always)]
                pub fn is_empty(&self) -> bool {
                    self.start() > self.end()
                }

                #[inline(always)]
                pub fn iter(&self) -> NumIter<$num> {
                    NumIter::new(self.start(), self.end(), $num::ONE)
                }

                #[inline(always)]
                pub fn contains(&self, val: $num) -> bool {
                    let start = self.start();
                    let end = self.end();
                    if start < end {
                        val >= start && val < end
                    } else if start > end {
                        val <= start && val > end
                    } else {
                        false
                    }
                }
            }

            impl AtomicRangeStrict<$atom> {
                #[inline(always)]
                pub const fn default() -> Self {
                    Self(AtomicRange {
                        start: <$atom>::new(0u8 as _),
                        end: <$atom>::new(1u8 as _),

                        strict: true,
                    })
                }

                #[inline(always)]
                pub const fn new(start: $num, end: $num) -> Self {
                    if start >= end {
                        panic!("strict range does not accept invalid values!");
                    }

                    Self(AtomicRange {
                        start: <$atom>::new(start),
                        end: <$atom>::new(end),

                        strict: true,
                    })
                }
            }
            impl Deref for AtomicRangeStrict<$atom> {
                type Target = AtomicRange<$atom>;

                #[inline(always)]
                fn deref(&self) -> &AtomicRange<$atom> {
                    &self.0
                }
            }
            impl DerefMut for AtomicRangeStrict<$atom> {
                #[inline(always)]
                fn deref_mut(&mut self) -> &mut AtomicRange<$atom> {
                    &mut self.0
                }
            }
        )*
    }
}

atomic_range_impl!(
    AtomicU8    = u8,
    AtomicU16   = u16,
    AtomicU32   = u32,
    AtomicUsize = usize,
    AtomicU64   = u64,
    AtomicU128  = u128,
);

#[derive(Debug)]
pub struct StableClock {
    started_elapsed: scc2::Atom<(Instant, AtomicDuration)>,
    tick: Duration,
    join_handle: std::thread::JoinHandle<()>,
}

static GLOBAL_STABLE_CLOCK: Lazy<StableClock> = Lazy::new(StableClock::default);

impl StableClock {
    #[inline(always)]
    pub fn global() -> &'static Self {
        &*GLOBAL_STABLE_CLOCK
    }

    #[inline(always)]
    pub fn is_time_jumped(&self) -> bool {
        self.time_diff() > (self.tick * 10)
    }

    #[inline(always)]
    pub fn time_diff(&self) -> Duration {
        let stable_now = self.now();
        let instant_now = Instant::now();

        if stable_now > instant_now {
            stable_now.duration_since(instant_now)
        } else {
            instant_now.duration_since(stable_now)
        }
    }

    #[inline(always)]
    pub fn now(&self) -> Instant {
        if self.join_handle.is_finished() {
            panic!("unexpectedly StableClock tick thread exited!");
        }

        let (started, elapsed) = &*(self.started_elapsed.get().expect("no value"));
        let elapsed = elapsed.add_nanos(1).get();
        (*started) + elapsed
    }

    #[inline(always)]
    fn default() -> Self {
        Self::new(Duration::from_millis(100))
    }

    #[inline(always)]
    fn new(tick: Duration) -> Self {
        let started_elapsed = scc2::Atom::new((Instant::now(), AtomicDuration::new(0, 0)));
        let join_handle = {
            let tick_ns = tick.as_nanos();
            std::thread::Builder::new()
                .name(String::from("stable-clock"))
                .spawn(move || {
                    let this = Self::global();
                    let mut se = this.started_elapsed.get().expect("unexpected no value");
                    let mut diff;
                    let max_diff = tick * 1;
                    loop {
                        std::thread::sleep(tick);
                        se.1.add_nanos(tick_ns);
                        diff = this.time_diff();
                        if diff > tick && diff < max_diff {
                            se = Arc::new((Instant::now(), AtomicDuration::new(0, 0)));
                            this.started_elapsed.set_arc(se.clone());
                        }
                    }
                }).expect("unable to spawn tick thread!")
        };

        Self {
            started_elapsed,
            tick,
            join_handle,
        }
    }
}
