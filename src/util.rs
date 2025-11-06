//! util.rs

pub use core::marker::{PhantomData, PhantomPinned};

use core::{
    ops::{Range, Deref, DerefMut},
    fmt,
};

use std::{
    net::{
        IpAddr, Ipv4Addr, Ipv6Addr,
        SocketAddr,
    },
    time::{Instant, Duration},
};

use once_cell::sync::{Lazy, OnceCell};

use portable_atomic::{
    *,
    Ordering::Relaxed,
};

use num_traits::*;

use scc2::ebr::AtomicOwned;

// FIXME(hack) until the stabilize of #![feature(negative_impls)] 

/// the PhantomData contains a !Send type.
pub type PhantomNonSend = PhantomData<std::sync::MutexGuard<'static, ()>>;

/// the PhantomData contains a !Sync type.
pub type PhantomNonSync = PhantomData<core::cell::Cell<()>>;

/// the PhantomData contains a !Send + !Sync type.
pub type PhantomNonSendSync = PhantomData<(PhantomNonSend, PhantomNonSync)>;

/// the PhantomData contains a !Send + !Unpin type.
pub type PhantomNonSendUnpin = PhantomData<(PhantomNonSend, PhantomPinned)>;

/// the PhantomData contains a !Sync + !Unpin type.
pub type PhantomNonSyncUnpin = PhantomData<(PhantomNonSync, PhantomPinned)>;

/// the PhantomData contains a !Send + !Sync + !Unpin type.
pub type PhantomNonSendSyncUnpin = PhantomData<(PhantomNonSendSync, PhantomPinned)>;

/// helper macro for use "try" (`?` operator) for Option in constant context.
#[macro_export]
macro_rules! option_try {
    ($c:expr) => {
        match $c {
            Some(v) => v,
            _ => {
                return None;
            }
        }
    }
}

/// helper macro for use Option::unwrap() in constant context.
#[macro_export]
macro_rules! option_unwrap {
    ($c:expr) => {
        match $c {
            Some(v) => v,
            _ => {
                panic!("called `Option::unwrap()` on a `None` value!");
            }
        }
    }
}

/// helper macro for use "try" (`?` operator) for Result in constant context.
#[macro_export]
macro_rules! result_try {
    ($c:expr) => {
        match $c {
            Ok(v) => v,
            Err(e) => {
                return Err(e);
            }
        }
    }
}

/// helper macro for use Result::unwrap() in constant context.
#[macro_export]
macro_rules! result_unwrap {
    ($c:expr) => {
        match $c {
            Ok(v) => v,
            _ => {
                panic!("called `Result::unwrap()` on an `Err` value!");
            }
        }
    }
}

macro_rules! gen_atomic_checked {
    ($($op:ident,)*) => {
        /// the "checked" version of atomic types.
        pub trait AtomicChecked {
            /// the inner value of this Atomic type.
            type Item;

            $(
                #[doc = concat!("the ", stringify!($op), " method.")]
                fn $op(&self, val: Self::Item) -> Option<Self::Item>;
            )*
        }

        /// helper trait for float numbers
        trait Checked: Sized {
            $(
                #[doc = concat!("the ", stringify!($op), " method.")]
                fn $op(self, val: Self) -> Option<Self>;
            )*
        }

        checked_float_impls!(f32 = $($op,)*);
        checked_float_impls!(f64 = $($op,)*);
    }
}

macro_rules! checked_float_impls {
    ($name:ident = $($op:ident,)*) => {
        impl Checked for $name {
            $(
                #[inline(always)]
                fn $op(self, val: Self) -> Option<Self> {
                    const OP: &'static str = stringify!($op);

                    const IS_ADD: bool = str_eq(OP, "checked_add");
                    const IS_SUB: bool = str_eq(OP, "checked_sub");
                    const IS_MUL: bool = str_eq(OP, "checked_mul");
                    const IS_DIV: bool = str_eq(OP, "checked_div");
                    const IS_REM: bool = str_eq(OP, "checked_rem");

                    if self.is_nan() || val.is_nan() {
                        return None;
                    }
                    if self.is_infinite() || val.is_infinite() {
                        return None;
                    }

                    let res =
                        if IS_ADD {
                            self + val
                        } else if IS_SUB {
                            self - val
                        } else if IS_MUL {
                            self * val
                        } else if IS_DIV {
                            if val.is_zero() {
                                return None;
                            }
                            self / val
                        } else if IS_REM {
                            if val.is_zero() {
                                return None;
                            }
                            self % val
                        } else {
                            unreachable!();
                        };

                    if res.is_infinite() || res.is_nan() {
                        return None;
                    }

                    Some(res)
                }
            )*
        }
    }
}

gen_atomic_checked!(
    checked_add,
    checked_sub,
    checked_mul,
    checked_div,
    checked_rem,
);

macro_rules! atomic_checked_impl {
    ($atom:ident, $t:ident = $($op:ident,)*) => {
        impl AtomicChecked for $atom {
            type Item = $t;

            $(
                #[inline(always)]
                fn $op(&self, val: $t) -> Option<$t> {
                    const ZERO: $t = 0u8 as $t;

                    const OP: &'static str = stringify!($op);

                    const IS_ADD_OR_SUB: bool = str_eq(OP, "checked_add") || str_eq(OP, "checked_sub");
                    const IS_DIV_OR_REM: bool = str_eq(OP, "checked_div") || str_eq(OP, "checked_rem");
                    const IS_MUL:        bool = str_eq(OP, "checked_mul");

                    if val == ZERO {
                        if IS_ADD_OR_SUB {
                            return Some(self.load(Relaxed));
                        }
                        if IS_MUL {
                            return Some(self.swap(ZERO, Relaxed));
                        }
                        if IS_DIV_OR_REM {
                            return None;
                        }
                    }

                    let mut old = self.load(Relaxed);
                    let mut new;
                    loop {
                        new = old.$op(val)?;
                        match
                            self.compare_exchange(
                                old,     new,
                                Relaxed, Relaxed,
                            )
                        {
                            Ok(prev) => {
                                return Some(prev);
                            },
                            Err(cur) => {
                                old = cur;
                            }
                        }
                    }
                }
            )*
        }
    }
}

macro_rules! atomic_checked_impls {
    ($($atom:ident = $t:ident,)*) => {
        $(
            atomic_checked_impl!(
                $atom, $t =
                    checked_add,
                    checked_sub,
                    checked_mul,
                    checked_div,
                    checked_rem,
            );
        )*
    }
}

atomic_checked_impls!(
    AtomicU8    = u8,
    AtomicU16   = u16,
    AtomicU32   = u32,
    AtomicU64   = u64,
    AtomicU128  = u128,
    AtomicUsize = usize,

    AtomicI8    = i8,
    AtomicI16   = i16,
    AtomicI32   = i32,
    AtomicI64   = i64,
    AtomicI128  = i128,
    AtomicIsize = isize,

    AtomicF32   = f32,
    AtomicF64   = f64,
);

/// check the equality of two byte slices in constant context.
#[inline(always)]
pub const fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    let a_len = a.len();
    if a_len != b.len() {
        return false;
    }

    let mut i = 0;
    while i < a_len {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }

    true
}

/// check the equality of two str in constant context.
#[inline(always)]
pub const fn str_eq(a: &str, b: &str) -> bool {
    bytes_eq(a.as_bytes(), b.as_bytes())
}

/// convert a part of slice to array.
#[inline(always)]
pub const fn slice_to_array<T: Copy, const N: usize>(slice: &[T], start: usize) -> Option<[T; N]> {
    let slice_len = slice.len();

    if slice_len == 0 {
        return None;
    }

    if N == 0 {
        return Some([slice[0]; N]);
    }

    if slice_len < N {
        return None;
    }
    if slice_len < start {
        return None;
    }
    if (slice_len - start) < N {
        return None;
    }

    let mut out = [slice[0]; N];
    let mut i = start;
    let mut ii = 0;
    while ii < N {
        out[ii] = slice[i];
        i += 1;
        ii += 1;
    }
    Some(out)
}

/// the Ordering of List push/pop
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ListOrdering {
    /// First in, first out.
    FIFO = b'f',

    /// Last in, first out.
    LIFO = b'l',

    /// Unordered (or ordering unspecified): ordering is not guaranteed.
    Unordered = b'?',
}

/// Concurrent List.
pub trait List<T> {
    /// create new empty list.
    fn new() -> Self;

    /// get the ordering of list.
    fn ordering() -> ListOrdering;

    /// get the current length of stored items.
    fn len(&self) -> usize;

    /// push provided item to the set of items.
    /// * the item must not be discarded. if there is no space, the Err(item) should be returned back.
    /// * if this operation is successfully, it must ensures the pushed item will be appears in pop operations.
    /// * this operation must not blocking.
    fn push(&self, val: T) -> Result<(), T>;

    /// try to pop item from the set of items.
    /// * return None if empty.
    /// * this operation must not blocking.
    fn pop(&self) -> Option<T>;
}

#[cfg(feature="crossbeam-deque")]
/// crossbeam_deque::Injector version of "event channel"
pub mod injector {
    use super::{ListOrdering, event_channel};
    use crossbeam_deque::Injector;

    /// alias of `event_channel::Channel<T, Injector<T>>`
    pub type InjectorChannel<T> = event_channel::Channel<T, Injector<T>>;

    /// alias of `event_channel::Sender<T, Injector<T>>`
    pub type Sender<T> = event_channel::Sender<T, Injector<T>>;

    /// alias of `event_channel::Receiver<T, Injector<T>>`
    pub type Receiver<T> = event_channel::Receiver<T, Injector<T>>;

    impl<T> super::List<T> for Injector<T> {
        fn new() -> Self {
            Injector::new()
        }

        fn ordering() -> ListOrdering {
            ListOrdering::FIFO
        }

        fn len(&self) -> usize {
            Injector::len(self)
        }

        fn push(&self, val: T) -> Result<(), T> {
            Injector::push(self, val);
            Ok(())
        }

        fn pop(&self) -> Option<T> {
            let mut ret = Injector::steal(self);
            while ret.is_retry() {
                ret = Injector::steal(self);
            }
            ret.success()
        }
    }
}

/// helper for support blocking wait of [`List`] (powered by [`event_listener::Event`])
#[cfg(feature="event-listener")]
pub mod event_channel {
    use super::*;

    use event_listener::{Event, Listener, listener};

    /// the private Inner of InjectorChannel
    pub struct Inner<T, L: List<T>> {
        bounded: Option<usize>,
        list: L,
        send_event: Event,
        recv_event: Event,
        _pd: PhantomData<T>,
    }

    impl<T, L: List<T>> fmt::Debug for Inner<T, L> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Inner")
             .field("bounded", &self.bounded)
             .field("list", &format!("dyn List[T; {}]", self.list.len()))
             .field("send_event", &self.send_event)
             .field("recv_event", &self.recv_event)
             .finish()
        }
    }

    /// the "channel" backed by `List`.
    pub struct Channel<T, L: List<T>>(Arc<Inner<T, L>>);

    impl<T, L: List<T>> fmt::Debug for Channel<T, L> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_tuple("Channel")
             .field(&self.0)
             .finish()
        }
    }

    impl<T, L: List<T>> Clone for Channel<T, L> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }

    impl<T, L: List<T>> Deref for Channel<T, L> {
        type Target = Inner<T, L>;

        fn deref(&self) -> &Inner<T, L> {
            &self.0
        }
    }

    impl<T, L: List<T>> Channel<T, L> {
        /// create new InjectorChanel without the limit of maximum capacity.
        #[inline(always)]
        pub fn unbounded() -> Self {
            Self::new(None)
        }

        /// create new paired (Sender, Receiver) of InjectorChannel without the limit of maximum capacity.
        #[inline(always)]
        pub fn unbounded_split() -> (Sender<T, L>, Receiver<T, L>) {
            Self::unbounded().split()
        }

        /// create new InjectorChanel and limits the maximum capacity.
        #[inline(always)]
        pub fn bounded(size: usize) -> Self {
            Self::new(Some(size))
        }

        /// create new paired (Sender, Receiver) of InjectorChannel and limits the maximum capacity.
        #[inline(always)]
        pub fn bounded_split(size: usize) -> (Sender<T, L>, Receiver<T, L>) {
            Self::bounded(size).split()
        }

        /// create new InjectorChanel.
        #[inline(always)]
        pub fn new(bounded: Option<usize>) -> Self {
            if let Some(size) = bounded {
                assert!(size > 0);
            }

            Self(Arc::new(Inner {
                bounded,
                list: L::new(),
                send_event: Event::new(),
                recv_event: Event::new(),
                _pd: PhantomData,
            }))
        }

        /// create new paired (Sender, Receiver) of InjectorChannel.
        #[inline(always)]
        pub fn new_split(bounded: Option<usize>) -> (Sender<T, L>, Receiver<T, L>) {
            Self::new(bounded).split()
        }

        /// split the InjectorChannel to (Sender, Receiver) pair.
        #[inline(always)]
        pub fn split(&self) -> (Sender<T, L>, Receiver<T, L>) {
            let sender = Sender(self.clone());
            let receiver = Receiver(self.clone());
            (sender, receiver)
        }

        /// Returns the number of messages in the channel.
        #[inline(always)]
        pub fn len(&self) -> usize {
            self.list.len()
        }

        /// try to send new message to channel.
        /// * return Ok(usize) if the message has been sent, and return the number of notified receivers.
        /// * return Err(T) if the channel is full and unable to send this message.
        #[inline(always)]
        pub fn try_send(&self, msg: T) -> Result<usize, T> {
            let len = self.len();

            if let Some(size) = self.bounded {
                if len >= size {
                    return Err(msg);
                }
            }

            self.list.push(msg)?;
            Ok(self.send_event.notify_relaxed(len.checked_add(1).unwrap_or(len)))
        }

        /// the blocking version of `try_send`.
        /// * if channel is full, this method will blocking until have space.
        /// * return the number of notified receivers.
        ///
        /// NOTE: this method is without timeout or deadline, it will be blocking forever if no space available.
        #[inline(always)]
        pub fn send(&self, mut msg: T) -> usize {
            loop {
                match self.try_send(msg) {
                    Ok(count) => {
                        return count;
                    },
                    Err(m) => {
                        msg = m;

                        // listener macro create listener on stack.
                        // avoid recv_event.listen() due to it alloc heap.
                        listener!(self.recv_event => recv_event_listener);

                        // waiting until recv event happen.
                        recv_event_listener.wait();
                    }
                }
            }
        }

        /// timeout version of `send`.
        /// * if channel is full, this method will blocking until have space or timed out.
        /// * the return value is same as `try_send`.
        #[inline(always)]
        pub fn send_timeout(&self, msg: T, timeout: Duration) -> Result<usize, T> {
            match Instant::now().checked_add(timeout) {
                Some(deadline) => self.send_deadline(msg, deadline),
                _ => Err(msg),
            }
        }

        /// deadline version of `send`.
        /// * if channel is full, this method will blocking until have space or "current time is after the deadline".
        /// * the return value is same as `try_send`.
        #[inline(always)]
        pub fn send_deadline(&self, mut msg: T, deadline: Instant) -> Result<usize, T> {
            while Instant::now() < deadline {
                match self.try_send(msg) {
                    Ok(count) => {
                        return Ok(count);
                    },
                    Err(m) => {
                        msg = m;

                        // listener macro create listener on stack.
                        // avoid recv_event.listen() due to it alloc heap.
                        listener!(self.recv_event => recv_event_listener);

                        // waiting until recv event happen.
                        recv_event_listener.wait_deadline(deadline);
                    }
                }
            }

            Err(msg)
        }

        /// try to receive one message from channel.
        /// * return Some(T) if successfully to receive the message.
        /// * return None if this channel is empty.
        #[inline(always)]
        pub fn try_recv(&self) -> Option<T> {
            let maybe = self.list.pop();
            if maybe.is_some() {
                self.recv_event.notify_relaxed(1);
            }
            maybe
        }

        /// the blocking version of `try_recv`.
        /// * if there is no messages in channel, this method will blocking until have one.
        /// * return the received message.
        ///
        /// NOTE: this method is without timeout or deadline, it will be blocking forever if no messages available.
        #[inline(always)]
        pub fn recv(&self) -> T {
            loop {
                match self.try_recv() {
                    Some(msg) => {
                        return msg;
                    },
                    _ => {
                        // listener macro create listener on stack.
                        // avoid send_event.listen() due to it alloc heap.
                        listener!(self.send_event => send_event_listener);

                        // waiting until send event happen.
                        send_event_listener.wait();
                    },
                }
            }
        }

        /// timeout version of `recv`.
        /// * if channel is empty, this method will blocking until have one or timed out.
        /// * the return value is same as `try_recv`.
        #[inline(always)]
        pub fn recv_timeout(&self, timeout: Duration) -> Option<T> {
            match Instant::now().checked_add(timeout) {
                Some(deadline) => self.recv_deadline(deadline),
                _ => None,
            }
        }

        /// deadline version of `recv`.
        /// * if channel is empty, this method will blocking until have one or "current time is after the deadline".
        /// * the return value is same as `try_recv`.
        #[inline(always)]
        pub fn recv_deadline(&self, deadline: Instant) -> Option<T> {
            while Instant::now() < deadline {
                match self.list.pop() {
                    Some(msg) => {
                        return Some(msg);
                    },
                    _ => {
                        // listener macro create listener on stack.
                        // avoid send_event.listen() due to it alloc heap.
                        listener!(self.send_event => send_event_listener);

                        // waiting until send event happen.
                        send_event_listener.wait_deadline(deadline);
                    }
                }
            }

            None
        }
    }

    /// the send side of Channel.
    pub struct Sender<T, L: List<T>>(Channel<T, L>);

    impl<T, L: List<T>> fmt::Debug for Sender<T, L> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_tuple("Sender")
             .field(&self.0)
             .finish()
        }
    }

    impl<T, L: List<T>> Clone for Sender<T, L> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }

    impl<T, L: List<T>> Sender<T, L> {
        /// see [`Channel::len()`].
        #[inline(always)]
        pub fn len(&self) -> usize {
            self.0.len()
        }

        /// see [`Channel::try_send()`].
        #[inline(always)]
        pub fn try_send(&self, msg: T) -> Result<usize, T> {
            self.0.try_send(msg)
        }

        /// see [`Channel::send()`].
        #[inline(always)]
        pub fn send(&self, msg: T) -> usize {
            self.0.send(msg)
        }

        /// see [`Channel::send_timeout()`].
        #[inline(always)]
        pub fn send_timeout(&self, msg: T, timeout: Duration) -> Result<usize, T> {
            self.0.send_timeout(msg, timeout)
        }

        /// see [`Channel::send_deadline()`].
        #[inline(always)]
        pub fn send_deadline(&self, msg: T, deadline: Instant) -> Result<usize, T> {
            self.0.send_deadline(msg, deadline)
        }
    }

    /// the receive side of Channel.
    pub struct Receiver<T, L: List<T>>(Channel<T, L>);

    impl<T, L: List<T>> fmt::Debug for Receiver<T, L> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_tuple("Receiver")
             .field(&self.0)
             .finish()
        }
    }

    impl<T, L: List<T>> Clone for Receiver<T, L> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }

    impl<T, L: List<T>> Receiver<T, L> {
        /// see [`Channel::len()`].
        #[inline(always)]
        pub fn len(&self) -> usize {
            self.0.len()
        }

        /// see [`Channel::try_recv()`].
        #[inline(always)]
        pub fn try_recv(&self) -> Option<T> {
            self.0.try_recv()
        }

        /// see [`Channel::recv()`].
        #[inline(always)]
        pub fn recv(&self) -> T {
            self.0.recv()
        }

        /// see [`Channel::recv_timeout()`].
        #[inline(always)]
        pub fn recv_timeout(&self, timeout: Duration) -> Option<T> {
            self.0.recv_timeout(timeout)
        }

        /// see [`Channel::recv_deadline()`].
        #[inline(always)]
        pub fn recv_deadline(&self, deadline: Instant) -> Option<T> {
            self.0.recv_deadline(deadline)
        }
    }
}

/// the "unordered set" similar to scc::Bag but without the limit of maximum capacity.
#[derive(Debug)]
pub struct Storage<T, const N: usize> {
    array: [AtomicOwned<T>; N],
    len: AtomicUsize,

    has: [AtomicBool; N],
    last_nothing: AtomicUsize,
    last_something: AtomicUsize,
}

impl<T: 'static, const N: usize> Default for Storage<T, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static, const N: usize> From<[AtomicOwned<T>; N]> for Storage<T, N> {
    fn from(array: [AtomicOwned<T>; N]) -> Storage<T, N> {

        let mut len = 0;
        let mut has = [false; N];
        let mut last_nothing = 0;
        let mut last_something = 0;

        let g = scc2::ebr::Guard::new();
        for i in 0..N {
            if array[i].load(Relaxed, &g).as_ref().is_some() {
                len += 1;
                has[i] = true;
                last_something = i;
            } else {
                last_nothing = i;
            }
        }

        let len = AtomicUsize::new(len);
        let has = has.map(AtomicBool::new);
        let last_nothing = AtomicUsize::new(last_nothing);
        let last_something = AtomicUsize::new(last_something);

        Self {
            array,
            len,
            has,
            last_nothing,
            last_something,
        }
    }
}

impl<T: 'static, const N: usize> From<[T; N]> for Storage<T, N> {
    fn from(val: [T; N]) -> Storage<T, N> {
        val.map(AtomicOwned::new).into()
    }
}

impl<T: 'static, const N: usize> Storage<T, N> {
    /// create new empty [`Storage`].
    ///
    /// panic if N == 0.
    #[inline(always)]
    pub const fn new() -> Self {
        assert!(N > 0);

        Self {
            array: [const { AtomicOwned::null() }; N],
            len: AtomicUsize::new(0),

            has: [const { AtomicBool::new(false) }; N],
            last_nothing: AtomicUsize::new(0),
            last_something: AtomicUsize::new(0),
        }
    }

    /// the current length of items in Storage.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len.load(Relaxed)
    }

    /// push item to Storage.
    #[inline(always)]
    pub fn push(&self, item: T) -> Result<(), scc2::ebr::Owned<T>> {
        self.push_owned(scc2::ebr::Owned::new(item))
    }

    /// push Owned-wrapped item to Storage.
    #[inline(always)]
    pub fn push_owned(&self, mut owned: scc2::ebr::Owned<T>) -> Result<(), scc2::ebr::Owned<T>> {
        use scc2::ebr::{Ptr, Tag};

        if self.len() >= N {
            return Err(owned);
        }

        let g = scc2::ebr::Guard::new();

        let mut idx = [0usize; N];
        idx[0] = self.last_nothing.load(Relaxed);

        let mut ii = 1;
        for i in 0..N {
            if idx[0] == i {
                continue;
            }
            idx[ii] = i;
            ii += 1;
        }

        for i in idx.into_iter() {
            if self.has[i].load(Relaxed) {
                continue;
            }

            match
                self.array[i].compare_exchange(
                    Ptr::null(),
                    (Some(owned), Tag::None),
                    Relaxed,
                    Relaxed,
                    &g
                )
            {
                Ok((prev, _new)) => {
                    assert!(prev.is_none());

                    self.has[i].store(true, Relaxed);
                    self.last_something.store(i, Relaxed);

                    self.len.checked_add(1);
                    return Ok(());
                },
                Err((o, cur)) => {
                    owned = o.unwrap();

                    assert!(! cur.is_null());
                    self.has[i].store(true, Relaxed);

                    self.last_something.store(i, Relaxed);
                }
            }
        }

        Err(owned)
    }

    /// pop item from Storage.
    #[inline(always)]
    pub fn pop(&self) -> Option<scc2::ebr::Owned<T>> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        let mut idx = [0usize; N];
        idx[0] = self.last_something.load(Relaxed);

        let mut ii = 1;
        for i in 0..N {
            if idx[0] == i {
                continue;
            }
            idx[ii] = i;
            ii += 1;
        }

        let mut maybe_owned;
        for i in idx.into_iter() {
            if self.has[i].load(Relaxed) {
                maybe_owned = self.array[i].swap(
                    (None, scc2::ebr::Tag::None),
                    Relaxed
                ).0;

                self.has[i].store(false, Relaxed);
                self.last_nothing.store(i, Relaxed);

                if let Some(owned) = maybe_owned {
                    self.len.checked_sub(1);
                    return Some(owned);
                }
            }
        }
        None
    }
}

/// the multi [`Storage`]s backed by linked list.
pub struct LinkedStorage<T, const N: usize> {
    inner: Storage<T, N>,
    next: AtomicOwned<Self>,
}

impl<T: 'static, const N: usize> Default for LinkedStorage<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static, const N: usize> From<Storage<T, N>> for LinkedStorage<T, N> {
    fn from(inner: Storage<T, N>) -> LinkedStorage<T, N> {
        Self {
            inner,
            next: AtomicOwned::null(),
        }
    }
}

impl<T: 'static, const N: usize> LinkedStorage<T, N> {
    /// create new [`LinkedStorage`].
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            inner: Storage::new(),
            next: AtomicOwned::null(),
        }
    }

    #[inline(always)]
    fn try_next<'g>(&self, guard: &'g scc2::ebr::Guard) -> Option<&'g Self> {
        self.next.load(Relaxed, guard).as_ref()
    }

    /// get the next instance.
    #[inline(always)]
    fn next<'g>(&self, guard: &'g scc2::ebr::Guard) -> &'g Self {
        let mut ptr = self.next.load(Relaxed, guard);
        let mut maybe_next;
        let mut new = None;
        loop {
            maybe_next = ptr.as_ref();

            if maybe_next.is_some() {
                return maybe_next.unwrap();
            }
            assert!(ptr.is_null());

            if new.is_none() {
                new = Some(scc2::ebr::Owned::new(Self::new()));
            }

            match
                self.next.compare_exchange(
                    ptr,
                    (new, scc2::ebr::Tag::None),
                    Relaxed,
                    Relaxed,
                    guard,
                )
            {
                Ok((prev, new)) => {
                    assert!(prev.is_none());
                    maybe_next = new.as_ref();
                    assert!(maybe_next.is_some());
                    return maybe_next.unwrap();
                },
                Err((n, cur)) => {
                    new = n;
                    ptr = cur;
                }
            }
        }
    }

    /// try push `T` to this instance. if full, push to the next instance.
    #[inline(always)]
    pub fn push(&self, item: T) {
        self.push_owned(scc2::ebr::Owned::new(item))
    }

    /// try push `Owned<T>` to this instance. if full, push to the next instance.
    #[inline(always)]
    pub fn push_owned(&self, mut owned: scc2::ebr::Owned<T>) {
        let g = scc2::ebr::Guard::new();
        let mut this = self;
        while let Err(o) = this.inner.push_owned(owned) {
            owned = o;
            this = this.next(&g);
        }
    }

    /// try pop `Owned<T>` from this instance. if empty, try looking for the next instance if any.
    #[inline(always)]
    pub fn pop(&self) -> Option<scc2::ebr::Owned<T>> {
        let g = scc2::ebr::Guard::new();
        let mut this = self;
        let mut maybe;
        loop {
            maybe = this.inner.pop();
            if maybe.is_some() {
                return maybe;
            }

            if let Some(next) = this.try_next(&g) {
                this = next;
            } else {
                return None;
            }
        }
    }
}

/// some hacky methods
pub mod hack {
    use super::*;

    /// try to get the zero point of Instant.
    #[inline(always)]
    pub fn instant_zero() -> Instant {
        let t = Instant::now();
        t - instant_to_duration(t)
    }

    /// try to get internal value of Instant.
    #[inline(always)]
    pub fn instant_to_duration(t: Instant) -> Duration {
        // method 1: try to dump the internal value.
        for _ in core::iter::once(()) {
            let mut dh = DumpHasher::new();

            use core::hash::Hash;
            t.hash(&mut dh);

            let dh = dh.data();

            use HashWrite::*;
            match dh.len() {
                2 => {
                    let secs =
                        match dh[0] {
                            U64(v) => v,
                            I64(v) => {
                                if v >= 0 {
                                    v as u64
                                } else {
                                    break;
                                }
                            },
                            _ => {
                                break;
                            }
                        };

                    if let U32(subsec_nanos) = dh[1] {
                        #[cfg(test)]
                        dbg!({"instant_to_duration method 1";}, dh, t);

                        return Duration::new(secs, subsec_nanos);
                    }
                },
                _ => {
                    break;
                }
            }

            break;
        }

        // method 1 fails.
        // try doing method 2: sub to zero.
        let mut val = Duration::new(0, 0);
        let mut d = Duration::new(1, 0);
        let mut errs = 0;

        let mut tmp = t;
        const L: Duration = Duration::new(0, 1);
        while errs < 100000 {
            if let Some(v) = tmp.checked_sub(d) {
                if ! format!("{v:?}").contains('-') {
                    tmp = v;
                    val += d;
                    d *= 3;
                    continue;
                }
            }

            if d <= L {
                break;
            }

            errs += 1;
            d /= 2;
        }

        #[cfg(test)]
        dbg!({"instant_to_duration method 2";}, val, t, d, errs);

        val
    }
}

macro_rules! gen_hash_write {
    ($($v:ident = $t:ty,)*) => {
        /// the value types written to hasher.
        #[derive(Debug, Clone)]
        #[repr(u8)]
        pub enum HashWrite {
            $(
                #[doc = concat!("the [`", stringify!($t), "`] value.")]
                $v($t),
            )*
        }

        $(
            impl From<$t> for HashWrite {
                #[inline(always)]
                fn from(val: $t) -> HashWrite {
                    HashWrite::$v(val)
                }
            }

            impl TryFrom<HashWrite> for $t {
                type Error = HashWrite;

                #[inline(always)]
                fn try_from(val: HashWrite) -> Result<$t, HashWrite> {
                    match val {
                        HashWrite::$v(inner) => {
                            Ok(inner)
                        },
                        other => {
                            Err(other)
                        }
                    }
                }
            }
        )*

        impl From<&[u8]> for HashWrite {
            #[inline(always)]
            fn from(val: &[u8]) -> HashWrite {
                HashWrite::Bytes(val.to_vec())
            }
        }
    }
}

gen_hash_write!(
    Bytes = Vec<u8>,

    U8    = u8,
    U16   = u16,
    U32   = u32,
    U64   = u64,
    U128  = u128,
    Usize = usize,

    I8    = i8,
    I16   = i16,
    I32   = i32,
    I64   = i64,
    I128  = i128,
    Isize = isize,
);

/// A pseudo-hasher that used by dump the internal or private field of provided value.
///
/// this DumpHasher does not doing hash data anymore, it just save all of data passed to Hasher trait to a Vec.
#[derive(Debug, Clone)]
pub struct DumpHasher {
    data: Vec<HashWrite>,
}

impl DumpHasher {
    /// create new DumpHasher instance.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
        }
    }

    /// get the data before hashed.
    #[inline(always)]
    pub const fn data<'a>(&'a self) -> &'a Vec<HashWrite> {
        &(self.data)
    }
}

macro_rules! dump_hasher_impl {
    ($($n:ident = $t:ty,)*) => {
        impl core::hash::Hasher for DumpHasher {
            #[inline(always)]
            fn finish(&self) -> u64 {
                0
            }

            $(
                #[inline(always)]
                fn $n(&mut self, val: $t) {
                    self.data.push(val.into());
                }
            )*
        }
    }
}

dump_hasher_impl!(
    write       = &[u8],

    write_u8    = u8,
    write_u16   = u16,
    write_u32   = u32,
    write_u64   = u64,
    write_u128  = u128,
    write_usize = usize,

    write_i8    = i8,
    write_i16   = i16,
    write_i32   = i32,
    write_i64   = i64,
    write_i128  = i128,
    write_isize = isize,
);

/// the atomic version of Duration.
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

impl From<Duration> for AtomicDuration {
    fn from(val: Duration) -> Self {
        Self::from_nanos(val.as_nanos())
    }
}
impl From<AtomicDuration> for Duration {
    fn from(val: AtomicDuration) -> Self {
        val.get()
    }
}

impl AtomicDuration {
    /// one second is equal to one billion nanoseconds.
    const NANOS_PER_SEC: u128 = 1_000_000_000;

    /// convert from seconds to nanoseconds.
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

    /// return the zero (default) value of AtomicDuration.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self::new(0, 0)
    }

    /// create new AtomicDuration from u128 nanoseconds.
    #[inline(always)]
    pub const fn from_nanos(val: u128) -> Self {
        Self {
            total_ns: AtomicU128::new(val),
            changed: None,
        }
    }

    /// create new AtomicDuration from Duration.
    #[inline(always)]
    pub const fn from(val: Duration) -> Self {
        Self::from_nanos(val.as_nanos())
    }

    /// create new AtomicDuration with seconds and (subsecond) nanos.
    #[inline(always)]
    pub const fn new(s: u64, n: u32) -> Self {
        let ns = Self::secs_to_nanos(s) + (n as u128);
        Self {
            total_ns: AtomicU128::new(ns),
            changed: None,
        }
    }

    /// create new AtomicDuration with trace whether it's changed.
    #[inline(always)]
    pub const fn new_with_trace(s: u64, n: u32) -> Self {
        let mut this = Self::new(s, n);
        this.trace_change();
        this
    }

    /// make this instance of AtomicDuration to trace change.
    #[inline(always)]
    pub const fn trace_change(&mut self) -> &mut Self {
        self.changed = Some(AtomicBool::new(false));
        self
    }

    /// set seconds of this AtomicDuration.
    #[inline(always)]
    pub fn set_secs(&self, s: u64) -> &Self {
        self.set(Duration::new(s, self.subsec_nanos()))
    }

    /// set sub-seconds (unit nanoseconds) of this AtomicDuration.
    #[inline(always)]
    pub fn set_subsec_nanos(&self, n: u32) -> &Self {
        self.set(Duration::new(self.as_secs(), n))
    }

    /// set seconds + subseconds (by Duration) of this AtomicDuration.
    #[inline(always)]
    pub fn set(&self, d: Duration) -> &Self {
        self.total_ns.store(d.as_nanos(), Relaxed);
        if let Some(ref changed) = self.changed {
            changed.store(true, Relaxed);
        }
        self
    }

    /// add seconds to this AtomicDuration.
    #[inline(always)]
    pub fn add_secs(&self, s: u64) -> &Self {
        if self.total_ns.checked_add(Self::secs_to_nanos(s)).is_some() {
            if let Some(ref changed) = self.changed {
                changed.store(true, Relaxed);
            }
        }
        self
    }

    /// add nanoseconds to this AtomicDuration.
    /// if `n > 1_000_000_000`, extra seconds will be added to AtomicDuration.
    #[inline(always)]
    pub fn add_nanos(&self, n: u128) -> &Self {
        if self.total_ns.checked_add(n).is_some() {
            if let Some(ref changed) = self.changed {
                changed.store(true, Relaxed);
            }
        }
        self
    }

    /// add Duration to this AtomicDuration.
    #[inline(always)]
    pub fn add(&self, d: Duration) -> &Self {
        if self.total_ns.checked_add(d.as_nanos()).is_some() {
            if let Some(ref changed) = self.changed {
                changed.store(true, Relaxed);
            }
        }
        self
    }

    /// sub Duration to this AtomicDuration.
    #[inline(always)]
    pub fn sub(&self, d: Duration) -> &Self {
        if self.total_ns.checked_sub(d.as_nanos()).is_some() {
            if let Some(ref changed) = self.changed {
                changed.store(true, Relaxed);
            }
        }
        self
    }

    /// get the seconds (without subseconds) of this AtomicDuration.
    #[inline(always)]
    pub fn as_secs(&self) -> u64 {
        self.get().as_secs()
    }

    /// get the subseconds (without seconds) of this AtomicDuration.
    #[inline(always)]
    pub fn subsec_nanos(&self) -> u32 {
        self.get().subsec_nanos()
    }

    /// get the Duration from this AtomicDuration.
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

/// the atomic version of [`Instant`].
#[derive(Debug)]
pub struct AtomicInstant {
    anchor: OnceCell<Instant>,
    offset: AtomicDuration,
    op: AtomicU8,
    init_zero: bool,
}

impl AtomicInstant {
    /// No Operation.
    pub const OP_NOP:     u8 = b'=';

    /// Add offset to anchor.
    pub const OP_ADD:     u8 = b'+';

    /// Sub offset from anchor.
    pub const OP_SUB:     u8 = b'-';

    /// intermediate state.
    pub const OP_PENDING: u8 = b'.';

    /// create new un-initialized AtomicInstant.
    #[inline(always)]
    pub const fn init() -> Self {
        Self {
            anchor: OnceCell::new(),
            offset: AtomicDuration::zero(),
            op: AtomicU8::new(Self::OP_NOP),
            init_zero: true,
        }
    }

    /// create new AtomicInstant from currently monotonic time from [`Instant::now()`].
    #[inline(always)]
    pub const fn init_now() -> Self {
        let mut this = Self::init();
        this.init_zero = false;
        this
    }

    /// create new AtomicInstant with provided value.
    #[inline(always)]
    pub const fn new(anchor: Instant) -> Self {
        let mut this = Self::init();
        this.anchor = OnceCell::with_value(anchor);
        this
    }

    /// create and initialize the "zero" value of AtomicInstant.
    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(hack::instant_zero())
    }

    /// create and initialize the "now" value of AtomicInstant.
    #[inline(always)]
    pub fn now() -> Self {
        Self::new(Instant::now())
    }

    /// try to peek the anchor value without initialize it.
    #[inline(always)]
    pub fn peek_anchor(&self) -> Option<Instant> {
        self.anchor.get().copied()
    }

    /// get the anchor value or initialize it.
    #[inline(always)]
    pub fn anchor(&self) -> Instant {
        *(self.anchor.get_or_init(if self.init_zero { hack::instant_zero } else { Instant::now }))
    }

    /// get the current offset value of AtomicInstant.
    #[inline(always)]
    pub fn offset(&self) -> Duration {
        self.offset.get()
    }

    /// get the operation "flag".
    #[inline(always)]
    pub fn op(&self) -> u8 {
        let mut op;
        loop {
            op = self.op.load(Relaxed);
            if op != Self::OP_PENDING {
                return op;
            }
        }
    }

    /// lock the operation "flag".
    #[inline(always)]
    fn op_lock(&self) -> u8 {
        let mut op;
        loop {
            op = self.op.load(Relaxed);
            if op != Self::OP_PENDING {
                if self.op.compare_exchange(op, Self::OP_PENDING, Relaxed, Relaxed).is_ok() {
                    return op;
                }
            }
        }
    }

    /// set this AtomicInstant.
    #[inline(always)]
    pub fn set(&self, t: Instant) -> &Self {
        self.op_lock();

        let anchor = self.anchor();
        let op =
            if t == anchor {
                Self::OP_NOP
            } else if t > anchor {
                self.offset.set(t - anchor);
                Self::OP_ADD
            } else {
                self.offset.set(anchor - t);
                Self::OP_SUB
            };
        self.op.store(op, Relaxed);

        self
    }

    /// get this AtomicInstant.
    #[inline(always)]
    pub fn get(&self) -> Instant {
        self._calc(self.anchor())
    }

    /// peek the Instant without initialize it.
    #[inline(always)]
    pub fn peek(&self) -> Option<Instant> {
        let anchor = self.peek_anchor()?;
        Some(self._calc(anchor))
    }

    /// calculate anchor +/- (add or sub) of offset
    #[inline(always)]
    fn _calc(&self, anchor: Instant) -> Instant {
        let op = self.op();
        match op {
            Self::OP_NOP => {
                anchor
            },
            Self::OP_ADD => {
                anchor + self.offset.get()
            },
            Self::OP_SUB => {
                anchor - self.offset.get()
            },
            _ => {
                panic!("unexpected invalid value of self.op");
            }
        }
    }

    /// add (+) Duration to AtomicInstant.
    #[inline(always)]
    pub fn add(&self, d: Duration) -> &Self {
        let mut op = self.op_lock();
        match op {
            Self::OP_ADD => {
                self.offset.add(d);
            },
            Self::OP_SUB => {
                let pd = self.offset.get();
                if pd >= d {
                    self.offset.sub(d);
                } else {
                    op = Self::OP_ADD;
                    self.offset.set(d - pd);
                }
            },
            Self::OP_NOP => {
                op = Self::OP_ADD;
                self.offset.set(d);
            },
            _ => {
                panic!("unexpected invalid value of self.op");
            }
        }
        self.op.store(op, Relaxed);

        self
    }

    /// sub (-) Duration to AtomicInstant.
    #[inline(always)]
    pub fn sub(&self, d: Duration) -> &Self {
        let mut op = self.op_lock();
        match op {
            Self::OP_ADD => {
                let pd = self.offset.get();
                if pd >= d {
                    self.offset.sub(d);
                } else {
                    op = Self::OP_SUB;
                    self.offset.set(d - pd);
                }
            },
            Self::OP_SUB => {
                self.offset.add(d);
            },
            Self::OP_NOP => {
                op = Self::OP_SUB;
                self.offset.set(d);
            },
            _ => {
                panic!("unexpected invalid value of self.op");
            }
        }
        self.op.store(op, Relaxed);

        self
    }
}

/// the atomic version of [`Ipv4Addr`].
#[derive(Debug)]
pub struct AtomicIpv4Addr {
    bits: AtomicU32,
}

impl Default for AtomicIpv4Addr {
    #[inline(always)]
    fn default() -> Self {
        const { Self::default() }
    }
}

impl AtomicIpv4Addr {
    /// the default value (0.0.0.0) of AtomicIpv4Addr.
    #[inline(always)]
    pub const fn default() -> Self {
        Self::new(Ipv4Addr::UNSPECIFIED)
    }

    /// create AtomicIpv4Addr with provided value.
    #[inline(always)]
    pub const fn new(ip4: Ipv4Addr) -> Self {
        Self {
            bits:
                AtomicU32::new(
                    u32::from_be_bytes(ip4.octets())
                )
        }
    }

    /// get bits of AtomicIpv4Addr.
    #[inline(always)]
    pub fn get_bits(&self) -> u32 {
        self.bits.load(Relaxed)
    }

    /// set bits of AtomicIpv4Addr.
    #[inline(always)]
    pub fn set_bits(&self, bits: u32) -> &Self {
        self.bits.store(bits, Relaxed);
        self
    }

    /// get Ipv4Addr from AtomicIpv4Addr.
    #[inline(always)]
    pub fn get(&self) -> Ipv4Addr {
        Ipv4Addr::from(self.get_bits().to_be_bytes())
    }

    /// set Ipv4Addr from AtomicIpv4Addr.
    #[inline(always)]
    pub fn set(&self, ip4: Ipv4Addr) -> &Self {
        self.set_bits(u32::from_be_bytes(ip4.octets()))
    }
}

/// the atomic version of [`Ipv6Addr`].
#[derive(Debug)]
pub struct AtomicIpv6Addr {
    bits: AtomicU128,
}

impl Default for AtomicIpv6Addr {
    #[inline(always)]
    fn default() -> Self {
        const { Self::default() }
    }
}

impl AtomicIpv6Addr {
    /// the default value (::) of AtomicIpv6Addr.
    #[inline(always)]
    pub const fn default() -> Self {
        Self::new(Ipv6Addr::UNSPECIFIED)
    }

    /// create new AtomicIpv6Addr from provided value.
    #[inline(always)]
    pub const fn new(ip6: Ipv6Addr) -> Self {
        Self {
            bits:
                AtomicU128::new(
                    u128::from_be_bytes(ip6.octets())
                )
        }
    }

    /// get bits of AtomicIpv6Addr.
    #[inline(always)]
    pub fn get_bits(&self) -> u128 {
        self.bits.load(Relaxed)
    }

    /// set bits of AtomicIpv6Addr.
    #[inline(always)]
    pub fn set_bits(&self, bits: u128) -> &Self {
        self.bits.store(bits, Relaxed);
        self
    }

    /// get Ipv6Addr from AtomicIpv6Addr.
    #[inline(always)]
    pub fn get(&self) -> Ipv6Addr {
        Ipv6Addr::from(self.get_bits().to_be_bytes())
    }

    /// set Ipv6Addr from AtomicIpv6Addr.
    #[inline(always)]
    pub fn set(&self, ip6: Ipv6Addr) -> &Self {
        self.set_bits(u128::from_be_bytes(ip6.octets()))
    }
}

/// the atomic version of [`IpAddr`].
#[derive(Debug)]
pub struct AtomicIpAddr {
    kind: AtomicU8,
    data: AtomicU128,
}

impl Default for AtomicIpAddr {
    #[inline(always)]
    fn default() -> Self {
        const { Self::default() }
    }
}

impl AtomicIpAddr {
    /// IPv4.
    pub const KIND_IPV4:    u8 = 4;

    /// IPv6.
    pub const KIND_IPV6:    u8 = 6;

    /// intermediate state.
    pub const KIND_PENDING: u8 = 0xfe;

    /// default value (0.0.0.0) of AtomicIpAddr.
    #[inline(always)]
    pub const fn default() -> Self {
        Self::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED))
    }

    /// create new AtomicIpAddr from provided value.
    #[inline(always)]
    pub const fn new(ip: IpAddr) -> Self {
        let (kind, data) = Self::ip2tuple(ip);
        Self {
            kind: AtomicU8::new(kind),
            data: AtomicU128::new(data),
        }
    }

    /// convert IpAddr to (kind, data)
    #[inline(always)]
    const fn ip2tuple(ip: IpAddr) -> (u8, u128) {
        let kind;
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

    /// the kind of IpAddr.
    #[inline(always)]
    pub fn kind(&self) -> u8 {
        self.kind.load(Relaxed)
    }

    /// whether is AtomicIpAddr is IPv4?
    #[inline(always)]
    pub fn is_ipv4(&self) -> bool {
        self.kind() == Self::KIND_IPV4
    }

    /// whether is AtomicIpAddr is IPv6?
    #[inline(always)]
    pub fn is_ipv6(&self) -> bool {
        self.kind() == Self::KIND_IPV6
    }

    /// data part of AtomicIpAddr.
    #[inline(always)]
    fn data(&self) -> u128 {
        self.data.load(Relaxed)
    }

    /// get IpAddr from AtomicIpAddr.
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

    /// set IpAddr from AtomicIpAddr.
    #[inline(always)]
    pub fn set(&self, ip: IpAddr) -> &Self {
        self.kind.store(Self::KIND_PENDING, Relaxed);

        let (kind, data) = Self::ip2tuple(ip);

        self.data.store(data, Relaxed);
        self.kind.store(kind, Relaxed);

        self
    }
}

/// the atomic version of [`SocketAddr`].
#[derive(Debug)]
pub struct AtomicSocketAddr {
    ready: AtomicBool,
    ip: AtomicIpAddr,
    port: AtomicU16,
}

impl Default for AtomicSocketAddr {
    fn default() -> Self {
        const { Self::default() }
    }
}

impl AtomicSocketAddr {
    /// default value (0.0.0.0:0) of AtomicSocketAddr.
    #[inline(always)]
    pub const fn default() -> Self {
        Self::new(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0))
    }

    /// create new AtomicSocketAddr from provided value.
    #[inline(always)]
    pub const fn new(addr: SocketAddr) -> Self {
        Self {
            ready: AtomicBool::new(true),
            ip: AtomicIpAddr::new(addr.ip()),
            port: AtomicU16::new(addr.port()),
        }
    }

    /// get the IP address of this AtomicSocketAddr.
    #[inline(always)]
    pub fn ip(&self) -> IpAddr {
        self.ip.get()
    }

    /// get the port of this AtomicSocketAddr.
    #[inline(always)]
    pub fn port(&self) -> u16 {
        self.port.load(Relaxed)
    }

    /// get SocketAddr from this AtomicSocketAddr.
    #[inline(always)]
    pub fn get(&self) -> SocketAddr {
        while ! self.ready.load(Relaxed) {
            // waiting...
        }
        SocketAddr::new(self.ip(), self.port())
    }

    /// set SocketAddr from this AtomicSocketAddr.
    #[inline(always)]
    pub fn set(&self, addr: SocketAddr) -> &Self {
        self.ready.store(false, Relaxed);
        self.ip.set(addr.ip());
        self.port.store(addr.port(), Relaxed);
        self.ready.store(true, Relaxed);

        self
    }
}

/// the atomic version of [`Range`].
#[derive(Debug)]
pub struct AtomicRange<AT: fmt::Debug> {
    start: AT,
    end: AT,

    strict: bool,
}

/// strict version of AtomicRange.
#[derive(Debug)]
pub struct AtomicRangeStrict<AT: fmt::Debug>(AtomicRange<AT>);

/// number iterator.
#[derive(Debug, Clone)]
pub struct NumIter<T>
where
    T: fmt::Debug + Clone + PartialOrd + ConstZero + ConstOne + Bounded + CheckedAdd + CheckedSub,
{
    stopped: bool,

    current: T,
    until: T,

    step: T,
    incr: bool,
}

impl<T> NumIter<T>
where
    T: fmt::Debug + Clone + PartialOrd + ConstZero + ConstOne + Bounded + CheckedAdd + CheckedSub + Euclid,
{
    /// create new NumIter.
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

    /// the remaining size of NumIter.
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
    T:  fmt::Debug
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
        impl<AT: fmt::Debug> AtomicRange<AT> {
            $(
                #[doc = concat!("create AtomicRange with ", stringify!($num), " value.")]
                #[inline(always)]
                pub const fn $num() -> AtomicRange<$atom> {
                    AtomicRange::<$atom>::default()
                }
            )*
        }
        impl<AT: fmt::Debug> AtomicRangeStrict<AT> {
            $(
                #[doc = concat!("create AtomicRangeStrict with ", stringify!($num), " value.")]
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
                /// default value (0..0) of AtomicRange.
                #[inline(always)]
                pub const fn default() -> Self {
                    const ZERO: $num = 0u8 as $num;
                    Self {
                        start: <$atom>::new(ZERO),
                        end: <$atom>::new(ZERO),

                        strict: false,
                    }
                }

                /// create new AtomicRange.
                #[inline(always)]
                pub const fn new(start: $num, end: $num) -> Self {
                    Self {
                        start: <$atom>::new(start),
                        end: <$atom>::new(end),

                        strict: false,
                    }
                }

                /// convert Range to AtomicRange.
                #[inline(always)]
                pub const fn from_range(val: Range<$num>) -> Self {
                    Self::new(val.start, val.end)
                }

                /// make this AtomicRange strictly. this is cannot undo.
                #[inline(always)]
                pub const fn strict(&mut self) -> &mut Self {
                    self.strict = true;
                    self
                }

                /// the start value of AtomicRange.
                #[inline(always)]
                pub fn start(&self) -> $num {
                    self.start.load(Relaxed)
                }

                /// the (excluded) end value of AtomicRange.
                #[inline(always)]
                pub fn end(&self) -> $num {
                    self.end.load(Relaxed)
                }

                /// get Range from this AtomicRange.
                #[inline(always)]
                pub fn range(&self) -> Range<$num> {
                    Range {
                        start: self.start(),
                        end: self.end(),
                    }
                }

                /// set the start value of AtomicRange.
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

                /// set the end value of AtomicRange.
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

                /// set Range to this AtomicRange.
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

                /// create iterator from this AtomicRange.
                #[inline(always)]
                pub fn iter(&self) -> NumIter<$num> {
                    NumIter::new(self.start(), self.end(), $num::ONE)
                }

                /// whether this AtomicRange contains the provided value?
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
                /// default value (0..1) of AtomicRangeStrict.
                #[inline(always)]
                pub const fn default() -> Self {
                    Self(AtomicRange {
                        start: <$atom>::new(0u8 as _),
                        end: <$atom>::new(1u8 as _),

                        strict: true,
                    })
                }

                /// create new AtomicRangeStrict.
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

/// unused. "stable clock" that does not jump during system hibernate.
#[derive(Debug)]
pub struct StableClock {
    started_elapsed: scc2::Atom<(Instant, AtomicDuration)>,
    tick: Duration,
    join_handle: std::thread::JoinHandle<()>,
}

impl StableClock {
    /// get the global instance of StableClock.
    #[inline(always)]
    pub fn global() -> &'static Self {
        static GLOBAL: Lazy<StableClock> = Lazy::new(StableClock::default);

        &*GLOBAL
    }

    /// whether the time is jumped?
    #[inline(always)]
    pub fn is_time_jumped(&self) -> bool {
        self.time_diff() > (self.tick * 10)
    }

    /// the different between StableClock and Instant now.
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

    /// the "stable" now.
    #[inline(always)]
    pub fn now(&self) -> Instant {
        if self.join_handle.is_finished() {
            panic!("unexpectedly StableClock tick thread exited!");
        }

        let (started, elapsed) = &*(self.started_elapsed.get().expect("no value"));
        let elapsed = elapsed.add_nanos(1).get();
        (*started) + elapsed
    }

    /// private default of StableClock.
    #[inline(always)]
    fn default() -> Self {
        Self::new(Duration::from_millis(100))
    }

    /// private new of StableClock.
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
                            se = scc2::ebr::Shared::new((Instant::now(), AtomicDuration::new(0, 0)));
                            this.started_elapsed.set_shared(se.clone());
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

