use core::net::{IpAddr, Ipv4Addr, Ipv6Addr};

pub trait Ip: Sized {
    fn from_ip(ip: IpAddr) -> Self;
    fn to_ip(&self) -> IpAddr;
}

impl Ip for IpAddr {
    #[inline(always)]
    fn from_ip(ip: IpAddr) -> IpAddr {
        ip
    }

    #[inline(always)]
    fn to_ip(&self) -> IpAddr {
        *self
    }
}
impl Ip for Ipv4Addr {
    #[inline(always)]
    fn from_ip(ip: IpAddr) -> Ipv4Addr {
        ip.ext().to_ipv4().expect("unable to convert provided IpAddr to IPv4 Address!")
    }

    #[inline(always)]
    fn to_ip(&self) -> IpAddr {
        IpAddr::V4(*self)
    }
}
impl Ip for Ipv6Addr {
    #[inline(always)]
    fn from_ip(ip: IpAddr) -> Ipv6Addr {
        ip.ext().to_ipv6()
    }

    #[inline(always)]
    fn to_ip(&self) -> IpAddr {
        IpAddr::V6(*self)
    }
}

impl<const N: usize> Ip for [u8; N] {
    #[inline(always)]
    fn from_ip(ip: IpAddr) -> [u8; N] {
        assert!(N == 4 || N == 16);
        let mut out = [0u8; N];
        match ip.ext().to_canonical() {
            IpAddr::V4(ip4) => {
                out.copy_from_slice(&(ip4.octets()));
            },
            IpAddr::V6(ip6) => {
                out.copy_from_slice(&(ip6.octets()));
            },
        }
        out
    }

    #[inline(always)]
    fn to_ip(&self) -> IpAddr {
        match N {
            4 => {
                let mut out = [0u8; 4];
                out.copy_from_slice(self);
                IpAddr::V4(Ipv4Addr::from(out))
            },
            16 => {
                let mut out = [0u8; 16];
                out.copy_from_slice(self);
                IpAddr::V6(Ipv6Addr::from(out))
            },
            _ => {
                panic!("Unknown IP octets length!");
            },
        }
    }
}

impl Ip for (u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8) {
    #[inline(always)]
    fn from_ip(ip: IpAddr) -> (u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8) {
        let octets = ip.ext().to_ipv6().to_octets_ipv6();

        (
            octets[0],
            octets[1],
            octets[2],
            octets[3],
            octets[4],
            octets[5],
            octets[6],
            octets[7],
            octets[8],
            octets[9],
            octets[10],
            octets[11],
            octets[12],
            octets[13],
            octets[14],
            octets[15],
        )
    }

    #[inline(always)]
    fn to_ip(&self) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(self.0, self.1, self.2, self.3))
    }
}

impl Ip for u32 {
    #[inline(always)]
    fn from_ip(ip: IpAddr) -> u32 {
        ip.ext().to_canonical().to_bits_ipv4().expect("IPv6 is cannot encoded to u32.")
    }

    #[inline(always)]
    fn to_ip(&self) -> IpAddr {
        IpAddr::V4(Ipv4Addr::from(self.to_be_bytes()))
    }
}

impl Ip for u128 {
    #[inline(always)]
    fn from_ip(ip: IpAddr) -> u128 {
        ip.to_bits()
    }

    #[inline(always)]
    fn to_ip(&self) -> IpAddr {
        IpAddr::V6(Ipv6Addr::from(self.to_be_bytes()))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct IP(IpAddr);

impl Ip for IP {
    #[inline(always)]
    fn from_ip(ip: IpAddr) -> IP {
        IP(ip)
    }

    #[inline(always)]
    fn to_ip(&self) -> IpAddr {
        self.0
    }
}

pub trait IpAddrExt: Ip {
    /// A method for convert IpAddr to custom struct, this is for avoid warning for use nightly-only unstable method in IpAddr
    #[inline(always)]
    fn ext(&self) -> IP {
        IP::from_ip(self.to_ip())
    }

    /* if */
    #[inline(always)]
    fn is_ipv4(&self) -> bool {
        self.to_ip().is_ipv4()
    }

    #[inline(always)]
    fn is_ipv6(&self) -> bool {
        self.to_ip().is_ipv6()
    }

    #[inline(always)]
    fn is_ipv4_mapped(&self) -> bool {
        if let IpAddr::V6(ip6) = self.to_ip() {
            let x = ip6.segments();
            x[0] == 0 && x[1] == 0 && x[2] == 0 && x[3] == 0 && x[4] == 0 && x[5] == 0xffff
        } else {
            false
        }
    }

    #[inline(always)]
    fn is_ipv4_compatible(&self) -> bool {
        if let IpAddr::V6(ip6) = self.to_ip() {
            let x = ip6.segments();
            x[0] == 0 && x[1] == 0 && x[2] == 0 && x[3] == 0 && x[4] == 0 && x[5] == 0
        } else {
            false
        }
    }

    /* convert */
    #[inline(always)]
    fn to_ipv4(&self) -> Option<Ipv4Addr> {
        match self.to_ip() {
            IpAddr::V4(ip4) => Some(ip4),
            IpAddr::V6(ip6) => {
                if ip6 == Ipv6Addr::LOCALHOST {
                    None
                } else {
                    // allow IPv4-compatible IPv6 address
                    ip6.to_ipv4()
                }
            },
        }
    }

    #[inline(always)]
    fn to_ipv6(&self) -> Ipv6Addr {
        match self.to_ip() {
            IpAddr::V4(ip4) => ip4.to_ipv6_mapped(),
            IpAddr::V6(ip6) => ip6,
        }
    }

    #[inline(always)]
    fn to_canonical(&self) -> IpAddr {
        if let Some(ip4) = self.to_ipv4() {
            IpAddr::V4(ip4)
        } else {
            self.to_ip()
        }
    }

    #[inline(always)]
    fn to_bits(&self) -> u128 {
        self.to_bits_ipv6()
    }

    #[inline(always)]
    fn to_bits_ipv4(&self) -> Option<u32> {
        self.to_octets_ipv4().map(u32::from_be_bytes)
    }

    #[inline(always)]
    fn to_bits_ipv6(&self) -> u128 {
        u128::from_be_bytes(self.to_octets_ipv6())
    }

    #[inline(always)]
    fn from_bits(bits: u128) -> Self {
        Self::from_bits_ipv6(bits)
    }

    #[inline(always)]
    fn from_bits_ipv4(bits: u32) -> Self {
        Self::from_octets_ipv4(bits.to_be_bytes())
    }

    #[inline(always)]
    fn from_bits_ipv6(bits: u128) -> Self {
        Self::from_octets_ipv6(&(bits.to_be_bytes()))
    }

    #[inline(always)]
    fn to_octets(&self) -> [u8; 16] {
        self.to_octets_ipv6()
    }

    #[inline(always)]
    fn to_octets_ipv4(&self) -> Option<[u8; 4]> {
        self.to_ipv4().map(|x| { x.octets() })
    }

    #[inline(always)]
    fn to_octets_ipv6(&self) -> [u8; 16] {
        self.to_ipv6().octets()
    }

    #[inline(always)]
    fn from_octets(octets: &[u8]) -> Option<Self> {
        Some(Self::from_ip(
            match octets.len() {
                4 => IpAddr::V4(Ipv4Addr::new(octets[0], octets[1], octets[2], octets[3])),
                16 => {
                    let mut x = [0u8; 16];
                    x.copy_from_slice(octets);
                    IpAddr::V6(Ipv6Addr::from(x))
                },
                _ => {
                    return None;
                }
            }
        ))
    }

    #[inline(always)]
    fn from_octets_ipv4(octets: [u8; 4]) -> Self {
        Self::from_octets(&octets).unwrap()
    }

    #[inline(always)]
    fn from_octets_ipv6(octets: &[u8; 16]) -> Self {
        Self::from_octets(octets).unwrap()
    }

    /* Ipv4 specified, but may also apply to IPv6 Address */

    #[inline(always)]
    fn is_unspecified(&self) -> bool {
        match self.to_canonical() {
            IpAddr::V4(ip4) => {
                ip4 == Ipv4Addr::UNSPECIFIED
            },
            IpAddr::V6(ip6) => {
                ip6 == Ipv6Addr::UNSPECIFIED
            },
        }
    }

    #[inline(always)]
    fn is_loopback(&self) -> bool {
        match self.to_canonical() {
            IpAddr::V4(ip4) => {
                ip4.octets()[0] == 127
            },
            IpAddr::V6(ip6) => {
                ip6 == Ipv6Addr::LOCALHOST
            },
        }
    }

    #[inline(always)]
    fn is_multicast(&self) -> bool {
        match self.to_canonical() {
            IpAddr::V4(ip4) => {
                let a = ip4.octets()[0];
                a >= 224 && a <= 239
            },
            IpAddr::V6(ip6) => {
                ip6.octets()[0] == 0xff
            },
        }
    }

    #[inline(always)]
    fn is_private(&self) -> bool {
        match self.to_canonical() {
            IpAddr::V4(ip4) => {
                let x = ip4.octets();

                x[0] == 10
                ||
                (x[0] == 172 && x[1] >= 16 && x[1] <= 31)
                ||
                (x[0] == 192 && x[1] == 168)
            },
            IpAddr::V6(ip6) => {
                ! self.is_global()
            },
        }
    }

    #[inline(always)]
    fn is_link_local(&self) -> bool {
        if self.is_unicast_link_local() {
            return true;
        }
        if self.is_unique_local() {
            return true;
        }

        if let IpAddr::V4(ip4) = self.to_canonical() {
            let x = ip4.octets();
            x[0] == 169 && x[1] == 254
        } else {
            false
        }
    }

    #[inline(always)]
    fn is_documentation(&self) -> bool {
        match self.to_canonical() {
            IpAddr::V4(ip4) => {
                let x = ip4.octets();

                (x[0] == 192 && x[1] == 0 && x[2] == 2)
                ||
                (x[0] == 198 && x[1] == 51 && x[2] == 100)
                ||
                (x[0] == 203 && x[1] == 0 && x[2] == 113)
            },
            IpAddr::V6(ip6) => {
                let x = ip6.segments();

                (x[0] == 0x2001 && x[1] == 0x0db8)
                ||
                (x[0] == 0x3fff && x[1] <= 0x0fff)
            },
        }
    }

    #[inline(always)]
    fn is_benchmarking(&self) -> bool {
        match self.to_canonical() {
            IpAddr::V4(ip4) => {
                let x = ip4.octets();
                x[0] == 198 && (x[1] == 18 || x[1] == 19)
            },
            IpAddr::V6(ip6) => {
                let x = ip6.segments();
                x[0] == 0x2001 && (x[1] == 0x0002 || x[1] == 0x0200) && x[2] == 0x0000
            },
        }
    }

    #[inline(always)]
    fn is_broadcast(&self) -> bool {
        match self.to_canonical() {
            IpAddr::V4(ip4) => {
                ip4 == Ipv4Addr::BROADCAST
            },
            IpAddr::V6(ip6) => {
                // ipv6 does not have broadcast address.
                false
            },
        }
    }

    #[inline(always)]
    fn is_shared(&self) -> bool {
        match self.to_canonical() {
            IpAddr::V4(ip4) => {
                let x = ip4.octets();
                x[0] == 100 && x[1] >= 64 && x[1] <= 127
            },
            IpAddr::V6(ip6) => {
                // TODO add subnets used by 6to4, 6in4, teredo, etc.
                false
            },
        }
    }

    #[inline(always)]
    fn is_reserved(&self) -> bool {
        match self.to_canonical() {
            IpAddr::V4(ip4) => {
                ip4.octets()[0] >= 240 && ip4 != Ipv4Addr::BROADCAST
            },
            IpAddr::V6(ip6) => {
                // whether IPv6 have reserved addresses?
                false
            },
        }
    }

    /* IPv6 specified. implementations for IPv4 Address should always return false. */
    #[inline(always)]
    fn is_unicast_link_local(&self) -> bool {
        if let IpAddr::V6(ip6) = self.to_canonical() {
            (ip6.segments()[0] & 0xffc0) == 0xfe80
        } else {
            false
        }
    }

    #[inline(always)]
    fn is_unique_local(&self) -> bool {
        if let IpAddr::V6(ip6) = self.to_canonical() {
            (ip6.segments()[0] & 0xfe00) == 0xfc00
        } else {
            false
        }
    }
    /* end of IPv6 specified */

    /// checks whether the IP address is globally reachable.
    #[inline(always)]
    fn is_global(&self) -> bool {
        let ip = self.to_canonical().ext();
        !(
            ip.is_multicast()
            ||
            ip.is_loopback()
            ||
            ip.is_unspecified()
            ||
            (ip.is_ipv4() && ip.is_private())
            ||
            ip.is_link_local()
            ||
            ip.is_documentation()
            ||
            ip.is_broadcast()
            ||
            ip.is_unicast_link_local()
            ||
            ip.is_unique_local()
            ||
            ip.is_shared()
            ||
            ip.is_reserved()
        )
    }
}

impl<T: Ip> IpAddrExt for T {
}

use super::Spawn;

use std::time::{Instant, Duration};

type Ops = u128;

/// benchmark for IO-intensive (IO-bound) tasks.
pub struct IO {
    tasks: u8,
    spawn: Option<Spawn>,
    test_time: Duration,
}

impl IO {
    #[inline(always)]
    pub const fn default() -> Self {
        Self::builder()
        .tasks(8)
        .test_time_secs(15)
    }

    #[inline(always)]
    pub const fn builder() -> Self {
        Self {
            tasks: 0,
            spawn: None,
            test_time: Duration::new(0, 0),
        }
    }

    #[inline(always)]
    pub const fn tasks(mut self, count: u8) -> Self {
        self.tasks = count;
        self
    }

    #[inline(always)]
    pub const fn spawn(mut self, spawn: Spawn) -> Self {
        let _ = core::mem::ManuallyDrop::new(self.spawn.replace(spawn));
        self
    }

    #[inline(always)]
    pub const fn test_time(mut self, time: Duration) -> Self {
        self.test_time = time;
        self
    }
    #[inline(always)]
    pub const fn test_time_secs(self, secs: u64) -> Self {
        self.test_time(Duration::from_secs(secs))
    }

    /// prevent unused_mut lint
    #[inline(always)]
    fn used_mut(&mut self) {
        self.tasks += 0;

        if self.spawn.is_none() || self.tasks == 0 || self.test_time == Duration::new(0, 0) {
            panic!("not configured!");
        }
    }

    #[inline(always)]
    pub fn network_web_scanner(&mut self) -> (Ops, f64) {
        self.used_mut();

        use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

        #[inline(always)]
        fn rand_ipv4() -> Ipv4Addr {
            let mut ip4;
            loop {
                ip4 = Ipv4Addr::from(fastrand::u32(..));
                if ip4.ext().is_global() {
                    return ip4;
                }
            }
        }

        #[inline(always)]
        fn rand_ipv6() -> Ipv6Addr {
            let mut ip6;
            loop {
                ip6 = Ipv6Addr::from(fastrand::u128(..));
                if ip6.ext().is_global() {
                    return ip6;
                }
            }
        }

        #[inline(always)]
        fn rand_ip() -> IpAddr {
            if fastrand::bool() {
                IpAddr::V4(rand_ipv4())
            } else {
                IpAddr::V6(rand_ipv6())
            }
        }

        let spawn = self.spawn.as_mut().unwrap();

        todo!();

        //smol::net::TcpSocket::connect
    }

    #[inline(always)]
    pub fn moka_cache_concurrent_rw_access(&mut self) -> (Ops, f64) {
        self.used_mut();

        let spawn = self.spawn.as_mut().unwrap();

        let cache: moka::future::Cache<u16, usize> =
            //Arc::new(
                moka::future::Cache::builder()
                .max_capacity(65536)
                .time_to_idle(self.test_time / 100)
                .time_to_live(self.test_time / 10)
                .build();
            //);

        let (ops_tx, ops_rx) = crossbeam_channel::bounded(self.tasks as usize);

        for i in 0 .. self.tasks {
            let ops_tx = ops_tx.clone();
            let cache = cache.clone();
            let time = self.test_time;

            (spawn)(Box::pin(async move {
                let mut t;
                let mut t2;
                let mut elapsed = Duration::new(0, 0);
                let mut key = i as u16;
                let mut next_key;
                let mut ops = 0u64;
                while elapsed < time {
                    t = Instant::now();
                    {
                        cache.run_pending_tasks().await;
                        next_key =
                            match cache.get(&key).await {
                                Some(v) => v,
                                _ => {
                                    cache.insert(key, key as usize).await;
                                    key as usize
                                }
                            };
                        if next_key == (key as usize) {
                            next_key += 1;
                        }
                        next_key = next_key.wrapping_mul(2);
                        cache.insert(key, next_key).await;
                        next_key = next_key.wrapping_add(cache.iter().count());
                        key = (next_key % 65536) as u16;
                    }
                    t2 = t.elapsed();
                    if (ops % 1000) == 0 {
                        //dbg!((t2, elapsed));
                    }

                    elapsed += t2;
                    ops += 1;
                }
                ops_tx.send(ops).unwrap();
            }));
        }
        drop(ops_tx);

        let mut total_ops = 0u128;
        while let Ok(ops) = ops_rx.recv() {
            total_ops += ops as u128;
        }
        let ops_per_second = (total_ops as f64) / self.test_time.as_secs_f64();
        (total_ops, ops_per_second)
    }
}

#[inline(always)]
fn default() -> IO {
    IO::default()
        .tasks(crate::cpu_count() as u8)
}

static MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn test_io_bound_asyncute() {
    let _mutex_guard = MUTEX.lock().unwrap_or_else(|e| { e.into_inner() });

    dbg!(
        default()
            .spawn(Box::new(|f| { crate::spawn(f).detach(); }))
            .moka_cache_concurrent_rw_access()
    );
}

#[test]
fn test_io_bound_smolscale() {
    let _mutex_guard = MUTEX.lock().unwrap_or_else(|e| { e.into_inner() });

    dbg!(
        default()
            .spawn(Box::new(|f| { smolscale::spawn(f).detach(); }))
            .moka_cache_concurrent_rw_access()
    );
}

#[test]
fn test_io_bound_smolscale2() {
    let _mutex_guard = MUTEX.lock().unwrap_or_else(|e| { e.into_inner() });

    dbg!(
        default()
            .spawn(Box::new(|f| { smolscale2::spawn(f).detach(); }))
            .moka_cache_concurrent_rw_access()
    );
}

#[test]
fn test_io_bound_async_global_executor() {
    let _mutex_guard = MUTEX.lock().unwrap_or_else(|e| { e.into_inner() });

    dbg!(
        default()
            .spawn(Box::new(|f| { async_global_executor::spawn(f).detach(); }))
            .moka_cache_concurrent_rw_access()
    );
}

#[test]
fn test_io_bound_tokio() {
    let _mutex_guard = MUTEX.lock().unwrap_or_else(|e| { e.into_inner() });

    let tokio_runtime = tokio::runtime::Runtime::new().unwrap();
    let _enter_guard = tokio_runtime.enter();

    dbg!(
        default()
            .spawn(Box::new(|f| { tokio::spawn(f); }))
            .moka_cache_concurrent_rw_access()
    );
}
