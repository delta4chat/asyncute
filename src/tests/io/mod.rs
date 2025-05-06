fn find_primes_in_worse_method(mut cb: impl FnMut(u128) -> bool) {
    let mut n: u128 = 1;
    let mut is_prime = true;
    loop {
        if is_prime {
            if cb(n) {
                return;
            }
        }
        n += 1;

        is_prime = true;
        for d in 2..n {
            if (n % d) != 0 {
                is_prime = false;
                break;
            }
        }
    }
}

fn find_machine_e(mut cb: impl FnMut(f64) -> bool) {
    let mut init = 20914834.3492847762392393;
    let mut n;
    let mut n2;
    loop {
        n = init;
        n2 = n / 2.0;
        while n != n2 {
            n = n2 / 2.0;
            n2 = n / 2.0;
        }
        if cb(n) {
            return;
        }
        init = init * (2.0 + n);
        init *= 1.5;
    }
}

use std::time::Duration;
use crate::tests::cpu::IpAddrExt;
use crate::tests::Spawn;

type Ops = u128;

/// benchmark for CPU-intensive (CPU-bound) tasks.
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
}
