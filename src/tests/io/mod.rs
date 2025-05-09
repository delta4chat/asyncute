use std::sync::Arc;

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
    pub fn xor_echo_service(&mut self) -> (Ops, f64) {
        use smol::io::{AsyncReadExt, AsyncWriteExt};
        use portable_atomic::{AtomicBool, Ordering::Relaxed};
        use smoltimeout::TimeoutExt;

        self.used_mut();

        let spawn = self.spawn.as_ref().unwrap();

        let time = self.test_time;

        #[cfg(feature="flume")]
        let (ops_tx, ops_rx) = flume::bounded(self.tasks as usize);

        #[cfg(feature="crossbeam-channel")]
        let (ops_tx, ops_rx) = crossbeam_channel::bounded(self.tasks as usize);

        #[cfg(feature="kanal")]
        let (ops_tx, ops_rx) = kanal::bounded(self.tasks as usize);

        // create TCP server
        let tcp = smol::block_on(smol::net::TcpListener::bind("127.0.0.1:0")).unwrap();
        let addr = dbg!(tcp.local_addr().unwrap());

        let server_task = (spawn)({
            let ops_tx = ops_tx.clone();
            let spawn = spawn.clone();

            let accept = Arc::new(AtomicBool::new(true));

            Box::pin(async move {
            let handler = |mut conn: smol::net::TcpStream| {
                let accept = accept.clone();
                let ops_tx = ops_tx.clone();

                async move {
                let mut buf = [0u8; 1024];
                let mut len;
                let mut xor: u8 = 0x1f;
                let mut ops = 0;
                let mut elapsed = Duration::new(0, 0);
                let mut elapsed_secs;
                let mut t;
                let mut ts = Vec::with_capacity(10000);
                while elapsed < time {
                    t = std::time::Instant::now();
                    len =
                        match conn.read(&mut buf).await {
                            Ok(v) => v,
                            _ => {
                                break;
                            }
                        };
                    elapsed += t.elapsed();

                    elapsed_secs = elapsed.as_secs();
                    if ! ts.contains(&elapsed_secs) {
                        ts.push(elapsed_secs);
                        println!("{elapsed:?}");
                    }
                    //println!("{len}");

                    for i in 0..len {
                        buf[i] ^= xor;
                        xor = xor.wrapping_add(1);
                        xor ^= (i % 256) as u8;
                    }

                    t = std::time::Instant::now();
                    match conn.write_all(&buf[..len]).await {
                        Ok(v) => v,
                        _ => {
                            break;
                        }
                    }
                    elapsed += t.elapsed();

                    ops += 1;
                }
                ops_tx.send((ops, elapsed)).unwrap();
                accept.store(false, Relaxed);
            } };

            while accept.load(Relaxed) {
                if let Some(Ok((conn, peer))) = tcp.accept().timeout(Duration::from_secs(1)).await {
                    if accept.load(Relaxed) {
                        (spawn)(Box::pin(handler(conn)));
                    }
                }
            }
        }) });
        drop(ops_tx);

        let client_task = (spawn)({ let spawn = spawn.clone(); Box::pin(async move {
            for i in 0 .. crate::cpu_count() {
                let spawn = spawn.clone();
                (spawn)(Box::pin(async move {
                    let i = i as u8;
                    let mut client = smol::net::TcpStream::connect(addr).await.unwrap();
                    let mut data = [i; 4389];
                    loop {
                        match client.write_all(&data).await {
                            Ok(_) => {},
                            _ => {
                                break;
                            }
                        }
                        match client.read_exact(&mut data).await {
                            Ok(_) => {},
                            _ => {
                                break;
                            }
                        }
                    }
                }));
            }
        }) });

        let mut total_ops = 0u128;
        let mut total_time = Duration::new(0, 0);
        while let Ok((ops, time)) = ops_rx.recv() {
            total_ops += ops as u128;
            total_time += time;
        }
        let ops_per_second = (total_ops as f64) / total_time.as_secs_f64();
        (total_ops, ops_per_second)
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
            .spawn(Arc::new(|f| { crate::spawn(f).detach(); }))
            .xor_echo_service()
    );
}

#[test]
fn test_io_bound_smolscale() {
    let _mutex_guard = MUTEX.lock().unwrap_or_else(|e| { e.into_inner() });

    dbg!(
        default()
            .spawn(Arc::new(|f| { smolscale::spawn(f).detach(); }))
            .xor_echo_service()
    );
}

#[test]
fn test_io_bound_smolscale2() {
    let _mutex_guard = MUTEX.lock().unwrap_or_else(|e| { e.into_inner() });

    dbg!(
        default()
            .spawn(Arc::new(|f| { smolscale2::spawn(f).detach(); }))
            .xor_echo_service()
    );
}

#[test]
fn test_io_bound_async_global_executor() {
    let _mutex_guard = MUTEX.lock().unwrap_or_else(|e| { e.into_inner() });

    dbg!(
        default()
            .spawn(Arc::new(|f| { async_global_executor::spawn(f).detach(); }))
            .xor_echo_service()
    );
}

#[test]
fn test_io_bound_tokio() {
    let _mutex_guard = MUTEX.lock().unwrap_or_else(|e| { e.into_inner() });

    let tokio_runtime = tokio::runtime::Runtime::new().unwrap();
    let _enter_guard = tokio_runtime.enter();

    dbg!(
        default()
            .spawn(Arc::new(|f| { tokio::spawn(f); }))
            .xor_echo_service()
    );
}
