use crate::*;

use std::time::Instant;

#[test]
fn atomic_instant() {
    let a = dbg!(AtomicInstant::now());
    dbg!(a.add(Duration::from_secs(1)));
    dbg!(a.get());
    dbg!(a.sub(Duration::from_secs(2)));
    dbg!(a.get());
    dbg!(a.set(dbg!(Instant::now())));
    dbg!(a.get());
}

#[test]
fn hack_zero_instant() {
    dbg!(hack::instant_zero());
}

#[test]
fn idrs_gen() {
    let a = id::ID::new(0x90_ab_cd_ef_01_02_03_04).unwrap();
    let b = id::ID::auto_ns();
    for _ in 0..10 {
        eprintln!("{:x} (ID::new)", a.generate().unwrap());
        eprintln!("{:x} (ID::auto_ns)", b.generate().unwrap());
        eprintln!("{:x} (EXECUTOR_ID)", id::gen_executor_id().unwrap());
    }
}

#[inline(always)]
fn startup() {
    use once_cell::sync::Lazy;
    static STARTUP: Lazy<()> = Lazy::new(|| {
        env_logger::init();
        log::info!("env-logger init");
    });

    let _ = &*STARTUP;
}

#[test]
#[should_panic]
fn test_panic_messages() {
    startup();
    let task = spawn(async {
        panic!("this panic is for testing");
    });
    smol::block_on(task);
    println!("normal");
}

#[test]
fn test2() {
    startup();
    let sc = StableClock::global();

    for _ in 0..10 {
        dbg!(sc.now(), Instant::now(), sc.time_diff(), sc.is_time_jumped());
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn test1() {
    startup();
    let task = spawn(async {
        let mut d = Duration::from_secs(2);
        for i in 0..8 {
            let mut started = std::time::Instant::now();
            d /= 2;
            spawn(async move {
                let until = started + Duration::from_secs(3).min(d * 10);
                while started < until {
                    println!("({i}) | sleep={d:?} | elapsed={:?} | thread={:?}", started.elapsed(), std::thread::current());
                    started = std::time::Instant::now();
                    smol::Timer::after(d).await;
                }
            }).detach();
        }

        {
            smol::Timer::after(Duration::from_secs(5)).await;
        }
    });
    smol::block_on(task);
}

#[test]
// test for CPU usage if idle mostly.
fn test_idle() {
    let task = spawn(async {
        for i in 0..20u8 {
            smol::Timer::after(Duration::from_secs(1)).await;
        }
    });
    smol::block_on(task);
}

#[test]
// test the execute delay (cost of time from schedule and execute, aka channel send and recv)
fn test_execute_cost() {
    let mut costs = Vec::with_capacity(1001);
    let mut t;
    let mut elapsed;
    for _ in 0..1000u16 {
        t = Instant::now();
        elapsed = smol::block_on(spawn(async move { t.elapsed() }));
        dbg!(elapsed);
        costs.push(elapsed);
    }

    let t = Instant::now();
    let n = 1000u16;
    for _ in 0..n {
        smol::block_on(spawn(async {}));
    }
    let t = t.elapsed();
    costs.push(dbg!(Duration::from_secs_f64(t.as_secs_f64() / (n as f64))));

    dbg!(Duration::from_secs_f64(costs.iter().map(Duration::as_secs_f64).sum::<f64>() / (costs.len() as f64)));
}

#[test]
fn test_linked_storage() {
    let s = Storage::<u128, 4>::new();
    assert!(dbg!(s.push(1)).is_ok());
    assert!(dbg!(s.push(2)).is_ok());
    assert!(dbg!(s.push(3)).is_ok());
    assert!(dbg!(s.push(4)).is_ok());
    assert!(dbg!(s.push(5)).is_err());

    let ls = LinkedStorage::from(s);
    for i in 5..=103 {
        ls.push(i);
    }

    let g = scc2::ebr::Guard::new();
    let mut v = Vec::new();
    for _ in 1..=103u8 {
        assert!(dbg!(ls.pop().map(|x| { let x = x.get_guarded_ref(&g); v.push(*x); x })).is_some());
    }
    assert!(dbg!(ls.pop()).is_none());

    assert_eq!(v.len(), 103);
    for i in 1..=103 {
        assert!(v.contains(&i));
    }
}
