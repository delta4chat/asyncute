use crate::*;

use std::time::Instant;

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

