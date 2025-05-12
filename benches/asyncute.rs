use criterion::{black_box, criterion_group, criterion_main, Criterion};

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

fn asyncute_latency_benchmark(c: &mut Criterion) {
    c.bench_function("asyncute_spawn_cost", |b| {
        b.iter(|| {
            let _ = black_box(smol::block_on(black_box(asyncute::spawn(black_box(async {
                black_box(1);
            })))));
        })
    });

    c.bench_function("asyncute_cpu_bound", |b| {
        b.iter(|| {
            let _ = black_box(smol::block_on(black_box(asyncute::spawn(black_box(async {
                let mut c: u32 = 0;
                const M: u32 = 25000000;
                black_box(find_machine_e(black_box(|_| {
                    c += 1;
                    c > M
                })))
            })))));
        })
    });


    c.bench_function("asyncute_executor_status_new", |b| {
        b.iter(|| {
            let _ = black_box(asyncute::ExecutorStatus::new());
        })
    });

    c.bench_function("asyncute_executor_status_current", |b| {
        b.iter(|| {
            let _ = black_box(asyncute::ExecutorStatus::current());
        })
    });

    let mut status = asyncute::ExecutorStatus::new();
    c.bench_function("asyncute_executor_status_update", |b| {
        b.iter(|| {
            let _ = black_box(status.update());
        })
    });
}

criterion_group!(benches, asyncute_latency_benchmark);
criterion_main!(benches);
