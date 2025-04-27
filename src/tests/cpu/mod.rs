fn find_primes_in_worse_method(cb: impl FnMut(u128) -> bool) {
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

fn find_machine_e(cb: impl FnMut(f64) -> bool) {
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
