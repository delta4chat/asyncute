macro_rules! check_exclusive_features {
    ($($feature:literal,)*) => {
        const _: () = {
            let mut count = 0;
            $(
                if cfg!(feature=$feature) {
                    count += 1;
                }
            )*
            if count == 0 {
                panic!(concat!("none of features ", concat!($("'", $feature, "' ", )*), "enabled! but exactly one required!"));
            }
            if count > 1 {
                panic!(concat!("features ", concat!($("'", $feature, "' ", )*), "are mutually exclusive and cannot both be enabled!"));
            }
        };
    }
}

check_exclusive_features!(
    "flume",
    "crossbeam-channel",
    "kanal",

    "std-mpmc",
);

fn main() {
    #[cfg(feature="kanal")]
    println!("cargo:warning=kanal v0.1.1 channels is buggy (recv_timeout busy waiting) currently! this only for test and should not use in release until it get fixed in published kanal crate");
}

