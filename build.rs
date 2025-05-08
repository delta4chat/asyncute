#[cfg(all(feature="flume", feature="crossbeam-channel"))]
compile_error!("features 'flume' and 'crossbeam-channel' are mutually exclusive and cannot both be enabled!");

fn main() {
}

