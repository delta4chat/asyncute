use std::sync::Arc;

mod cpu;
mod io;

mod test;

use core::{
    future::Future,
    pin::Pin,
};

pub type Spawn = Arc<dyn Fn(Pin<Box<dyn Future<Output=()> + Send + 'static>>) + Send + Sync>;

