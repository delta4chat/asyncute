mod cpu;
//mod io;

mod test;

use core::{
    future::Future,
    pin::Pin,
};

pub type Spawn = Box<dyn FnMut(Pin<Box<dyn Future<Output=()> + Send + Sync + 'static>>)>;

