pub enum Pool {
    Thread(ThreadPool),
    Process(Process),
}

pub struct ThreadPool {
}

pub struct ProcessPool {
}
