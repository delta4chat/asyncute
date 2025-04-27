static ID_COUNTER: AtomicU64 = AtomicU64::new(1);

#[inline(always)]
pub(crate) fn gen_executor_id() -> Option<u64> {
    let mut id;
    loop {
        id = ID_COUNTER.load(Relaxed);

        if id == u32::MAX {
            return None;
        }

        if
            ID_COUNTER.compare_exchange(
                id, id.checked_add(1)?,
                Relaxed, Relaxed,
            ).is_ok()
        {
            return Some(id);
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct IDServer {
    udp: std::net::UdpSocket,
}

