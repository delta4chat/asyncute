use crate::*;

use std::{
    sync::Arc,
    time::Instant,
};

use portable_atomic::{
    AtomicU32,
    AtomicBool,
    Ordering::Relaxed,
};

use once_cell::sync::OnceCell;

#[derive(Debug)]
pub(crate) struct ExecutorState {
    /// per-process unique ID for each executor.
    /// this ID will not reuse even a executor exits.
    pub id: ExecutorId,

    /// checks whether a executor working (for run the Runnable).
    pub working: AtomicBool,

    /// whether this executor can be exit if idle.
    pub exitable: bool,

    /// start timestamp of this executor.
    pub started: Instant,

    /// how many time used for working?
    pub working_time: AtomicDuration,

    /// the join_handle initial by external caller that creates this executor.
    pub join_handle: OnceCell<Arc<std::thread::JoinHandle<()>>>,

    /// requests a exitable executor thread to exit in next checkpoint.
    /// this is no effect for persist executors.
    pub please_exit: AtomicBool,
}

impl ExecutorState {
    #[inline(always)]
    pub fn new(id: ExecutorId) -> Self {
        Self {
            id,
            working: AtomicBool::new(false),
            started: Instant::now(),
            working_time: AtomicDuration::zero(),
            exitable: false,
            join_handle: OnceCell::new(),
            please_exit: AtomicBool::new(false),
        }
    }

    #[inline(always)]
    pub fn is_working(&self) -> bool {
        self.working.load(Relaxed)
    }

    /// returns range within 0.0 ~ 1.0 (f64).
    /// * 1.0 means all of time is working.
    /// * 0.0 means all of time is idle.
    #[inline(always)]
    pub fn working_ratio(&self) -> f64 {
        let running_time = self.started.elapsed();
        let working_time = self.working_time.get();
        working_time.as_secs_f64() / running_time.as_secs_f64()
    }

    #[inline(always)]
    pub fn exit(&self) -> Result<(), &'static str> {
        if self.exitable {
            self.please_exit.store(true, Relaxed);
            Ok(())
        } else {
            Err("executor is not exitable!")
        }
    }
}

#[derive(Debug)]
pub struct Executor {
    /// variable for avoid double drop.
    dropped: bool,

    /// store common state between executor and external caller.
    state: Arc<ExecutorState>,

    /// channel receiver for receive runnable.
    runinfo_rx: Receiver<RunInfo>,

    _not_send_sync_unpin: PhantomNonSendSyncUnpin,
}

impl Executor {
    #[inline(always)]
    pub(crate) const fn new(
        state: Arc<ExecutorState>,
        runinfo_rx: Receiver<RunInfo>,
    ) -> Self {
        Self {
            dropped: false,
            runinfo_rx,
            state,
            _not_send_sync_unpin: PhantomData,
        }
    }

    /// run a runnable, catch unwind if it panic (if possible)
    #[inline(always)]
    fn run_one(&self, runinfo: RunInfo, bulk: bool) {
        let id = self.state.id;

        log::debug!("Runnable ScheduleInfo: {:?}", runinfo.info);
        
        let taskinfo = runinfo.runnable.metadata().clone();

        if ! bulk {
            self.state.working.store(true, Relaxed);
        }
        let t = Instant::now();
        match std::panic::catch_unwind(move || { runinfo.runnable.run() }) {
            Ok(_) => {},
            Err(boxed_any) => {
                let ref_any = &*boxed_any;
                // core::any::type_name_of_val() is useless due to it does not resolve trait objects to real type name...
                let type_id = ref_any.type_id();

                macro_rules! log_any {
                    ($($t:ty,)*) => {
                        if false {
                        }
                        $(
                            else if let Some(val) = ref_any.downcast_ref::<$t>() {
                                log::error!(concat!("Executor {} Runnable panic! TypeId={:?} ", stringify!($t), " = {:?}"), id, type_id, val);
                            }
                        )*
                        else {
                            log::error!("Executor {id} Runnable panic! TypeId={type_id:?} dyn Any = {ref_any:?}");
                        }
                    }
                }

                log_any!(
                    String, &'_ String,
                    &'_ str,
                    &'_ dyn std::error::Error,
                    &'_ dyn std::fmt::Debug,
                    std::io::Error,
                    std::io::ErrorKind,
                    std::panic::PanicInfo,
                    std::panic::PanicHookInfo,
                    core::panic::PanicInfo,
                );
            }
        }
        let t = t.elapsed();
        if ! bulk {
            self.state.working.store(false, Relaxed);
        }

        taskinfo.0.run_count.checked_add(1);
        taskinfo.0.run_took.add(t);
        self.state.working_time.add(t);

        if ProfileConfig::global().is_enabled() {
            let p = Profile::global();
            p.started();
            p.runnable.run_count.checked_add(1);
        }
    }

    #[inline(always)]
    pub fn run(self) -> std::io::Result<()> {
        let rx = &self.runinfo_rx;

        let _defer = Defer::new(|| {
            self.state.working.store(false, Relaxed);
        });

        let exitable = self.state.exitable;
        let interval = MonitorConfig::global().interval();
        let max_idle = interval * 2;

        let mut worked;
        let mut t = Instant::now();

        #[cfg(not(feature="crossbeam-deque"))]
        let error_closed = Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "runinfo channel closed!"));

        loop {
            worked = false;
            loop {
                #[cfg(feature="crossbeam-deque")]
                match rx.try_recv() {
                    Some(runinfo) => {
                        if ! worked {
                            worked = true;
                            self.state.working.store(true, Relaxed);
                        }

                        self.run_one(runinfo, true);
                    }

                    _ => {
                        // crossbeam-deque is always open and no "closed" state.
                        break;
                    }
                }

                #[cfg(not(feature="crossbeam-deque"))]
                match rx.try_recv() {
                    Ok(runinfo) => {
                        #[cfg(feature="kanal")]
                        let runinfo =
                            match runinfo {
                                Some(v) => v,
                                _ => {
                                    break;
                                }
                            };

                        if ! worked {
                            worked = true;
                            self.state.working.store(true, Relaxed);
                        }

                        self.run_one(runinfo, true);
                    },
                    Err(err) => {
                        #[cfg(feature="flume")]
                        match err {
                            flume::TryRecvError::Empty => {
                                break;
                            },
                            _ => {
                                return error_closed;
                            }
                        }

                        #[cfg(feature="crossbeam-channel")]
                        if err.is_empty() {
                            break;
                        } else {
                            return error_closed;
                        }

                        #[cfg(feature="kanal")]
                        return error_closed;

                        // std::sync::mpmc is ported from crossbeam-channel's implementation.
                        // but it uses std::sync::mpsc's Error type, and without such as "is_empty" methods.
                        #[cfg(feature="std-mpmc")]
                        match err {
                            std::sync::mpmc::TryRecvError::Empty => {
                                break;
                            },
                            _ => {
                                return error_closed;
                            }
                        }
                    }
                }
            }

            if worked {
                #[cfg(test)]
                log::trace!("{} worked", self.state.id);

                self.state.working.store(false, Relaxed);
                if exitable {
                    t = Instant::now();
                }
            } else {
                #[cfg(test)]
                log::trace!("{} not worked", self.state.id);
            }

            #[cfg(feature="crossbeam-deque")]
            match rx.recv_timeout(interval) {
                Some(runinfo) => {
                    if exitable {
                        t = Instant::now();
                    }
                    self.run_one(runinfo, false);
                },
                _ => {
                    // crossbeam-deque is always open and no "closed" state.
                }
            }

            #[cfg(not(feature="crossbeam-deque"))]
            match rx.recv_timeout(interval) {
                Ok(runinfo) => {
                    if exitable {
                        t = Instant::now();
                    }
                    self.run_one(runinfo, false);
                },
                Err(err) => {
                    #[cfg(feature="flume")]
                    match err {
                        flume::RecvTimeoutError::Timeout => {},
                        _ => {
                            return error_closed;
                        }
                    }

                    #[cfg(feature="crossbeam-channel")]
                    if ! err.is_timeout() {
                        return error_closed;
                    }

                    #[cfg(feature="kanal")]
                    match err {
                        kanal::ReceiveErrorTimeout::Timeout => {},
                        _ => {
                            return error_closed;
                        }
                    }

                    // std::sync::mpmc is ported from crossbeam-channel's implementation.
                    // but it uses std::sync::mpsc's Error type, and without such as "is_timeout" methods.
                    #[cfg(feature="std-mpmc")]
                    match err {
                        std::sync::mpmc::RecvTimeoutError::Timeout => {},
                        _ => {
                            return error_closed;
                        }
                    }
                }
            }

            if exitable {
                if t.elapsed() > max_idle {
                    break;
                }
                if self.state.working_ratio() < 0.1 {
                    break;
                }
                if self.state.please_exit.load(Relaxed) {
                    break;
                }
            }
        }
        Ok(())
    }
}

impl Drop for Executor {
    #[inline(always)]
    fn drop(&mut self) {
        if self.dropped {
            return;
        }
        self.dropped = true;

        // remove this executor from global index.
        EXECUTOR_INDEX.remove(&self.state.id);

        // set working state to idle.
        self.state.working.store(false, Relaxed);

    }
}

