#![doc = include_str!("README.md")]

#![forbid(unsafe_code)]

#![cfg_attr(any(nightly, feature="nightly"), feature(doc_auto_cfg))]

#![cfg_attr(not(test), warn(missing_docs))]

#![cfg_attr(feature="std-mpmc", feature(mpmc_channel))]

pub mod executor;
pub use executor::*;

pub mod defer;
pub use defer::*;

pub mod util;
pub use util::*;

pub mod id;
pub use id::*;

#[cfg(test)]
mod tests;

use core::{
    future::Future,
    ops::{Deref, Range},
    pin::Pin,
    task::{Poll, Context},
};

use std::{
    sync::Arc,
    time::{Instant, Duration},
};

#[cfg(feature="flume")]
use flume::{Sender, Receiver};

#[cfg(feature="crossbeam-channel")]
use crossbeam_channel::{Sender, Receiver};

#[cfg(feature="kanal")]
use kanal::{Sender, Receiver};

#[cfg(feature="crossbeam-deque")]
use crate::util::injector::{InjectorChannel, Sender, Receiver};

#[cfg(feature="std-mpmc")]
use std::sync::mpmc::{Sender, Receiver};

use async_task::ScheduleInfo;

use portable_atomic::{
    AtomicBool,
    AtomicU8,
    AtomicU64,
    AtomicUsize,
    AtomicF64,
    Ordering::Relaxed
};
use once_cell::sync::Lazy;

/// the inner of TaskInfo.
#[derive(Debug)]
pub struct TaskInfoInner {
    dropped: bool,

    id: TaskId,
    nice: i8, // TODO unimplemented
    run_count: AtomicU64,
    run_took: AtomicDuration,
}
impl TaskInfoInner {
    #[inline(always)]
    fn new(nice: i8) -> Self {
        let mut this =
            Self {
                dropped: true,

                id: gen_task_id().expect("Task ID exhausted!"),
                nice,
                run_count: AtomicU64::new(0),
                run_took: AtomicDuration::zero(),
            };

        if ProfileConfig::global().is_enabled() {
            if RunnableProfile::global().alive_count.checked_add(1).is_some() {
                this.dropped = false;
            }
        }

        this
    }
}

impl Drop for TaskInfoInner {
    #[inline(always)]
    fn drop(&mut self) {
        if self.dropped {
            return;
        }
        self.dropped = true;

        RunnableProfile::global().alive_count.checked_sub(1);
    }
}

/// the information associated to each Task and Runnable.
#[derive(Debug, Clone)]
pub struct TaskInfo(Arc<TaskInfoInner>);

impl TaskInfo {
    /// create new TaskInfo
    #[inline(always)]
    fn new(nice: i8) -> Self {
        Self (
            Arc::new(TaskInfoInner::new(nice))
        )
    }
}

impl Deref for TaskInfo {
    type Target = TaskInfoInner;

    #[inline(always)]
    fn deref(&self) -> &TaskInfoInner {
        self.0.as_ref()
    }
}

/// the `async_task::Task` type with `TaskInfo`
pub type Task<T> = async_task::Task<T, TaskInfo>;

/// the `async_task::Runnable` type with `TaskInfo`
pub type Runnable = async_task::Runnable<TaskInfo>;

/// alias type for Executor ID (currently u128).
pub type ExecutorId = u128;

/// alias type for Task ID (currently u128).
pub type TaskId = u128;

/// the global index of executor
static EXECUTOR_INDEX: Lazy<scc2::HashIndex<ExecutorId, Arc<ExecutorState>, ahash::RandomState>> = Lazy::new(Default::default);

/// the JoinHandle of monitor thread
static MONITOR_THREAD_JH: scc2::Atom<std::thread::JoinHandle<()>> = scc2::Atom::init();

/// the Runnable + ScheduleInfo from async-task
pub(crate) struct RunInfo {
    /// Runnable
    runnable: Runnable,

    /// ScheduleInfo
    info: ScheduleInfo,
}

/// the maximum capacity of RunInfo global channel.
pub const RUNINFO_CHANNEL_CAPACITY: usize = 1048576;

/// the global queue of RunInfo.
static RUNINFO_CHANNEL: Lazy<(Sender<RunInfo>, Receiver<RunInfo>)> =
    Lazy::new(|| {
        #[cfg(feature="flume")]
        return flume::bounded(RUNINFO_CHANNEL_CAPACITY);

        #[cfg(feature="crossbeam-channel")]
        return crossbeam_channel::bounded(RUNINFO_CHANNEL_CAPACITY);

        #[cfg(feature="kanal")]
        return kanal::bounded(RUNINFO_CHANNEL_CAPACITY);

        #[cfg(feature="crossbeam-deque")]
        return InjectorChannel::bounded_split(RUNINFO_CHANNEL_CAPACITY);

        #[cfg(feature="std-mpmc")]
        return std::sync::mpmc::sync_channel(RUNINFO_CHANNEL_CAPACITY);
    });

/// get the global Runnable queue (sender & receiver)
#[inline(always)]
fn get_runinfo_channel() -> &'static (Sender<RunInfo>, Receiver<RunInfo>) {
    &*RUNINFO_CHANNEL
}

/// get the send side of global Runnable queue
#[inline(always)]
fn get_runinfo_tx() -> &'static Sender<RunInfo> {
    &(get_runinfo_channel().0)
}

/// get the receive side of global Runnable queue
#[inline(always)]
fn get_runinfo_rx() -> &'static Receiver<RunInfo> {
    &(get_runinfo_channel().1)
}

/// cached, fallbacking, and usize version of `std::thread::available_parallelism`.
#[inline(always)]
pub fn cpu_count() -> usize {
    static CPUS: AtomicUsize = AtomicUsize::new(0);

    let mut count = CPUS.load(Relaxed);
    if count > 0 {
        return count;
    }
    count = std::thread::available_parallelism().map(|nz| { nz.get() }).unwrap_or(1);
    if count == 0 {
        count = 1;
    }
    CPUS.store(count, Relaxed);
    count
}

/// the Profile of Runnable.
pub struct RunnableProfile {
    last_update: AtomicInstant,

    /*
    /// used time for each Runnable.run()
    run_took: scc2::Queue<Duration>,
    */

    alive_count: AtomicU64,
    run_count: AtomicU64,
    run_frequency: AtomicF64,
    queue_count: AtomicU64,
    queue_frequency: AtomicF64,
}
impl RunnableProfile {
    /// the minimum update interval of RunnableProfile.
    pub const UPDATE_INTERVAL: Duration = Duration::new(5, 0);

    /// get the global instance of RunnableProfile.
    #[inline(always)]
    pub const fn global() -> &'static Self {
        static GLOBAL: RunnableProfile =
            RunnableProfile {
                last_update: AtomicInstant::init(),

                alive_count: AtomicU64::new(0),

                run_count: AtomicU64::new(0),
                run_frequency: AtomicF64::new(0.0),

                queue_count: AtomicU64::new(0),
                queue_frequency: AtomicF64::new(0.0),
            };

        &GLOBAL
    }

    /// update frequency.
    #[inline(always)]
    pub fn update(&self) -> bool {
        let elapsed_secs =
            if let Some(started) = Profile::global().started() {
                started.elapsed().as_secs_f64()
            } else {
                return false;
            };

        if self.last_update.get().elapsed() < Self::UPDATE_INTERVAL {
            return false;
        }
        self.last_update.set(Instant::now());

        if elapsed_secs == 0.0 {
            self.run_frequency.store(0.0, Relaxed);
            self.queue_frequency.store(0.0, Relaxed);
        } else {
            let runs = self.run_count() as f64;
            self.run_frequency.store(runs / elapsed_secs, Relaxed);

            let queues = self.queue_count() as f64;
            self.queue_frequency.store(queues / elapsed_secs, Relaxed);
        }

        true
    }

    /// all "alive" Runnable that is not terminated.
    #[inline(always)]
    pub fn alive_count(&self) -> u64 {
        self.alive_count.load(Relaxed)
    }

    /// all handled Runnable (by running it)
    #[inline(always)]
    pub fn run_count(&self) -> u64 {
        self.run_count.load(Relaxed)
    }

    /// how many Runnable ran within one second (in average)
    #[inline(always)]
    pub fn run_frequency(&self) -> f64 {
        self.run_frequency.load(Relaxed)
    }

    /// all scheduled Runnable (by queues to crossbeam-channel)
    #[inline(always)]
    pub fn queue_count(&self) -> u64 {
        self.queue_count.load(Relaxed)
    }

    /// how many Runnable queues within one second (in average)
    #[inline(always)]
    pub fn queue_frequency(&self) -> f64 {
        self.queue_frequency.load(Relaxed)
    }
}

/// the Profile of Future.
pub struct FutureProfile {
    alive_count: AtomicU64,
    poll_count: AtomicU64,
    pending_count: AtomicU64,
    ready_count: AtomicU64,
}
impl FutureProfile {
    /// get the global instance of FutureProfile.
    #[inline(always)]
    pub const fn global() -> &'static Self {
        static GLOBAL: FutureProfile =
            FutureProfile {
                alive_count: AtomicU64::new(0),
                poll_count: AtomicU64::new(0),
                pending_count: AtomicU64::new(0),
                ready_count: AtomicU64::new(0),
            };

        &GLOBAL
    }

    /// how many futures alive currently?
    #[inline(always)]
    pub fn alive_count(&self) -> u64 {
        self.alive_count.load(Relaxed)
    }

    /// total numer of Future::poll() calls. 
    #[inline(always)]
    pub fn poll_count(&self) -> u64 {
        self.poll_count.load(Relaxed)
    }

    /// total numer of Poll::Pending results.
    #[inline(always)]
    pub fn pending_count(&self) -> u64 {
        self.pending_count.load(Relaxed)
    }

    /// total numer of Poll::Ready results.
    #[inline(always)]
    pub fn ready_count(&self) -> u64 {
        self.ready_count.load(Relaxed)
    }
}

/// the Profile of asyncute.
pub struct Profile {
    /// the started time of Profile.
    started: AtomicInstant,

    /// RunnableProfile.
    pub runnable: &'static RunnableProfile,

    /// FutureProfile.
    pub future: &'static FutureProfile,
}
impl Profile {
    /// get the global instance of Profile.
    #[inline(always)]
    pub const fn global() -> &'static Self {
        static GLOBAL: Profile =
            Profile {
                started: AtomicInstant::init_now(),
                runnable: RunnableProfile::global(),
                future: FutureProfile::global(),
            };

        &GLOBAL
    }

    /// try to starting profile.
    ///
    /// return false if the profile already started.
    #[inline(always)]
    pub fn start(&self) -> bool {
        ProfileConfig::global().enable()
    }

    /// stop the profile.
    #[inline(always)]
    pub fn stop(&self) {
        ProfileConfig::global().disable();
    }

    /// check whether profile is started, and return the start time.
    #[inline(always)]
    pub fn started(&self) -> Option<Instant> {
        if ProfileConfig::global().is_enabled() {
            self.started.peek()
        } else {
            None
        }
    }
}

/// Configuration of Profile.
#[derive(Debug)]
pub struct ProfileConfig {
    /// whether enables profile recording?
    pub enabled: AtomicBool,

    /// output UDP socket peer of recorded profiles.
    ///
    /// for interval 10 seconds, if "the port is zero" then prints to stderr, otherwise it will sends datagram to provided SocketAddr and not to print.
    pub remote: AtomicSocketAddr,
}
impl ProfileConfig {
    /// get the global instance of ProfileConfig.
    #[inline(always)]
    pub const fn global() -> &'static Self {
        static GLOBAL: ProfileConfig =
            ProfileConfig {
                enabled: AtomicBool::new(false),
                remote: AtomicSocketAddr::default(),
            };

        &GLOBAL
    }

    /// to enable (start) the profile.
    ///
    /// return false if the profile already started.
    #[inline(always)]
    pub fn enable(&self) -> bool {
        if self.is_enabled() {
            return false;
        }

        Profile::global().started.set(Instant::now());
        self.enabled.store(true, Relaxed);

        true
    }


    /// to disable (stop) the profile.
    #[inline(always)]
    pub fn disable(&self) {
        self.enabled.store(false, Relaxed);
    }

    /// check whether the profile is enabled.
    #[inline(always)]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Relaxed)
    }
}

/// Defines how the executor spawns additional threads to handle workload.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExecutorSpawnPolicy {
    /// # "On-Demand"
    /// Conservative strategy: only spawns new temporary threads when the current load exceeds a configured threshold.
    ///
    /// * Resource Usage: Low to Medium — minimizes idle threads until they are needed.
    /// * Performance: Moderate — may introduce slight latency when threads are created on demand.
    /// * Responsiveness: Normal — reacts after workload increases, not preemptively.
    /// * Suitable Scenarios: Workloads with intermittent bursts where conserving resources is more important than immediate thread availability.
    OnDemand = Self::ON_DEMAND,

    /// # "Proactive"
    /// Aggressive strategy: maintains a pool of idle temporary threads in advance to reduce latency spikes.
    ///
    /// * Resource Usage: Medium to High — keeps spare threads ready, increasing baseline resource consumption.
    /// * Performance: High — immediate thread availability minimizes task queuing delays.
    /// * Responsiveness: Fast — can service sudden spikes in workload without waiting for thread creation.
    /// * Suitable Scenarios: Low-latency or real-time systems where consistent quick response is critical.
    Proactive = Self::PROACTIVE,

    /// # "Fixed"
    /// Static strategy: uses only the configured total thread count, never spawning or retiring temporary threads.
    ///
    /// * Resource Usage: Low & Predictable — consumes a constant, fixed number of threads.
    /// * Performance: Medium & Consistent — throughput remains steady without dynamic fluctuations.
    /// * Responsiveness: Limited by fixed capacity — may queue tasks if concurrency demand exceeds the fixed thread pool.
    /// * Suitable Scenarios: Environments with stable, well-known concurrency requirements or strict resource constraints.
    ///
    /// NOTE: please do not use this to doing blocking I/O.
    Fixed = Self::FIXED,
}

impl ExecutorSpawnPolicy {
    /// enum value of `Self::OnDemand`
    pub const ON_DEMAND: u8 = b'd';

    /// enum value of `Self::Proactive`
    pub const PROACTIVE: u8 = b'p';

    /// enum value of `Self::Fixed`
    pub const FIXED:     u8 = b'f';

    /// try to create `ExecutorSpawnPolicy` from u8 value if valid.
    #[inline(always)]
    pub const fn new(val: u8) -> Option<Self> {
        match val {
            Self::ON_DEMAND => Some(Self::OnDemand),
            Self::PROACTIVE => Some(Self::Proactive),
            Self::FIXED     => Some(Self::Fixed),
            _ => None
        }
    }

    /// get the enum value of `ExecutorSpawnPolicy`.
    #[inline(always)]
    pub const fn value(self) -> u8 {
        match self {
            Self::OnDemand  => Self::ON_DEMAND,
            Self::Proactive => Self::PROACTIVE,
            Self::Fixed     => Self::FIXED,
        }
    }

    /// check whether this policy is "on-demand".
    #[inline(always)]
    pub const fn is_ondemand(self) -> bool {
        match self {
            Self::OnDemand => true,
            _ => false
        }
    }

    /// check whether this policy is "proactive".
    #[inline(always)]
    pub const fn is_proactive(self) -> bool {
        match self {
            Self::Proactive => true,
            _ => false
        }
    }

    /// check whether this policy is "fixed".
    #[inline(always)]
    pub const fn is_fixed(self) -> bool {
        match self {
            Self::Fixed => true,
            _ => false
        }
    }

    /// convert to AtomicU8.
    #[inline(always)]
    pub const fn to_atomic(self) -> AtomicU8 {
        AtomicU8::new(self.value())
    }

    /// load from AtomicU8.
    #[inline(always)]
    pub fn from_atomic(atom: &AtomicU8) -> Option<Self> {
        Self::new(atom.load(Relaxed))
    }

    /// spawn one exitable executor if allowed.
    #[inline(always)]
    fn spawn_temporary_executor(self, status: &ExecutorStatus) -> bool {
        if self.is_fixed() {
            return false;
        }

        let range = ExecutorConfig::global().temporary_threads_range.range();

        let len = status.temporary.len();

        if len < range.start {
            log::info!("spawn new exitable executor due to temporary threads <= minimum number.");
            spawn_executor(true);
            return true;
        }

        if len >= range.end {
            let mut ids = status.temporary.clone();
            let mut id;
            let g = scc2::ebr::Guard::new();
            while ids.len() >= range.end {
                id = match ids.pop() {
                         Some(v) => v,
                         _ => {
                             break;
                         }
                     };
                if let Some(state) = EXECUTOR_INDEX.peek(&id, &g) {
                    assert!(state.exitable);
                    log::info!("request {} to exit (it exceeds the limit of temporary executors)", id.count());
                    state.exit().unwrap();
                }
            }
            return false;
        }

        let overload_threshold = ExecutorConfig::global().overload_threshold.load(Relaxed);

        let mut all_overload = true;
        for (id, load) in status.work_load.iter() {
            if (*load) < overload_threshold {
                all_overload = false;
            } else {
                log::warn!("executor {} is overloading: {load}", id.count());
            }
        }

        let backlog = get_runinfo_tx().len();
        if
            status.idle.len() <= 1
            ||
            all_overload
            ||
            backlog > (status.running.len() * 10)
            ||
            backlog > 1000
        {
            log::info!("monitored high overload in all executors: spawn new exitable executor");

            spawn_executor(true);
            return true;
        }

        false
    }
}

/// Configuration of Executor.
#[derive(Debug)]
pub struct ExecutorConfig {
    /// interval of executor's recv_timeout.
    interval: AtomicDuration,

    /// `minimum..=maximum` number of total executor threads.
    pub total_threads_range: AtomicRangeStrict<AtomicUsize>,

    /// maximum number of temporary executors.
    /// if setting to 0, it will disable preempt (spawn new temporary executor if too busy).
    ///
    /// it is not recommended to disable this, due to Asyncute is designed for "hang-up-free".
    ///
    /// for example, if user runs blocking I/O or CPU-bound tasks in one of executor (or even many executors), then some new temporary executors will be spawn "as soon as busy".
    ///
    /// this is primacy different in "blocking handle" between Asyncute and Goroutine.
    ///
    /// in Golang, if a goroutine is used for blocking, all of asynchronous IO poll tasks will be moved to a native OS thread other than the thread of blocking goroutine used to executes blocking operations.
    /// this is not essential, and make the design more complexity.
    ///
    /// in Asyncute, if a Future is used for blocking, this is no problem!
    /// all other idle executors will continues to handle async tasks, and no move needed (due to asyncute does not have per-executor queues for working-stealing).
    pub temporary_threads_range: AtomicRangeStrict<AtomicUsize>,

    /// Policy that control the spawning of Executor.
    spawn_policy: AtomicU8,

    /// if a executor's working ratio >= this threshold, it will be assumed as "overload" (busy).
    overload_threshold: AtomicF64,

    /// if a executor is temporary (exitable), and it's working ratio <= this threshold, it will be exited automatically.
    standby_threshold: AtomicF64,
}

impl ExecutorConfig {
    /// the minimum interval of executor loop.
    pub const MIN_INTERVAL: Duration = Duration::from_millis(100);

    /// the maximum interval of executor loop.
    pub const MAX_INTERVAL: Duration = Duration::new(5, 0);

    /// get the global instance of ExecutorConfig.
    #[inline(always)]
    pub const fn global() -> &'static Self {
        static GLOBAL: ExecutorConfig =
            ExecutorConfig {
                interval: AtomicDuration::new_with_trace(1, 0),
                total_threads_range: AtomicRangeStrict::<AtomicUsize>::new(1, usize::MAX),
                temporary_threads_range: AtomicRangeStrict::<AtomicUsize>::new(0, usize::MAX),
                spawn_policy: ExecutorSpawnPolicy::OnDemand.to_atomic(),
                overload_threshold: AtomicF64::new(0.85),
                standby_threshold: AtomicF64::new(0.5),
            };

        &GLOBAL
    }

    /// get the executor interval.
    pub fn interval(&self) -> Duration {
        self.interval.get()
    }

    /// set the executor interval.
    pub fn set_interval(&self, val: Duration) -> bool {
        if val < Self::MIN_INTERVAL || val > Self::MAX_INTERVAL {
            return false;
        }

        self.interval.set(val);
        true
    }

    /// get the executor spawn policy.
    #[inline(always)]
    pub fn spawn_policy(&self) -> ExecutorSpawnPolicy {
        ExecutorSpawnPolicy::from_atomic(&self.spawn_policy).expect("unexpected unknown variant of ExecutorSpawnPolicy")
    }

    /// set the executor spawn policy.
    #[inline(always)]
    pub fn set_spawn_policy(&self, policy: ExecutorSpawnPolicy) {
        self.spawn_policy.store(policy.value(), Relaxed);
    }

    /// get the threshold of overload.
    #[inline(always)]
    pub fn overload_threshold(&self) -> f64 {
        let val = self.overload_threshold.load(Relaxed);
        assert!(val >= 0.0 && val <= 1.0);
        val
    }

    /// set the threshold of overload.
    #[inline(always)]
    pub fn set_overload_threshold(&self, val: f64) -> bool {
        // check special values
        if val.is_nan() || val.is_infinite() || val.is_subnormal() {
            return false;
        }

        // check range
        if val < 0.0 || val > 1.0 {
            return false;
        }

        self.overload_threshold.store(val, Relaxed);
        true
    }

    /// get the threshold of standby.
    #[inline(always)]
    pub fn standby_threshold(&self) -> f64 {
        let val = self.standby_threshold.load(Relaxed);
        assert!(val >= 0.0 && val <= 1.0);
        val
    }

    /// set the threshold of standby.
    #[inline(always)]
    pub fn set_standby_threshold(&self, val: f64) -> bool {
        // check special values
        if val.is_nan() || val.is_infinite() || val.is_subnormal() {
            return false;
        }

        // check range
        if val < 0.0 || val > 1.0 {
            return false;
        }

        self.overload_threshold.store(val, Relaxed);
        true
    }
}

/// Configuration of Monitor.
#[derive(Debug)]
pub struct MonitorConfig {
    interval: AtomicDuration,
}

impl MonitorConfig {
    /// the minimum interval of monitor loop.
    pub const INTERVAL_MIN: Duration = Duration::new(1, 0);

    /// the maximum interval of monitor loop.
    pub const INTERVAL_MAX: Duration = Duration::new(10, 0);

    /// get the global instance of MonitorConfig.
    #[inline(always)]
    pub const fn global() -> &'static Self {
        static GLOBAL: MonitorConfig =
            MonitorConfig {
                interval: AtomicDuration::new_with_trace(3, 0),
            };

        &GLOBAL
    }

    /// get the monitor interval
    #[inline(always)]
    pub fn interval(&self) -> Duration {
        self.interval.get()
    }

    /// set monitor "sleep" interval for `std::thread::park_timeout`
    /// interval seconds must in range `1..=10`
    #[inline(always)]
    pub fn set_interval(&self, mut interval: Duration) -> Duration {
        interval = interval.clamp(Self::INTERVAL_MIN, Self::INTERVAL_MAX);
        self.interval.set(interval);
        interval
    }
}

/// The Configuration of Asyncute.
#[derive(Debug)]
pub struct Config {
    /// Profile Configuration.
    pub profile: &'static ProfileConfig,

    /// Executor Configuration.
    pub executor: &'static ExecutorConfig,

    /// Monitor Configuration.
    pub monitor: &'static MonitorConfig,
}

impl Config {
    /// get the global instance of Config.
    #[inline(always)]
    pub const fn global() -> &'static Self {
        static GLOBAL: Config =
            Config {
                profile: ProfileConfig::global(),
                executor: ExecutorConfig::global(),
                monitor: MonitorConfig::global(),
            };

        &GLOBAL
    }

    /// shortcut of `self.executor.total_threads_range.set_range(range)`
    #[inline(always)]
    pub fn set_threads(&self, range: Range<usize>) -> bool {
        self.executor.total_threads_range.set_range(range)
    }
}

/// the status snapshot of executors.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExecutorStatus {
    /// whether this struct initialized?
    last_update: Option<Instant>,

    /// total executors count, includes exited, running, working executors.
    pub total: Vec<ExecutorId>,

    /// working executors is busying for running Runnables.
    pub working: Vec<ExecutorId>,

    /// idle executors is not busy. it just waiting for new Runnables.
    pub idle: Vec<ExecutorId>,

    /// running executors means it's thread is running (not exited).
    pub running: Vec<ExecutorId>,

    /// removes executors due to it has been exited.
    pub remove: Vec<ExecutorId>,

    /// temporary executors is can be exit if there is no Runnables a duration.
    pub temporary: Vec<ExecutorId>,

    /// persist executors will never exits even there is no Runnables.
    pub persist: Vec<ExecutorId>,

    /// the work load of all executors. tuple = (u128 ID, f64 Ratio).
    /// the f64 is means the percent of working time. that is within 0.0~1.0
    /// * 0.0 means 0% of time is working.
    /// * 0.2 means 20% of time is working.
    /// * 1.0 means 100% of time is working.
    pub work_load: Vec<(ExecutorId, f64)>,
}
impl ExecutorStatus {
    /// create new empty status.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            last_update: None,

            total: Vec::new(),

            working: Vec::new(),
            idle: Vec::new(),

            running: Vec::new(),
            remove: Vec::new(),

            temporary: Vec::new(),
            persist: Vec::new(),

            work_load: Vec::new(),
        }
    }

    /// get the last update time of status.
    #[inline(always)]
    pub const fn last_update(&self) -> Option<Instant> {
        self.last_update
    }

    /// get the current status.
    #[inline(always)]
    pub fn current() -> Self {
        let mut this = Self::new();
        this.update();
        this
    }

    /// update the status of current executors.
    #[inline(always)]
    pub fn update(&mut self) -> &mut Self {
        self.total.clear();

        self.working.clear();
        self.idle.clear();

        self.running.clear();
        self.remove.clear();

        self.temporary.clear();
        self.persist.clear();

        self.work_load.clear();

        let g = scc2::ebr::Guard::new();
        for (id, state) in EXECUTOR_INDEX.iter(&g) {
            if self.total.contains(id) {
                continue;
            }
            self.total.push(*id);

            self.work_load.push((*id, state.working_ratio()));

            if state.exitable {
                self.temporary.push(*id);
            } else {
                self.persist.push(*id);
            }

            if let Some(jh) =
                state.join_handle.get()
            {
                if jh.is_finished() {
                    self.remove.push(*id);
                } else {
                    self.running.push(*id);

                    if state.is_working() {
                        self.working.push(*id);
                    } else {
                        self.idle.push(*id);
                    }
                }
            }
        }
        drop(g);

        for id in self.remove.iter() {
            EXECUTOR_INDEX.remove(id);
        }

        self.last_update = Some(Instant::now());

        self
    }
}

/// force to spawn new executor.
#[inline(always)]
pub fn spawn_executor(exitable: bool) {
    let id = gen_executor_id().expect("Executor ID exhausted!");

    let mut state = ExecutorState::new(id);
    state.exitable = exitable;
    let state = Arc::new(state);

    let jh = {
        let state = state.clone();
        let rx = get_runinfo_rx().clone();

        let flag: &'static str =
            if exitable {
                "t" // temporary
            } else {
                "p" // persist
            };

        let parent = std::thread::current();

        let idc = id.count();

        std::thread::Builder::new()
            .name(format!("ac-exec-{idc}-{flag}"))
            .stack_size(1048576 * 20)
            .spawn(move || {
                parent.unpark();
                let executor = Executor::new(state, rx);
                executor.run().expect("executor error");
            }).expect("unable to spawn thread for new executor!")
    };
    state.join_handle.set(Arc::new(jh)).expect("unable to set join handle for ExecutorState");

    EXECUTOR_INDEX.insert(id, state).expect("Executor ID should be unique but duplicated!");

    std::thread::park_timeout(Duration::from_secs(3));
}

/// # Deprecated
/// please use `Config::global().monitor.interval()` instead.
///
/// get monitor interval
#[inline(always)]
#[deprecated(since="0.0.3", note="please use `Config::global().monitor.interval()` instead.")]
pub fn get_monitor_interval() -> Duration {
    MonitorConfig::global().interval()
}

/// # Deprecated
/// please use `Config::global().monitor.set_interval()` instead.
///
/// set monitor "sleep" interval for `std::thread::park_timeout`
/// interval seconds must in range `1..=10`
#[inline(always)]
#[deprecated(since="0.0.3", note="please use `Config::global().monitor.set_interval()` instead.")]
pub fn set_monitor_interval(interval: Duration) -> Duration {
    MonitorConfig::global().set_interval(interval)
}

/// the monitor loop.
#[inline(always)]
fn monitor_loop() {
    // for prevent monitor thread exit unexpectedly if before set Atom.
    std::thread::park_timeout(Duration::from_secs(15));

    let current_thread_id = std::thread::current().id();

    let mut status = ExecutorStatus::new();
    let cpus: usize = cpu_count();

    let config = Config::global();
    let mut interval = config.monitor.interval();
    loop {
        if config.monitor.interval.changed() {
            interval = config.monitor.interval();
        }

        if let Some(jh) = MONITOR_THREAD_JH.get() {
            if jh.is_finished() {
                start_monitor().unwrap();
                return;
            }
            if jh.thread().id() != current_thread_id {
                start_monitor().unwrap();
                return;
            }
        } else {
            start_monitor().unwrap();
            return;
        }

        while status.update().running.len() < cpus {
            log::info!("monitor spawn new persist executor");

            spawn_executor(false);
        }

        log::trace!("monitor status = {:?}", &status);

        config.executor.spawn_policy().spawn_temporary_executor(&status);

        if ProfileConfig::global().is_enabled() {
            RunnableProfile::global().update();
        }

        std::thread::park_timeout(interval);
    }
}

/// Check whether the monitor is running normally.
#[inline(always)]
pub fn is_monitor_running() -> bool {
    if let Some(jh) = MONITOR_THREAD_JH.get() {
        ! jh.is_finished()
    } else {
        false
    }
}

/// wake the monitor manually.
#[inline(always)]
pub fn wake_monitor() -> bool {
    if let Some(jh) = MONITOR_THREAD_JH.get() {
        if ! jh.is_finished() {
            jh.thread().unpark();
            return true;
        }
    }

    false
}

/// start monitor if it is not exists or exited.
#[inline(always)]
pub fn start_monitor() -> std::io::Result<()> {
    static STARTING: AtomicBool = AtomicBool::new(false);

    let mut defer = Defer::new(|| {
        let _ = STARTING.compare_exchange(true, false, Relaxed, Relaxed);
    });
    if STARTING.compare_exchange(false, true, Relaxed, Relaxed).is_err() {
        defer.cancel();
        return Err(std::io::Error::new(std::io::ErrorKind::ResourceBusy, "monitor is starting by another caller"));
    }

    if is_monitor_running() {
        defer.run();
        return Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "another monitor running."));
    }

    let jh =
        std::thread::Builder::new()
        .name(String::from("ac-monitor"))
        .spawn(monitor_loop)?;
    let jh = scc2::ebr::Shared::new(jh);
    MONITOR_THREAD_JH.set_shared(jh.clone());
    jh.thread().unpark();

    defer.run();
    Ok(())
}

/// the scheduler of asyncute.
#[inline(always)]
pub fn scheduler(
    runnable: Runnable,
    info: ScheduleInfo
) {
    #[cfg(test)]
    log::trace!("scheduler called");

    let tx = get_runinfo_tx();

    #[cfg(feature="crossbeam-deque")]
    tx.send(RunInfo { runnable, info });

    #[cfg(not(feature="crossbeam-deque"))]
    tx.send(RunInfo { runnable, info })
      .expect("unable to send Runnable to executors: channel closed");

    #[cfg(test)]
    log::trace!("scheduler sent. tx.len()={}", tx.len());

    if tx.len() > 100 || ExecutorConfig::global().spawn_policy().is_proactive() {
        wake_monitor();
    }

    if ProfileConfig::global().is_enabled() {
        let p = Profile::global();
        p.started();
        p.runnable.queue_count.checked_add(1);
    }
}

/// the `scheduler` function wrapped by `async_task::WithInfo`.
const SCHEDULER: async_task::WithInfo<fn(Runnable, ScheduleInfo)> = async_task::WithInfo(scheduler);

/// spawn new `Future` to asyncute, and return the Task handle associated to this Future.
#[inline(always)]
pub fn spawn<F>(f: F) -> Task<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let _ = start_monitor();

    /// the wrapped future for profile.
    struct WrappedFuture<T, F: Future<Output=T>> {
        dropped: bool,

        id: TaskId,
        fut: Pin<Box<F>>,
    }

    impl<T, F: Future<Output=T>> WrappedFuture<T, F> {
        /// create new WrappedFuture.
        #[inline(always)]
        fn new(id: TaskId, f: F) -> Self {
            let mut this =
                Self {
                    dropped: true,

                    id,
                    fut: Box::pin(f),
                };

            if ProfileConfig::global().is_enabled() {
                let fp = FutureProfile::global();
                if fp.alive_count.checked_add(1).is_some() {
                    this.dropped = false;
                }
            }

            this
        }
    }

    impl<T, F: Future<Output=T>> Future for WrappedFuture<T, F> {
        type Output = T;

        #[inline(always)]
        fn poll(mut self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<T> {
            let res = self.fut.as_mut().poll(ctx);

            if ProfileConfig::global().is_enabled() {
                let fp = FutureProfile::global();
                fp.poll_count.checked_add(1);

                if res.is_pending() {
                    fp.pending_count.checked_add(1);
                } else if res.is_ready() {
                    fp.ready_count.checked_add(1);
                }
            }

            res
        }
    }

    impl<T, F: Future<Output=T>> Drop for WrappedFuture<T, F> {
        #[inline(always)]
        fn drop(&mut self) {
            if self.dropped {
                return;
            }
            self.dropped = true;

            FutureProfile::global().alive_count.checked_sub(1);
        }
    }

    let (runnable, task) = {
        let b =
            async_task::Builder::new()
            .propagate_panic(true)
            .metadata(TaskInfo::new(0));

        if ProfileConfig::global().is_enabled() {
            b.spawn(move |taskinfo| { WrappedFuture::new(taskinfo.id, f) }, SCHEDULER)
        } else {
            b.spawn(move |_| { f }, SCHEDULER)
        }
    };

    runnable.schedule();
    task
}

