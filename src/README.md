# async + execute = asyncute is a new async executor that execute async futures globally.

## Features:
1. Trace status and auto remove dead executor, and auto start new executor thread if all exists executor is working or too high backlogs.
2. unlike smolscale, there is no per-executor queues for queuing Runnable, asyncute only have single global MPMC queue (and all executors receives Runnable from it), so Executor no need to push Runnable back to global queue on droppped.
3. unlike other library that uses work-stealing queue, asyncute uses bounded channel for get notify if Runnable available, and if channel is full, spawn() will blocking instead of discarding.
4. uses propagate-panic always, uses `std::panic::catch_unwind` always. avoid single async task causes entire executor exit unexpectedly.
5. analysis and profile. it can easy to get the executor's status (idle, working, running, work loads, etc.), and start/stop profile in runtime to see average Runnable took how many time to run, current alive Runnables, count of ready/pending polls, etc.
6. spawn new executor manually, wake monnitor manually, and start monitor manually.
7. runtime configurable Config.

