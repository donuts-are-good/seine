use std::sync::OnceLock;
use std::thread;

use crossbeam_channel::{unbounded, Sender};

type Task = Box<dyn FnOnce() + Send + 'static>;

struct DispatchPool {
    sender: Sender<Task>,
}

impl DispatchPool {
    fn new() -> Self {
        let (sender, receiver) = unbounded::<Task>();
        let workers = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(4);
        for idx in 0..workers {
            let receiver = receiver.clone();
            thread::Builder::new()
                .name(format!("seine-dispatch-{idx}"))
                .spawn(move || {
                    while let Ok(task) = receiver.recv() {
                        task();
                    }
                })
                .expect("dispatch pool worker should spawn");
        }
        Self { sender }
    }

    fn submit(&self, task: Task) {
        self.sender
            .send(task)
            .expect("dispatch pool sender should remain available");
    }
}

fn pool() -> &'static DispatchPool {
    static POOL: OnceLock<DispatchPool> = OnceLock::new();
    POOL.get_or_init(DispatchPool::new)
}

pub(super) fn submit(task: Task) {
    pool().submit(task)
}
