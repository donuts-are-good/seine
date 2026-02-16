use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use blocknet_pow_spec::{POW_MEMORY_KB, POW_OUTPUT_LEN};

use crate::backend::{BackendEvent, MiningSolution};
use crate::types::hash_meets_target;

use super::{
    emit_error, emit_event, fixed_argon, flush_hashes, lane_quota_for_chunk, mark_worker_active,
    mark_worker_inactive, mark_worker_ready, request_shutdown, request_work_pause,
    set_thread_high_perf, should_flush_hashes, wait_for_work_update, Shared,
    MAX_DEADLINE_CHECK_INTERVAL, SOLVED_MASK,
};

pub(super) fn cpu_worker_loop(
    shared: Arc<Shared>,
    thread_idx: usize,
    core_id: Option<core_affinity::CoreId>,
) {
    set_thread_high_perf();
    if let Some(core_id) = core_id {
        let _ = core_affinity::set_for_current(core_id);
    }

    let hasher = fixed_argon::FixedArgon2id::new(POW_MEMORY_KB);
    let block_count = hasher.block_count();

    // Allocate the 2 GiB arena with explicit huge pages (2 MB) to avoid
    // the catastrophic TLB miss overhead of 524 K × 4 KB pages.
    //
    // Strategy:
    //   1. Try mmap MAP_HUGETLB — guaranteed 2 MB pages from the hugetlbfs
    //      pool (requires /proc/sys/vm/nr_hugepages >= arena_size/2MB).
    //   2. Fall back to regular mmap + MADV_HUGEPAGE (THP).
    //   3. Fall back to Vec (non-Linux, or mmap failure).
    //
    // MAP_POPULATE pre-faults all pages so the hash loop never page-faults.
    let mut _arena_guard: Option<MmapArena> = None;
    let mut _vec_fallback: Option<Vec<fixed_argon::PowBlock>> = None;
    let memory_blocks: &mut [fixed_argon::PowBlock];

    #[cfg(target_os = "linux")]
    {
        let block_bytes = block_count * std::mem::size_of::<fixed_argon::PowBlock>();
        let arena = MmapArena::new(block_count, block_bytes);
        _arena_guard = Some(arena);
        memory_blocks = _arena_guard.as_mut().unwrap().as_mut_slice();
    }
    #[cfg(not(target_os = "linux"))]
    {
        let mut v: Vec<fixed_argon::PowBlock> = vec![fixed_argon::PowBlock::default(); block_count];
        _vec_fallback = Some(v);
        memory_blocks = _vec_fallback.as_mut().unwrap().as_mut_slice();
    }

    let mut output = [0u8; POW_OUTPUT_LEN];
    mark_worker_ready(&shared);
    let mut local_generation = 0u64;
    let mut local_work = None;
    let mut worker_active = false;
    let mut nonce = 0u64;
    let mut lane_iters = 0u64;
    let lane_stride = shared.hash_slots.len().max(1) as u64;
    let hash_batch_size = shared.hash_batch_size.max(1);
    let control_check_interval_hashes = shared.control_check_interval_hashes.max(1);
    let mut lane_quota = 0u64;
    let mut pending_hashes = 0u64;
    let mut next_flush_at = Instant::now() + shared.hash_flush_interval;
    let mut control_hashes_remaining = 0u64;
    let mut next_deadline_check_at = Instant::now();

    loop {
        let global_generation = shared.work_generation.load(Ordering::Acquire);
        if global_generation != local_generation || local_work.is_none() {
            if local_work.is_some() {
                flush_hashes(&shared, thread_idx, &mut pending_hashes);
                mark_worker_inactive(&shared, &mut worker_active);
            }
            match wait_for_work_update(&shared, local_generation) {
                Ok(Some((generation, work))) => {
                    nonce = work.nonce_chunk.start_nonce.wrapping_add(thread_idx as u64);
                    lane_iters = 0;
                    lane_quota = lane_quota_for_chunk(
                        work.nonce_chunk.nonce_count,
                        thread_idx as u64,
                        lane_stride,
                    );
                    next_flush_at = Instant::now() + shared.hash_flush_interval;
                    control_hashes_remaining = 0;
                    next_deadline_check_at = Instant::now();
                    local_generation = generation;
                    local_work = Some(work);
                    if lane_quota > 0 {
                        mark_worker_active(&shared, &mut worker_active);
                    } else {
                        local_work = None;
                    }
                    continue;
                }
                Ok(None) => break,
                Err(err) => {
                    emit_error(
                        &shared,
                        format!("cpu thread {thread_idx}: control wait failed ({err})"),
                    );
                    request_shutdown(&shared);
                    break;
                }
            }
        }

        let Some(work) = local_work.as_ref() else {
            continue;
        };
        let template = &work.template;

        if lane_iters >= lane_quota {
            flush_hashes(&shared, thread_idx, &mut pending_hashes);
            mark_worker_inactive(&shared, &mut worker_active);
            local_work = None;
            continue;
        }

        if shared.solution_state.load(Ordering::Acquire) != template.work_id {
            flush_hashes(&shared, thread_idx, &mut pending_hashes);
            mark_worker_inactive(&shared, &mut worker_active);
            local_work = None;
            continue;
        }

        let now = Instant::now();
        let deadline_check_due = now >= next_deadline_check_at;
        if control_hashes_remaining == 0 || deadline_check_due {
            if now >= template.stop_at {
                flush_hashes(&shared, thread_idx, &mut pending_hashes);
                mark_worker_inactive(&shared, &mut worker_active);
                local_work = None;
                continue;
            }

            if control_hashes_remaining == 0 {
                control_hashes_remaining = lane_quota
                    .saturating_sub(lane_iters)
                    .clamp(1, control_check_interval_hashes);
            }
            if deadline_check_due {
                next_deadline_check_at = now + MAX_DEADLINE_CHECK_INTERVAL;
            }
        }

        let nonce_bytes = nonce.to_le_bytes();
        if hasher
            .hash_password_into_with_memory(
                &nonce_bytes,
                &template.header_base,
                &mut output,
                memory_blocks,
            )
            .is_err()
        {
            emit_error(
                &shared,
                format!("cpu thread {thread_idx}: hash_password_into_with_memory failed"),
            );
            request_shutdown(&shared);
            break;
        }

        lane_iters += 1;
        control_hashes_remaining -= 1;
        pending_hashes += 1;

        let now = Instant::now();
        let should_flush = should_flush_hashes(pending_hashes, now, next_flush_at, hash_batch_size);
        if should_flush {
            flush_hashes(&shared, thread_idx, &mut pending_hashes);
            next_flush_at = now + shared.hash_flush_interval;
        }

        if hash_meets_target(&output, &template.target) {
            let solved_state = SOLVED_MASK | template.work_id;
            if shared
                .solution_state
                .compare_exchange(
                    template.work_id,
                    solved_state,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                if let Err(err) = request_work_pause(&shared) {
                    emit_error(
                        &shared,
                        format!(
                            "cpu thread {thread_idx}: failed to pause workers after solution ({err})"
                        ),
                    );
                    request_shutdown(&shared);
                    break;
                }
                emit_event(
                    &shared,
                    BackendEvent::Solution(MiningSolution {
                        epoch: template.epoch,
                        nonce,
                        backend_id: shared.instance_id.load(Ordering::Acquire),
                        backend: "cpu",
                    }),
                );
                flush_hashes(&shared, thread_idx, &mut pending_hashes);
                mark_worker_inactive(&shared, &mut worker_active);
                local_work = None;
                continue;
            }
        }

        nonce = nonce.wrapping_add(lane_stride);
    }

    flush_hashes(&shared, thread_idx, &mut pending_hashes);
    mark_worker_inactive(&shared, &mut worker_active);
}

/// RAII wrapper around a `mmap`-backed arena.  Tries MAP_HUGETLB (explicit
/// 2 MB pages from the hugetlbfs pool) first, then falls back to regular
/// anonymous mmap with MADV_HUGEPAGE (THP).  MAP_POPULATE pre-faults all
/// pages so the hash loop never takes a soft fault.
#[cfg(target_os = "linux")]
struct MmapArena {
    ptr: *mut u8,
    byte_len: usize,
    block_count: usize,
    _huge: bool,
}

#[cfg(target_os = "linux")]
impl MmapArena {
    fn new(block_count: usize, byte_len: usize) -> Self {
        // Attempt 1: MAP_HUGETLB for guaranteed 2 MB pages.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                byte_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | libc::MAP_POPULATE,
                -1,
                0,
            )
        };
        if ptr != libc::MAP_FAILED {
            return Self {
                ptr: ptr as *mut u8,
                byte_len,
                block_count,
                _huge: true,
            };
        }

        // Attempt 2: regular anonymous mmap + MADV_HUGEPAGE (THP).
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                byte_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            panic!(
                "mmap failed for {} byte arena: {}",
                byte_len,
                std::io::Error::last_os_error()
            );
        }
        unsafe {
            libc::madvise(ptr, byte_len, libc::MADV_HUGEPAGE);
        }
        Self {
            ptr: ptr as *mut u8,
            byte_len,
            block_count,
            _huge: false,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [fixed_argon::PowBlock] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut fixed_argon::PowBlock, self.block_count) }
    }

    #[allow(dead_code)]
    fn is_huge(&self) -> bool {
        self._huge
    }
}

#[cfg(target_os = "linux")]
impl Drop for MmapArena {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.byte_len);
        }
    }
}

// Safety: The mmap region is private to the thread that created it.
// The arena is only accessed by a single worker thread.
#[cfg(target_os = "linux")]
unsafe impl Send for MmapArena {}
#[cfg(target_os = "linux")]
unsafe impl Sync for MmapArena {}
