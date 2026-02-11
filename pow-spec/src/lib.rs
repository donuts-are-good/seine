use argon2::{Error, Params};

pub const POW_MEMORY_KB: u32 = 2 * 1024 * 1024;
pub const POW_ITERATIONS: u32 = 1;
pub const POW_PARALLELISM: u32 = 1;
pub const POW_OUTPUT_LEN: usize = 32;
pub const POW_HEADER_BASE_LEN: usize = 92;

pub const CPU_LANE_MEMORY_BYTES: u64 = (POW_MEMORY_KB as u64) * 1024;

pub fn pow_params() -> Result<Params, Error> {
    Params::new(
        POW_MEMORY_KB,
        POW_ITERATIONS,
        POW_PARALLELISM,
        Some(POW_OUTPUT_LEN),
    )
}
