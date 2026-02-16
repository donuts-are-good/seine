use std::time::Duration;

/// Hardcoded dev fee wallet address (base58-encoded spend+view pubkeys).
pub const DEV_ADDRESS: &str = "Seinevoovs55xaLmMb5aCNb3pfqtzwNgJZQtLjGMQus5Ji4YrUJBgS42v1G8kHDdmqMn8xxQjsSpdJAdWvP1RYQ";

/// Dev fee percentage of total mining time.
pub const DEV_FEE_PERCENT: f64 = 2.5;

pub struct DevFeeTracker {
    fee_fraction: f64,
    total_elapsed: Duration,
    dev_elapsed: Duration,
    is_dev_round: bool,
}

impl DevFeeTracker {
    pub fn new() -> Self {
        Self {
            fee_fraction: DEV_FEE_PERCENT / 100.0,
            total_elapsed: Duration::ZERO,
            dev_elapsed: Duration::ZERO,
            is_dev_round: false,
        }
    }

    /// Returns true when dev fee mining is owed (dev_elapsed < total_elapsed * fee_fraction).
    fn should_mine_dev(&self) -> bool {
        if self.fee_fraction <= 0.0 {
            return false;
        }
        let owed = self.total_elapsed.as_secs_f64() * self.fee_fraction;
        self.dev_elapsed.as_secs_f64() < owed
    }

    /// Call at the start of each round. Sets whether this round mines for dev.
    /// Returns true if the mode changed from the previous round.
    pub fn begin_round(&mut self) -> bool {
        let was_dev = self.is_dev_round;
        self.is_dev_round = self.should_mine_dev();
        self.is_dev_round != was_dev
    }

    /// Call at the end of each round to accumulate elapsed time.
    pub fn end_round(&mut self, elapsed: Duration) {
        self.total_elapsed += elapsed;
        if self.is_dev_round {
            self.dev_elapsed += elapsed;
        }
    }

    pub fn is_dev_round(&self) -> bool {
        self.is_dev_round
    }

    /// Returns `Some(DEV_ADDRESS)` during dev rounds, `None` during user rounds.
    pub fn address(&self) -> Option<&'static str> {
        if self.is_dev_round {
            Some(DEV_ADDRESS)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_round_is_not_dev() {
        let mut tracker = DevFeeTracker::new();
        tracker.begin_round();
        // With zero elapsed time, owed dev time is 0, so should not be dev round
        assert!(!tracker.is_dev_round());
    }

    #[test]
    fn dev_round_triggers_after_user_mining() {
        let mut tracker = DevFeeTracker::new();

        // First round: user mining
        tracker.begin_round();
        assert!(!tracker.is_dev_round());
        tracker.end_round(Duration::from_secs(10));

        // Second round: now owed 0.25s dev time with 0s mined => dev round
        tracker.begin_round();
        assert!(tracker.is_dev_round());
    }

    #[test]
    fn returns_to_user_after_dev_round() {
        let mut tracker = DevFeeTracker::new();

        // User round: 10s
        tracker.begin_round();
        tracker.end_round(Duration::from_secs(10));

        // Dev round: 1s (owed 0.25s from 10s, now total=11s, dev=1s, owed=0.275s)
        tracker.begin_round();
        assert!(tracker.is_dev_round());
        tracker.end_round(Duration::from_secs(1));

        // Next round: dev_elapsed(1.0) > owed(11*0.025=0.275) => user round
        tracker.begin_round();
        assert!(!tracker.is_dev_round());
    }

    #[test]
    fn begin_round_reports_mode_change() {
        let mut tracker = DevFeeTracker::new();

        let changed = tracker.begin_round();
        assert!(!changed); // first round, was false, still false

        tracker.end_round(Duration::from_secs(10));

        let changed = tracker.begin_round();
        assert!(changed); // switched to dev
    }

    #[test]
    fn address_returns_dev_address_during_dev_round() {
        let mut tracker = DevFeeTracker::new();
        tracker.begin_round();
        assert!(tracker.address().is_none());

        tracker.end_round(Duration::from_secs(10));
        tracker.begin_round();
        assert_eq!(tracker.address(), Some(DEV_ADDRESS));
    }
}
