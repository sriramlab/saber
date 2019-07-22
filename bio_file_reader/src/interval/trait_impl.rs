use std::fmt;

use num::Integer;
use num::traits::NumAssignOps;

use crate::interval::traits::{Coalesce, CoalesceIntervals, Interval};

impl<I: Coalesce + Interval<Element=E> + Clone, E: Integer + NumAssignOps + Copy + fmt::Display> CoalesceIntervals<I, E> for Vec<I> {
    fn sort_and_coalesce_intervals(&self) -> Vec<I> {
        let mut intervals: Vec<I> = self.to_vec();
        intervals.sort_and_coalesce_intervals_inplace();
        intervals
    }

    fn sort_and_coalesce_intervals_inplace(&mut self) {
        self.sort_by_key(|i| i.get_start());
        let mut coalesced_intervals = Vec::new();
        for interval in self.drain(..) {
            match coalesced_intervals.last_mut() {
                None => coalesced_intervals.push(interval),
                Some(last_interval) => {
                    match last_interval.coalesce_with(&interval) {
                        None => coalesced_intervals.push(interval),
                        Some(new_interval) => *last_interval = new_interval,
                    }
                }
            }
        }
        *self = coalesced_intervals;
    }
}

#[cfg(test)]
mod tests {
    use crate::set::ContiguousIntegerSet;

    use super::CoalesceIntervals;

    #[test]
    fn test_sort_and_coalesce_intervals() {
        let intervals = vec![
            ContiguousIntegerSet::new(2, 4),
            ContiguousIntegerSet::new(1, 1),
            ContiguousIntegerSet::new(-10, -5),
            ContiguousIntegerSet::new(4, 5),
            ContiguousIntegerSet::new(9, 10),
            ContiguousIntegerSet::new(-1, 3)
        ];
        let sorted_intervals = intervals.sort_and_coalesce_intervals();
        assert_eq!(sorted_intervals, vec![
            ContiguousIntegerSet::new(-10, -5),
            ContiguousIntegerSet::new(-1, 5),
            ContiguousIntegerSet::new(9, 10)
        ])
    }
}
