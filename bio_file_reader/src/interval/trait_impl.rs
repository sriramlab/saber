use std::fmt;

use num::Integer;
use num::traits::NumAssignOps;

use crate::interval::{Interval, IntervalOverIntegers, MergeIntervals};

impl<E: Integer + NumAssignOps + Copy + fmt::Display> MergeIntervals<IntervalOverIntegers<E>, E> for Vec<IntervalOverIntegers<E>> {
    fn sort_and_coalesce_intervals(&self) -> Vec<IntervalOverIntegers<E>> {
        let mut intervals: Vec<IntervalOverIntegers<E>> = self.to_vec();
        intervals.sort_and_coalesce_intervals_inplace();
        intervals
    }

    fn sort_and_coalesce_intervals_inplace(&mut self) {
        self.sort_by_key(|i| i.get_start());
        let mut coalesced_intervals = Vec::new();
        for interval in self.into_iter() {
            match coalesced_intervals.last_mut() {
                None => coalesced_intervals.push(*interval),
                Some(last_interval) => {
                    match last_interval.coalesce_with(interval) {
                        None => coalesced_intervals.push(*interval),
                        Some(new_interval) => *last_interval = new_interval,
                    }
//                    if interval.get_start() <= last_interval.get_end() {
//                        last_interval.set_end(max(last_interval.get_end(), interval.get_end()));
//                    } else {
//                        coalesced_intervals.push(*interval);
//                    }
                }
            }
        }
        *self = coalesced_intervals;
    }
}

#[cfg(test)]
mod tests {
    use crate::interval::IntervalOverIntegers;

    use super::MergeIntervals;

    #[test]
    fn test_sort_and_coalesce_intervals() {
        let intervals = vec![
            IntervalOverIntegers::new(2, 4),
            IntervalOverIntegers::new(1, 1),
            IntervalOverIntegers::new(-10, -5),
            IntervalOverIntegers::new(4, 5),
            IntervalOverIntegers::new(9, 10),
            IntervalOverIntegers::new(-1, 3)
        ];
        let sorted_intervals = intervals.sort_and_coalesce_intervals();
        assert_eq!(sorted_intervals, vec![
            IntervalOverIntegers::new(-10, -5),
            IntervalOverIntegers::new(-1, 5),
            IntervalOverIntegers::new(9, 10)
        ])
    }
}
