use std::cmp::{max, min};
use std::fmt;

use num::integer::Integer;
use num::traits::NumAssignOps;

use crate::interval::traits::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ContiguousIntegerSet<E: Integer + Copy> {
    start: E,
    end: E,
}

impl<E: Integer + Copy> ContiguousIntegerSet<E> {
    pub fn new(start: E, end: E) -> ContiguousIntegerSet<E> {
        ContiguousIntegerSet {
            start,
            end,
        }
    }
}

impl<E: Integer + Copy> Coalesce for ContiguousIntegerSet<E> {
    fn coalesce_with(&self, other: &Self) -> Option<Self> {
        if self.start > other.end + E::one() || self.end + E::one() < other.start {
            None
        } else {
            Some(ContiguousIntegerSet::new(min(self.start, other.start), max(self.end, other.end)))
        }
    }
}

impl<E: Integer + Copy> Interval for ContiguousIntegerSet<E> {
    type Element = E;

    fn get_start(&self) -> E {
        self.start
    }

    fn set_start(&mut self, start: E) {
        self.start = start;
    }

    fn get_end(&self) -> E {
        self.end
    }

    fn set_end(&mut self, end: E) {
        self.end = end;
    }
}

pub struct IntegerSetCollector<E: Integer + Copy> {
    intervals: Vec<ContiguousIntegerSet<E>>
}

impl<E: Integer + NumAssignOps + Copy + fmt::Display> IntegerSetCollector<E> {
    pub fn new() -> IntegerSetCollector<E> {
        IntegerSetCollector {
            intervals: Vec::<ContiguousIntegerSet<E>>::new(),
        }
    }

    pub fn get_intervals_ref(&self) -> &Vec<ContiguousIntegerSet<E>> {
        &self.intervals
    }

    pub fn to_intervals(&self) -> Vec<ContiguousIntegerSet<E>> {
        self.intervals.clone()
    }

    pub fn into_intervals(self) -> Vec<ContiguousIntegerSet<E>> {
        self.intervals
    }

    pub fn to_integer_set(&self) -> IntegerSet<E> {
        IntegerSet::from(self.intervals.clone())
    }

    pub fn into_integer_set(self) -> IntegerSet<E> {
        IntegerSet::from(self.intervals)
    }

    pub fn append_larger_point(&mut self, point: E) -> Result<(), String> {
        match self.intervals.last_mut() {
            None => {
                self.intervals.push(ContiguousIntegerSet::new(point, point));
            }
            Some(interval) => {
                if point <= interval.end {
                    return Err(format!("The last encountered point {} is larger than the new point {} to be collected", interval.end, point));
                } else if point == interval.end + E::one() {
                    interval.end = point;
                } else {
                    self.intervals.push(ContiguousIntegerSet::new(point, point));
                }
            }
        }
        Ok(())
    }
}

#[derive(PartialEq, Debug)]
pub struct IntegerSet<E: Integer + Copy> {
    pub intervals: Vec<ContiguousIntegerSet<E>>
}

impl<E: Integer + Copy> IntegerSet<E> {
    pub fn new() -> IntegerSet<E> {
        IntegerSet {
            intervals: Vec::new()
        }
    }

    pub fn from(intervals: Vec<ContiguousIntegerSet<E>>) -> IntegerSet<E> {
        IntegerSet {
            intervals
        }
    }

    pub fn get_intervals_by_ref(&self) -> &Vec<ContiguousIntegerSet<E>> {
        &self.intervals
    }

    pub fn into_intervals(self) -> Vec<ContiguousIntegerSet<E>> {
        self.intervals
    }
}

#[cfg(test)]
mod tests {
    use crate::interval::traits::Coalesce;

    use super::{ContiguousIntegerSet, IntegerSetCollector};

    #[test]
    fn test_append_larger_point() {
        let mut collector = IntegerSetCollector::new();
        collector.append_larger_point(1).unwrap();
        collector.append_larger_point(4).unwrap();
        collector.append_larger_point(5).unwrap();
        collector.append_larger_point(7).unwrap();
        collector.append_larger_point(8).unwrap();
        collector.append_larger_point(9).unwrap();
        assert_eq!(collector.into_intervals(), vec![
            ContiguousIntegerSet::new(1, 1),
            ContiguousIntegerSet::new(4, 5),
            ContiguousIntegerSet::new(7, 9)
        ]);
    }

    #[test]
    fn test_coalesce_with() {
        fn test(a: i32, b: i32, c: i32, d: i32, expected: Option<ContiguousIntegerSet<i32>>) {
            let i1 = ContiguousIntegerSet::new(a, b);
            let i2 = ContiguousIntegerSet::new(c, d);
            let m1 = i1.coalesce_with(&i2);
            let m2 = i2.coalesce_with(&i1);
            assert_eq!(m1, m2);
            assert_eq!(m1, expected);
        }
        test(1, 3, 4, 5, Some(ContiguousIntegerSet::new(1, 5)));
        test(2, 3, 0, 5, Some(ContiguousIntegerSet::new(0, 5)));
        test(2, 5, 1, 3, Some(ContiguousIntegerSet::new(1, 5)));
        test(-3, -1, -1, 2, Some(ContiguousIntegerSet::new(-3, 2)));
        test(3, 5, 7, 9, None);
    }
}
