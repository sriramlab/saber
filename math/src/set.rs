use std::cmp::{max, min};
use std::fmt;
use std::iter::Sum;
use std::ops::{Sub, SubAssign};

use num::integer::Integer;
use num::traits::cast::ToPrimitive;
use num::traits::NumAssignOps;

use crate::interval::traits::{Coalesce, CoalesceIntervals, Interval};
use crate::sample::{Collecting, Constructable, Sample, ToIterator};
use crate::set::traits::Finite;

pub mod traits;

pub trait Set {
    fn is_empty(&self) -> bool;
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct ContiguousIntegerSet<E: Integer + Copy> {
    start: E,
    end: E,
}

impl<E: Integer + Copy> Set for ContiguousIntegerSet<E> {
    fn is_empty(&self) -> bool {
        self.start > self.end
    }
}

impl<E: Integer + Copy> ContiguousIntegerSet<E> {
    pub fn new(start: E, end: E) -> ContiguousIntegerSet<E> {
        ContiguousIntegerSet {
            start,
            end,
        }
    }
}

impl<E: Integer + Copy + ToPrimitive> Finite for ContiguousIntegerSet<E> {
    fn size(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            (self.end - self.start + E::one()).to_usize().unwrap()
        }
    }
}

/// returns an interval if only if the two intervals can be merged into
/// a single non-empty interval
/// An empty interval can be merged with any other non-empty interval
impl<E: Integer + Copy> Coalesce<Self> for ContiguousIntegerSet<E> {
    fn coalesce_with(&self, other: &Self) -> Option<Self> {
        if self.is_empty() && other.is_empty() {
            None
        } else if self.is_empty() {
            Some(*other)
        } else if other.is_empty() {
            Some(*self)
        } else {
            if self.start > other.end + E::one() || self.end + E::one() < other.start {
                None
            } else {
                Some(ContiguousIntegerSet::new(min(self.start, other.start), max(self.end, other.end)))
            }
        }
    }
}

impl<E: Integer + Copy> Coalesce<E> for ContiguousIntegerSet<E> {
    fn coalesce_with(&self, other: &E) -> Option<Self> {
        if self.is_empty() {
            Some(ContiguousIntegerSet::new(*other, *other))
        } else {
            if self.start > *other + E::one() || self.end + E::one() < *other {
                None
            } else {
                Some(ContiguousIntegerSet::new(min(self.start, *other), max(self.end, *other)))
            }
        }
    }
}

impl<E: Integer + Copy> Interval for ContiguousIntegerSet<E> {
    type Element = E;

    fn get_start(&self) -> E {
        self.start
    }

    fn get_end(&self) -> E {
        self.end
    }

    fn length(&self) -> E {
        self.end - self.start
    }
}

impl<E: Integer + Copy> ToIterator<ContiguousIntegerSetIter<E>, E> for ContiguousIntegerSet<E> {
    fn iter(&self) -> ContiguousIntegerSetIter<E> {
        ContiguousIntegerSetIter::from(*self)
    }
}

impl<E: Integer + Copy + ToPrimitive> Sample<ContiguousIntegerSetIter<E>, E> for ContiguousIntegerSet<E> {
    type Output = IntegerSet<E>;
}

pub struct ContiguousIntegerSetIter<E: Integer + Copy> {
    contiguous_integer_set: ContiguousIntegerSet<E>,
    current: E,
}

impl<E: Integer + Copy> ContiguousIntegerSetIter<E> {
    fn from(contiguous_integer_set: ContiguousIntegerSet<E>) -> ContiguousIntegerSetIter<E> {
        ContiguousIntegerSetIter {
            contiguous_integer_set,
            current: E::zero(),
        }
    }
}

impl<E: Integer + Copy> Iterator for ContiguousIntegerSetIter<E> {
    type Item = E;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current > self.contiguous_integer_set.end {
            None
        } else {
            let val = self.current;
            self.current = self.current + E::one();
            Some(val)
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct IntegerSet<E: Integer + Copy + ToPrimitive> {
    pub intervals: Vec<ContiguousIntegerSet<E>>
}

impl<E: Integer + Copy + ToPrimitive> IntegerSet<E> {
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

    pub fn from_slice(slice: &[[E; 2]]) -> IntegerSet<E> {
        let intervals = slice.iter()
                             .map(|pair| ContiguousIntegerSet::new(pair[0], pair[1]))
                             .collect();
        IntegerSet {
            intervals
        }
    }

    pub fn to_non_empty_intervals(&self) -> Self {
        self.clone().into_non_empty_intervals()
    }

    pub fn into_non_empty_intervals(mut self) -> Self {
        self.remove_empty_intervals();
        self
    }

    pub fn remove_empty_intervals(&mut self) {
        self.intervals.drain_filter(|i| i.is_empty());
    }

    pub fn get_intervals_by_ref(&self) -> &Vec<ContiguousIntegerSet<E>> {
        &self.intervals
    }

    pub fn into_intervals(self) -> Vec<ContiguousIntegerSet<E>> {
        self.intervals
    }

    pub fn num_intervals(&self) -> usize {
        self.intervals.len()
    }
}

impl<E: Integer + Copy + Sum + ToPrimitive> IntegerSet<E> {
    pub fn size(&self) -> usize {
        self.intervals.iter().map(|&i| i.size()).sum()
    }
}

impl<E: Integer + Copy + ToPrimitive> Set for IntegerSet<E> {
    fn is_empty(&self) -> bool {
        self.to_non_empty_intervals().intervals.is_empty()
    }
}

impl<E: Integer + Copy + ToPrimitive> Finite for IntegerSet<E> {
    fn size(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            let mut size = 0;
            for interval in self.intervals.iter() {
                size += interval.size();
            }
            size
        }
    }
}

impl<E: Integer + Copy + ToPrimitive> CoalesceIntervals<ContiguousIntegerSet<E>, E> for IntegerSet<E> {
    fn to_coalesced_intervals(&self) -> Vec<ContiguousIntegerSet<E>> {
        let mut intervals = self.to_non_empty_intervals().intervals;
        intervals.coalesce_intervals_inplace();
        intervals
    }

    fn coalesce_intervals_inplace(&mut self) {
        self.remove_empty_intervals();
        self.intervals.coalesce_intervals_inplace();
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub for ContiguousIntegerSet<E> {
    type Output = IntegerSet<E>;
    fn sub(self, rhs: ContiguousIntegerSet<E>) -> Self::Output {
        let a = self.get_start();
        let b = self.get_end();
        let c = rhs.get_start();
        let d = rhs.get_end();
        if self.is_empty() || rhs.is_empty() {
            return IntegerSet::from(vec![self]);
        }
        // [a, b] - [c, d]
        let set = IntegerSet::from(vec![
            ContiguousIntegerSet::new(a, min(b, c - E::one())),
            ContiguousIntegerSet::new(max(d + E::one(), a), b),
        ]);
        set.into_non_empty_intervals()
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub<ContiguousIntegerSet<E>> for IntegerSet<E> {
    type Output = Self;
    fn sub(self, rhs: ContiguousIntegerSet<E>) -> Self::Output {
        let diff_intervals: Vec<ContiguousIntegerSet<E>> = self.intervals.iter()
                                                               .flat_map(|i| (*i - rhs).intervals)
                                                               .collect();
        let diff = IntegerSet::from(diff_intervals);
        diff.into_coalesced()
    }
}

impl<E: Integer + Copy + ToPrimitive> SubAssign<ContiguousIntegerSet<E>> for IntegerSet<E> {
    fn sub_assign(&mut self, rhs: ContiguousIntegerSet<E>) {
        *self = self.to_owned() - rhs
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub<IntegerSet<E>> for ContiguousIntegerSet<E> {
    type Output = IntegerSet<E>;
    fn sub(self, rhs: IntegerSet<E>) -> Self::Output {
        let mut diff = IntegerSet::from(vec![self]);
        for interval in rhs.into_intervals().into_iter() {
            diff -= interval;
        }
        diff.into_coalesced()
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub for IntegerSet<E> {
    type Output = Self;
    fn sub(self, rhs: IntegerSet<E>) -> Self::Output {
        let mut diff = self;
        for interval in rhs.into_intervals().into_iter() {
            diff -= interval;
        }
        diff.into_coalesced()
    }
}

impl<E: Integer + Copy + ToPrimitive> SubAssign for IntegerSet<E> {
    fn sub_assign(&mut self, rhs: IntegerSet<E>) {
        *self = self.to_owned() - rhs
    }
}

impl<E: Integer + Copy + ToPrimitive> Constructable for IntegerSet<E> {
    type Output = IntegerSet<E>;
    fn new() -> Self::Output {
        IntegerSet::new()
    }
}

impl<E: Integer + Copy + ToPrimitive> Collecting<E> for IntegerSet<E> {
    fn collect(&mut self, item: E) {
        for interval in self.intervals.iter_mut() {
            match interval.coalesce_with(&item) {
                Some(i) => {
                    *interval = i;
                    return;
                }
                None => {}
            };
        }
        self.intervals.push(ContiguousIntegerSet::new(item, item));
    }
}

pub struct IntegerSetIter<E: Integer + Copy + ToPrimitive> {
    integer_set: IntegerSet<E>,
    current_interval_index: usize,
    current_element_index: E,
}

impl<E: Integer + Copy + ToPrimitive> IntegerSetIter<E> {
    fn from(integer_set: IntegerSet<E>) -> IntegerSetIter<E> {
        IntegerSetIter {
            integer_set,
            current_interval_index: 0,
            current_element_index: E::zero(),
        }
    }
}

impl<E: Integer + Copy + ToPrimitive> Iterator for IntegerSetIter<E> {
    type Item = E;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_interval_index >= self.integer_set.intervals.len() {
            None
        } else {
            let interval = &self.integer_set.intervals[self.current_interval_index];
            if self.current_element_index.to_usize().unwrap() >= interval.size() {
                self.current_interval_index += 1;
                self.current_element_index = E::zero();
                self.next()
            } else {
                let val = interval.start + self.current_element_index;
                self.current_element_index = self.current_element_index + E::one();
                Some(val)
            }
        }
    }
}

impl<E: Integer + Copy + ToPrimitive> ToIterator<IntegerSetIter<E>, E> for IntegerSet<E> {
    fn iter(&self) -> IntegerSetIter<E> {
        IntegerSetIter::from(self.clone())
    }
}

impl<E: Integer + Copy + ToPrimitive> Sample<IntegerSetIter<E>, E> for IntegerSet<E> {
    type Output = IntegerSet<E>;
}

pub struct IntegerSetCollector<E: Integer + Copy> {
    intervals: Vec<ContiguousIntegerSet<E>>
}

impl<E: Integer + NumAssignOps + Copy + ToPrimitive + fmt::Display> IntegerSetCollector<E> {
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

#[cfg(test)]
mod tests {
    use crate::interval::traits::*;

    use super::{ContiguousIntegerSet, IntegerSet, IntegerSetCollector};

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
        test(9, 5, 5, 7, Some(ContiguousIntegerSet::new(5, 7)));
    }

    #[test]
    fn test_sub_contiguous_integer_set() {
        fn test(a: &[i32; 2], b: &[i32; 2], expected: &[[i32; 2]]) {
            let s1 = ContiguousIntegerSet::new(a[0], a[1]);
            let s2 = ContiguousIntegerSet::new(b[0], b[1]);
            assert_eq!(s1 - s2, IntegerSet::from_slice(expected));
        }
        test(&[6, 5], &[-1, 3], &[[6, 5]]);
        test(&[2, 10], &[4, 9], &[[2, 3], [10, 10]]);
        test(&[2, 10], &[1, 8], &[[9, 10]]);
        test(&[2, 10], &[6, 8], &[[2, 5], [9, 10]]);
        test(&[2, 10], &[2, 10], &[]);
        test(&[2, 10], &[0, 12], &[]);
        test(&[3, 4], &[3, 4], &[]);
        test(&[3, 5], &[3, 3], &[[4, 5]]);
        test(&[3, 4], &[3, 3], &[[4, 4]]);
        test(&[-2, 5], &[-1, 3], &[[-2, -2], [4, 5]]);
    }

    #[test]
    fn test_integer_set_minus_contiguous_integer_set() {
        fn test(a: &[[i32; 2]], b: &[i32; 2], expected: &[[i32; 2]]) {
            let diff = IntegerSet::from_slice(a) - ContiguousIntegerSet::new(b[0], b[1]);
            assert_eq!(diff, IntegerSet::from_slice(expected));
        }
        test(&[[1, 5], [8, 12], [-4, -2]], &[100, -100], &[[-4, -2], [1, 5], [8, 12]]);
        test(&[[1, 5], [108, 12], [-4, -2]], &[-3, 8], &[[-4, -4]]);
        test(&[[1, 5], [8, 12], [-4, -2]], &[-3, 8], &[[-4, -4], [9, 12]]);
        test(&[[1, 5], [8, 12], [-4, -2]], &[-5, 8], &[[9, 12]]);
        test(&[[1, 5], [8, 12], [-4, -2]], &[-5, -5], &[[-4, -2], [1, 5], [8, 12]]);
        test(&[[1, 5], [8, 12], [-4, -2]], &[-5, 0], &[[1, 5], [8, 12]]);
        test(&[[1, 5], [8, 12]], &[6, 7], &[[1, 5], [8, 12]]);
    }

    #[test]
    fn test_contiguous_integer_set_minus_integer_set() {
        fn test(a: &[i32; 2], b: &[[i32; 2]], expected: &[[i32; 2]]) {
            let diff = ContiguousIntegerSet::new(a[0], a[1]) - IntegerSet::from_slice(b);
            assert_eq!(diff, IntegerSet::from_slice(expected));
        }
        test(&[1, 12], &[], &[[1, 12]]);
        test(&[1, 12], &[[12, 1]], &[[1, 12]]);
        test(&[1, 12], &[[2, 3], [5, 6]], &[[1, 1], [4, 4], [7, 12]]);
        test(&[1, 12], &[[-1, 3], [10, 13]], &[[4, 9]]);
    }

    #[test]
    fn test_sub_integer_set() {
        fn test(a: &[[i32; 2]], b: &[[i32; 2]], expected: &[[i32; 2]]) {
            let mut diff = IntegerSet::from_slice(a) - IntegerSet::from_slice(b);
            diff.coalesce_intervals_inplace();
            assert_eq!(diff, IntegerSet::from_slice(expected));
        }
        test(&[[1, 10]], &[[1, 3], [5, 7]], &[[4, 4], [8, 10]]);
        test(&[[0, 10]], &[[1, 3], [5, 7]], &[[0, 0], [4, 4], [8, 10]]);
        test(&[[0, 10], [15, 20]], &[[-1, 2], [5, 7]], &[[3, 4], [8, 10], [15, 20]]);
        test(&[[0, 10], [15, 20]], &[[-1, 2], [18, 22], [5, 7]], &[[3, 4], [8, 10], [15, 17]]);
        test(&[[0, 10], [15, 20]], &[[-1, 2], [18, 22], [5, 7]], &[[3, 4], [8, 10], [15, 17]]);
        test(&[[0, 10], [15, 20], [-10, -5]], &[[-1, 2], [18, 22], [5, 7], [-12, -3]], &[[3, 4], [8, 10], [15, 17]]);
    }
}
