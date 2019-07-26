use std::cmp::{max, min};
use std::iter::Sum;
use std::ops::{Sub, SubAssign};

use num::integer::Integer;
use num::traits::cast::ToPrimitive;

use crate::interval::traits::{Coalesce, CoalesceIntervals, Interval};
use crate::traits::{Collecting, Constructable, ToIterator};
use crate::sample::Sample;
use crate::set::Set;
use crate::set::traits::Finite;

/// represents the set of integers in [start, end]
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

impl<E: Integer + Copy> ToIterator<'_, ContiguousIntegerSetIter<E>, E> for ContiguousIntegerSet<E> {
    fn to_iter(&self) -> ContiguousIntegerSetIter<E> {
        ContiguousIntegerSetIter::from(*self)
    }
}

impl<E: Integer + Copy + ToPrimitive> Sample<'_, ContiguousIntegerSetIter<E>, E, OrderedIntegerSet<E>> for ContiguousIntegerSet<E> {}

pub struct ContiguousIntegerSetIter<E: Integer + Copy> {
    contiguous_integer_set: ContiguousIntegerSet<E>,
    current: E,
}

impl<E: Integer + Copy> From<ContiguousIntegerSet<E>> for ContiguousIntegerSetIter<E> {
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
pub struct OrderedIntegerSet<E: Integer + Copy + ToPrimitive> {
    intervals: Vec<ContiguousIntegerSet<E>>
}

impl<E: Integer + Copy + ToPrimitive> OrderedIntegerSet<E> {
    pub fn new() -> OrderedIntegerSet<E> {
        OrderedIntegerSet {
            intervals: Vec::new()
        }
    }

    pub fn from_slice(slice: &[[E; 2]]) -> OrderedIntegerSet<E> {
        let intervals = slice.iter()
                             .map(|pair| ContiguousIntegerSet::new(pair[0], pair[1]))
                             .collect();
        OrderedIntegerSet {
            intervals
        }.into_coalesced()
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

impl<E: Integer + Copy + Sum + ToPrimitive> Finite for OrderedIntegerSet<E> {
    fn size(&self) -> usize {
        self.intervals.iter().map(|&i| i.size()).sum()
    }
}

impl<E: Integer + Copy + ToPrimitive> From<Vec<ContiguousIntegerSet<E>>> for OrderedIntegerSet<E> {
    fn from(intervals: Vec<ContiguousIntegerSet<E>>) -> OrderedIntegerSet<E> {
        OrderedIntegerSet {
            intervals
        }.into_coalesced()
    }
}

impl<E: Integer + Copy + ToPrimitive> Set for OrderedIntegerSet<E> {
    fn is_empty(&self) -> bool {
        self.to_non_empty_intervals().intervals.is_empty()
    }
}

impl<E: Integer + Copy + ToPrimitive> CoalesceIntervals<ContiguousIntegerSet<E>, E> for OrderedIntegerSet<E> {
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
    type Output = OrderedIntegerSet<E>;
    fn sub(self, rhs: ContiguousIntegerSet<E>) -> Self::Output {
        let a = self.get_start();
        let b = self.get_end();
        let c = rhs.get_start();
        let d = rhs.get_end();
        if self.is_empty() || rhs.is_empty() {
            return OrderedIntegerSet::from(vec![self]);
        }
        // [a, b] - [c, d]
        let set = OrderedIntegerSet::from(vec![
            ContiguousIntegerSet::new(a, min(b, c - E::one())),
            ContiguousIntegerSet::new(max(d + E::one(), a), b),
        ]);
        set.into_non_empty_intervals()
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub<ContiguousIntegerSet<E>> for OrderedIntegerSet<E> {
    type Output = Self;
    fn sub(self, rhs: ContiguousIntegerSet<E>) -> Self::Output {
        let diff_intervals: Vec<ContiguousIntegerSet<E>> = self.intervals.iter()
                                                               .flat_map(|i| (*i - rhs).intervals)
                                                               .collect();
        let diff = OrderedIntegerSet::from(diff_intervals);
        diff.into_coalesced()
    }
}

impl<E: Integer + Copy + ToPrimitive> SubAssign<ContiguousIntegerSet<E>> for OrderedIntegerSet<E> {
    fn sub_assign(&mut self, rhs: ContiguousIntegerSet<E>) {
        *self = self.to_owned() - rhs
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub<OrderedIntegerSet<E>> for ContiguousIntegerSet<E> {
    type Output = OrderedIntegerSet<E>;
    fn sub(self, rhs: OrderedIntegerSet<E>) -> Self::Output {
        let mut diff = OrderedIntegerSet::from(vec![self]);
        for interval in rhs.into_intervals().into_iter() {
            diff -= interval;
        }
        diff.into_coalesced()
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub for OrderedIntegerSet<E> {
    type Output = Self;
    fn sub(self, rhs: OrderedIntegerSet<E>) -> Self::Output {
        let mut diff = self;
        for interval in rhs.into_intervals().into_iter() {
            diff -= interval;
        }
        diff.into_coalesced()
    }
}

impl<E: Integer + Copy + ToPrimitive> SubAssign for OrderedIntegerSet<E> {
    fn sub_assign(&mut self, rhs: OrderedIntegerSet<E>) {
        *self = self.to_owned() - rhs
    }
}

impl<E: Integer + Copy + ToPrimitive> Constructable for OrderedIntegerSet<E> {
    fn new() -> OrderedIntegerSet<E> {
        OrderedIntegerSet::new()
    }
}

impl<E: Integer + Copy + ToPrimitive> Collecting<E> for OrderedIntegerSet<E> {
    fn collect(&mut self, item: E) {
        enum CollectResult<E: Integer + Copy + ToPrimitive> {
            Collected,
            // collected the item but need to replace the i-th interval with
            // the first field of CollectedPendingReplaceAndRemoveNext,
            // where i is the second field of CollectedPendingReplaceAndRemoveNext
            CollectedPendingReplaceAndRemoveNext { replace_with: ContiguousIntegerSet<E>, replace_at_index: usize },
            NotCollected,
        }
        // optimize for the special case where the item is
        // to the right of or coalesceable with the last interval
        if let Some(last_interval) = self.intervals.last_mut() {
            if item > last_interval.end + E::one() {
                self.intervals.push(ContiguousIntegerSet::new(item, item));
                return;
            } else if let Some(interval) = last_interval.coalesce_with(&item) {
                *last_interval = interval;
                return;
            }
        }
        let mut search_result = CollectResult::NotCollected;
        for (i, interval) in self.intervals.iter_mut().enumerate() {
            if item + E::one() < interval.start {
                self.intervals.insert(i, ContiguousIntegerSet::new(item, item));
                return;
            }
            match interval.coalesce_with(&item) {
                Some(new_interval) => {
                    search_result = CollectResult::Collected;
                    *interval = new_interval;
                    if let Some(next_interval) = self.intervals.get(i + 1) {
                        if let Some(merged_interval) = new_interval.coalesce_with(next_interval) {
                            search_result = CollectResult::CollectedPendingReplaceAndRemoveNext {
                                replace_with: merged_interval,
                                replace_at_index: i,
                            };
                        }
                    }
                    break;
                }
                None => {}
            };
        }
        match search_result {
            CollectResult::Collected => {}
            CollectResult::CollectedPendingReplaceAndRemoveNext {
                replace_with: merged_interval,
                replace_at_index: i
            } => {
                self.intervals[i] = merged_interval;
                self.intervals.remove(i + 1);
            }
            CollectResult::NotCollected => self.intervals.push(ContiguousIntegerSet::new(item, item))
        }
    }
}

pub struct IntegerSetIter<E: Integer + Copy + ToPrimitive> {
    ordered_integer_set: OrderedIntegerSet<E>,
    current_interval_index: usize,
    current_element_index: E,
}

impl<E: Integer + Copy + ToPrimitive> From<OrderedIntegerSet<E>> for IntegerSetIter<E> {
    fn from(ordered_integer_set: OrderedIntegerSet<E>) -> IntegerSetIter<E> {
        IntegerSetIter {
            ordered_integer_set,
            current_interval_index: 0,
            current_element_index: E::zero(),
        }
    }
}

impl<E: Integer + Copy + ToPrimitive> Iterator for IntegerSetIter<E> {
    type Item = E;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_interval_index >= self.ordered_integer_set.intervals.len() {
            None
        } else {
            let interval = &self.ordered_integer_set.intervals[self.current_interval_index];
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

impl<E: Integer + Copy + ToPrimitive> ToIterator<'_, IntegerSetIter<E>, E> for OrderedIntegerSet<E> {
    fn to_iter(&self) -> IntegerSetIter<E> {
        IntegerSetIter::from(self.clone())
    }
}

impl<E: Integer + Copy + ToPrimitive + Sum> Sample<'_, IntegerSetIter<E>, E, OrderedIntegerSet<E>> for OrderedIntegerSet<E> {}

#[cfg(test)]
mod tests {
    use crate::interval::traits::*;
    use crate::traits::Collecting;

    use super::{ContiguousIntegerSet, OrderedIntegerSet};

    #[test]
    fn test_integer_set_collect() {
        let mut set = OrderedIntegerSet::new();
        set.collect(1);
        set.collect(4);
        set.collect(5);
        set.collect(7);
        set.collect(8);
        set.collect(9);
        assert_eq!(set.into_intervals(), vec![
            ContiguousIntegerSet::new(1, 1),
            ContiguousIntegerSet::new(4, 5),
            ContiguousIntegerSet::new(7, 9)
        ]);

        let mut set = OrderedIntegerSet::from_slice(&[[1, 3], [5, 7], [15, 20]]);
        set.collect(-5);
        set.collect(-1);
        set.collect(0);
        set.collect(-10);
        set.collect(4);
        set.collect(10);
        set.collect(12);
        set.collect(13);
        assert_eq!(set.intervals, vec![
            ContiguousIntegerSet::new(-10, -10),
            ContiguousIntegerSet::new(-5, -5),
            ContiguousIntegerSet::new(-1, 7),
            ContiguousIntegerSet::new(10, 10),
            ContiguousIntegerSet::new(12, 13),
            ContiguousIntegerSet::new(15, 20),
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
            assert_eq!(s1 - s2, OrderedIntegerSet::from_slice(expected));
        }
        test(&[6, 5], &[-1, 3], &[]);
        test(&[6, 5], &[1, 3], &[]);
        test(&[5, 10], &[3, 1], &[[5, 10]]);
        test(&[5, 8], &[-1, 3], &[[5, 8]]);
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
            let diff = OrderedIntegerSet::from_slice(a) - ContiguousIntegerSet::new(b[0], b[1]);
            assert_eq!(diff, OrderedIntegerSet::from_slice(expected));
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
            let diff = ContiguousIntegerSet::new(a[0], a[1]) - OrderedIntegerSet::from_slice(b);
            assert_eq!(diff, OrderedIntegerSet::from_slice(expected));
        }
        test(&[1, 12], &[], &[[1, 12]]);
        test(&[1, 12], &[[12, 1]], &[[1, 12]]);
        test(&[1, 12], &[[2, 3], [5, 6]], &[[1, 1], [4, 4], [7, 12]]);
        test(&[1, 12], &[[-1, 3], [10, 13]], &[[4, 9]]);
    }

    #[test]
    fn test_sub_integer_set() {
        fn test(a: &[[i32; 2]], b: &[[i32; 2]], expected: &[[i32; 2]]) {
            let mut diff = OrderedIntegerSet::from_slice(a) - OrderedIntegerSet::from_slice(b);
            diff.coalesce_intervals_inplace();
            assert_eq!(diff, OrderedIntegerSet::from_slice(expected));
        }
        test(&[[1, 10]], &[[1, 3], [5, 7]], &[[4, 4], [8, 10]]);
        test(&[[0, 10]], &[[1, 3], [5, 7]], &[[0, 0], [4, 4], [8, 10]]);
        test(&[[0, 10], [15, 20]], &[[-1, 2], [5, 7]], &[[3, 4], [8, 10], [15, 20]]);
        test(&[[0, 10], [15, 20]], &[[-1, 2], [18, 22], [5, 7]], &[[3, 4], [8, 10], [15, 17]]);
        test(&[[0, 10], [15, 20], [-10, -5]], &[[-1, 2], [18, 22], [5, 7], [-12, -3]], &[[3, 4], [8, 10], [15, 17]]);
    }
}
