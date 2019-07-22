use std::cmp::{max, min};
use std::fmt;

use num::integer::Integer;
use num::Num;
use num::traits::NumAssignOps;

pub mod trait_impl;

pub trait MergeIntervals<I: Interval<Element=E>, E: Num + Copy> {
    fn sort_and_coalesce_intervals(&self) -> Vec<I>;
    fn sort_and_coalesce_intervals_inplace(&mut self);
}

#[derive(Copy, Clone, PartialEq)]
pub enum IntervalShape {
    Open,
    Closed,
    LeftOpenRightClosed,
    LeftClosedRightOpen,
}

pub trait Interval {
    type Element: Num + Copy;

    fn get_start(&self) -> Self::Element;

    fn set_start(&mut self, start: Self::Element);

    fn get_end(&self) -> Self::Element;

    fn set_end(&mut self, end: Self::Element);

    fn get_shape(&self) -> IntervalShape;

    fn len(&self) -> Self::Element {
        self.get_end() - self.get_start()
    }
}

impl<T: Num + Copy> PartialEq for dyn Interval<Element=T> {
    fn eq(&self, other: &Self) -> bool {
        self.get_start() == other.get_start() && self.get_end() == other.get_end() && self.get_shape() == other.get_shape()
    }
}

impl<T: Num + Copy + fmt::Display> fmt::Display for dyn Interval<Element=T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.get_shape() {
            IntervalShape::Open => write!(f, "({}, {})", self.get_start(), self.get_end()),
            IntervalShape::Closed => write!(f, "[{}, {}]", self.get_start(), self.get_end()),
            IntervalShape::LeftOpenRightClosed => write!(f, "({}, {}]", self.get_start(), self.get_end()),
            IntervalShape::LeftClosedRightOpen => write!(f, "[{}, {})", self.get_start(), self.get_end())
        }
    }
}

impl<T: Num + Copy + fmt::Display> fmt::Debug for dyn Interval<Element=T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct IntervalOverIntegers<T: Integer + Copy> {
    start: T,
    end: T,
}

impl<E: Integer + Copy> IntervalOverIntegers<E> {
    pub fn new(start: E, end: E) -> IntervalOverIntegers<E> {
        IntervalOverIntegers {
            start,
            end,
        }
    }

    pub fn coalesce_with(&self, other: &IntervalOverIntegers<E>) -> Option<IntervalOverIntegers<E>> {
        if self.start > other.end + E::one() || self.end + E::one() < other.start {
            None
        } else {
            Some(IntervalOverIntegers::new(min(self.start, other.start), max(self.end, other.end)))
        }
    }
}

impl<E: Integer + Copy> Interval for IntervalOverIntegers<E> {
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

    fn get_shape(&self) -> IntervalShape {
        IntervalShape::Closed
    }
}

impl<E: Integer + Copy + fmt::Display + 'static> fmt::Display for IntervalOverIntegers<E>  {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self as &dyn Interval<Element=E>)
    }
}

impl<E: Integer + Copy + fmt::Display + 'static> fmt::Debug for IntervalOverIntegers<E>  {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self as &dyn Interval<Element=E>)
    }
}

pub struct ClosedIntegerIntervalCollector<E: Integer + Copy> {
    intervals: Vec<IntervalOverIntegers<E>>
}

impl<E: Integer + NumAssignOps + Copy + fmt::Display> ClosedIntegerIntervalCollector<E> {
    pub fn new() -> ClosedIntegerIntervalCollector<E> {
        ClosedIntegerIntervalCollector {
            intervals: Vec::<IntervalOverIntegers<E>>::new(),
        }
    }

    pub fn get_intervals_ref(&self) -> &Vec<IntervalOverIntegers<E>> {
        &self.intervals
    }

    pub fn get_intervals(self) -> Vec<IntervalOverIntegers<E>> {
        self.intervals
    }

    pub fn append_larger_point(&mut self, point: E) -> Result<(), String> {
        match self.intervals.last_mut() {
            None => {
                self.intervals.push(IntervalOverIntegers::new(point, point));
            }
            Some(interval) => {
                if interval.end > point {
                    return Err(format!("The last encountered point {} is larger than the new point {} to be collected", interval.end, point));
                } else if interval.end + E::one() == point {
                    interval.end = point;
                } else {
                    self.intervals.push(IntervalOverIntegers::new(point, point));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ClosedIntegerIntervalCollector, IntervalOverIntegers};

    #[test]
    fn test_append_larger_point() {
        let mut collector = ClosedIntegerIntervalCollector::new();
        collector.append_larger_point(1).unwrap();
        collector.append_larger_point(4).unwrap();
        collector.append_larger_point(5).unwrap();
        collector.append_larger_point(7).unwrap();
        collector.append_larger_point(8).unwrap();
        collector.append_larger_point(9).unwrap();
        assert_eq!(collector.get_intervals(), vec![
            IntervalOverIntegers::new(1, 1),
            IntervalOverIntegers::new(4, 5),
            IntervalOverIntegers::new(7, 9)
        ]);
    }

    #[test]
    fn test_coalesce_with() {
        fn test(a: i32, b: i32, c: i32, d: i32, expected: Option<IntervalOverIntegers<i32>>) {
            let i1 = IntervalOverIntegers::new(a, b);
            let i2 = IntervalOverIntegers::new(c, d);
            let m1 = i1.coalesce_with(&i2);
            let m2 = i2.coalesce_with(&i1);
            assert_eq!(m1, m2);
            assert_eq!(m1, expected);
        }
        test(1, 3, 4, 5, Some(IntervalOverIntegers::new(1, 5)));
        test(2, 3, 0, 5, Some(IntervalOverIntegers::new(0, 5)));
        test(2, 5, 1, 3, Some(IntervalOverIntegers::new(1, 5)));
        test(-3, -1, -1, 2, Some(IntervalOverIntegers::new(-3, 2)));
        test(3, 5, 7, 9, None);
    }
}
