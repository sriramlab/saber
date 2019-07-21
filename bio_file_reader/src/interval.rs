use std::fmt;

use num::integer::Integer;
use num::Num;
use num::traits::NumAssignOps;

#[derive(PartialEq)]
pub enum IntervalShape {
    Open,
    Closed,
    LeftOpenRightClosed,
    LeftClosedRightOpen,
}

/// `end` is exclusive
pub struct Interval<T> {
    start: T,
    end: T,
    shape: IntervalShape,
}

impl<T: fmt::Display> fmt::Display for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.shape {
            IntervalShape::Open => write!(f, "({}, {})", self.start, self.end),
            IntervalShape::Closed => write!(f, "[{}, {}]", self.start, self.end),
            IntervalShape::LeftOpenRightClosed => write!(f, "({}, {}]", self.start, self.end),
            IntervalShape::LeftClosedRightOpen => write!(f, "[{}, {})", self.start, self.end)
        }
    }
}

impl<T: fmt::Display> fmt::Debug for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<T: Num> Interval<T> {
    pub fn new(start: T, end: T, shape: IntervalShape) -> Interval<T> {
        Interval {
            start,
            end,
            shape,
        }
    }
}

impl<T: Num> PartialEq for Interval<T> {
    fn eq(&self, other: &Interval<T>) -> bool {
        self.start == other.start && self.end == other.end && self.shape == other.shape
    }
}

pub struct ClosedIntervalCollector<T: Num> {
    intervals: Vec<Interval<T>>,
}

impl<T: Integer + NumAssignOps + Copy + fmt::Display> ClosedIntervalCollector<T> {
    pub fn new() -> ClosedIntervalCollector<T> {
        ClosedIntervalCollector {
            intervals: Vec::<Interval<T>>::new(),
        }
    }

    pub fn get_intervals_ref(&self) -> &Vec<Interval<T>> {
        &self.intervals
    }

    pub fn get_intervals(self) -> Vec<Interval<T>> {
        self.intervals
    }

    pub fn append_larger_point(&mut self, point: T) -> Result<(), String> {
        match self.intervals.last_mut() {
            None => {
                self.intervals.push(Interval::new(point, point, IntervalShape::Closed));
            }
            Some(interval) => {
                if interval.end > point {
                    return Err(format!("The last encountered point {} is larger than the new point {} to be collected", interval.end, point));
                } else if interval.end + T::one() == point {
                    interval.end = point;
                } else {
                    self.intervals.push(Interval::new(point, point, IntervalShape::Closed));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ClosedIntervalCollector, Interval, IntervalShape};

    #[test]
    fn test_append_larger_point() {
        let mut collector = ClosedIntervalCollector::new();
        collector.append_larger_point(1).unwrap();
        collector.append_larger_point(4).unwrap();
        collector.append_larger_point(5).unwrap();
        collector.append_larger_point(7).unwrap();
        collector.append_larger_point(8).unwrap();
        collector.append_larger_point(9).unwrap();
        assert_eq!(collector.get_intervals(), vec![
            Interval::new(1, 1, IntervalShape::Closed),
            Interval::new(4, 5, IntervalShape::Closed),
            Interval::new(7, 9, IntervalShape::Closed)
        ]);
    }
}
