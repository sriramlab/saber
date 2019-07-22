use num::Num;

pub trait Interval {
    type Element: Num + Copy;

    fn get_start(&self) -> Self::Element;

    fn set_start(&mut self, start: Self::Element);

    fn get_end(&self) -> Self::Element;

    fn set_end(&mut self, end: Self::Element);

    fn len(&self) -> Self::Element {
        self.get_end() - self.get_start()
    }
}

pub trait Coalesce: std::marker::Sized {
    fn coalesce_with(&self, other: &Self) -> Option<Self>;
}

pub trait Topology {
    fn is_open(&self) -> bool;
    fn is_closed(&self) -> bool;
}

pub trait IntervalTopology: Topology {
    fn get_topology(&self) -> IntervalShape;
}

pub trait CoalesceIntervals<I: Interval<Element=E>, E: Num + Copy> {
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
