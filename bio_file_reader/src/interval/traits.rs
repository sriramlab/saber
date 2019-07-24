use num::Num;

pub trait Interval {
    type Element: Num + Copy;

    fn get_start(&self) -> Self::Element;

    fn get_end(&self) -> Self::Element;

    fn len(&self) -> Self::Element {
        self.get_end() - self.get_start()
    }
}

pub trait Coalesce: std::marker::Sized {
    fn coalesce_with(&self, other: &Self) -> Option<Self>;
}

pub trait CoalesceIntervals<I: Interval<Element=E>, E: Num + Copy>: std::marker::Sized {
    fn to_coalesced_intervals(&self) -> Vec<I>;

    fn coalesce_intervals_inplace(&mut self);

    fn into_coalesced(mut self) -> Self {
        self.coalesce_intervals_inplace();
        self
    }
}

pub trait Topology {
    fn is_open(&self) -> bool;
    fn is_closed(&self) -> bool;
}

/// returns None if the number of elements is infinite
pub trait Countable {
    fn cardinality(&self) -> Option<usize>;

    fn is_finite(&self) -> bool {
        self.cardinality().is_some()
    }
}
