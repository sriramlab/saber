use crate::set::ordered_integer_set::OrderedIntegerSet;

#[derive(Clone, PartialEq, Debug)]
pub struct IntegerPartition {
    pub partitions: Vec<OrderedIntegerSet<usize>>
}

impl IntegerPartition {
    pub fn new(partitions: Vec<OrderedIntegerSet<usize>>) -> IntegerPartition {
        IntegerPartition {
            partitions
        }
    }

    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    pub fn iter(&self) -> IntegerPartitionIter {
        IntegerPartitionIter {
            iter: self.partitions.iter()
        }
    }
}

pub struct IntegerPartitionIter<'a> {
    iter: std::slice::Iter<'a, OrderedIntegerSet<usize>>,
}

impl<'a> Iterator for IntegerPartitionIter<'a> {
    type Item = &'a OrderedIntegerSet<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
