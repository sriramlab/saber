use crate::set::ordered_integer_set::OrderedIntegerSet;

pub type Partition = OrderedIntegerSet<usize>;

#[derive(Clone, PartialEq, Debug)]
pub struct IntegerPartitions {
    partitions: Vec<Partition>
}

impl IntegerPartitions {
    pub fn new(partitions: Vec<Partition>) -> IntegerPartitions {
        IntegerPartitions {
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
    iter: std::slice::Iter<'a, Partition>,
}

impl<'a> Iterator for IntegerPartitionIter<'a> {
    type Item = &'a Partition;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
