use std::ops::Deref;

use num_traits::ToPrimitive;

pub fn n_choose_2(n: usize) -> usize {
n * (n - 1) / 2
}

pub fn kahan_sigma<'a, A, T: Iterator<Item=&'a A>>(element_iterator: T, op: Box<dyn Fn(A) -> f64>) -> f64
    where A: Copy + 'a, &'a A: Deref {
    // Kahan summation algorithm
    let mut sum = 0f64;
    let mut lower_bits = 0f64;
    for a in element_iterator {
        let y = op(*a) - lower_bits;
        let new_sum = sum + y;
        lower_bits = (new_sum - sum) - y;
        sum = new_sum;
    }
    sum
}

pub fn kahan_sigma_f32<'a, A, T: Iterator<Item=&'a A>>(element_iterator: T, op: Box<dyn Fn(A) -> f32>) -> f32
    where A: Copy + 'a, &'a A: Deref {
    // Kahan summation algorithm
    let mut sum = 0f32;
    let mut lower_bits = 0f32;
    for a in element_iterator {
        let y = op(*a) - lower_bits;
        let new_sum = sum + y;
        lower_bits = (new_sum - sum) - y;
        sum = new_sum;
    }
    sum
}

pub fn kahan_sigma_return_counter<'a, A, T: Iterator<Item=&'a A>>(element_iterator: T, op: Box<dyn Fn(A) -> f64>) -> (f64, usize)
    where A: Copy + 'a, &'a A: Deref {
    let mut count = 0usize;
    // Kahan summation algorithm
    let mut sum = 0f64;
    let mut lower_bits = 0f64;
    for a in element_iterator {
        count += 1;
        let y = op(*a) - lower_bits;
        let new_sum = sum + y;
        lower_bits = (new_sum - sum) - y;
        sum = new_sum;
    }
    (sum, count)
}

#[inline]
pub fn sum<'a, A, T: Iterator<Item=&'a A>>(element_iterator: T) -> f64
    where A: Copy + ToPrimitive + 'a, &'a A: Deref {
    kahan_sigma(element_iterator, Box::new(|a| a.to_f64().unwrap()))
}

pub fn sum_of_squares<'a, A, T: Iterator<Item=&'a A>>(element_iterator: T) -> f64
    where A: Copy + ToPrimitive + 'a, &'a A: Deref {
    kahan_sigma(element_iterator, Box::new(|a| {
        let a_f64 = a.to_f64().unwrap();
        a_f64 * a_f64
    }))
}

pub fn sum_of_squares_f32<'a, A, T: Iterator<Item=&'a A>>(element_iterator: T) -> f32
    where A: Copy + ToPrimitive + 'a, &'a A: Deref {
    kahan_sigma_f32(element_iterator, Box::new(|a| {
        let a_f32 = a.to_f32().unwrap();
        a_f32 * a_f32
    }))
}

pub fn mean<'a, A, T: Iterator<Item=&'a A>>(element_iterator: T) -> f64
    where A: Copy + ToPrimitive + 'a, &'a A: Deref {
    let (sum, count) = kahan_sigma_return_counter(element_iterator, Box::new(|a| a.to_f64().unwrap()));
    sum / count as f64
}

/// `ddof` stands for delta degress of freedom, and the sum of squares will be divided by `count - ddof`,
/// where `count` is the number of elements
/// for population variance, set `ddof` to 0
/// for sample variance, set `ddof` to 1
pub fn variance<'a, T: Clone + Iterator<Item=&'a A>, A>(element_iterator: T, ddof: usize) -> f64
    where A: Copy + ToPrimitive + 'a, &'a A: Deref {
    let mean = mean(element_iterator.clone());
    let (sum, count) = kahan_sigma_return_counter(element_iterator, Box::new(move |a| {
        let a_f64 = a.to_f64().unwrap() - mean;
        a_f64 * a_f64
    }));
    sum / (count - ddof) as f64
}

/// `ddof` stands for delta degress of freedom, and the sum of squares will be divided by `count - ddof`,
/// where `count` is the number of elements
/// for population standard deviation, set `ddof` to 0
/// for sample standard deviation, set `ddof` to 1
pub fn std<'a, T: Clone + Iterator<Item=&'a A>, A>(element_iterator: T, ddof: usize) -> f64
    where A: Copy + ToPrimitive + 'a, &'a A: Deref {
    variance(element_iterator, ddof).sqrt()
}

#[cfg(test)]
mod tests {
    extern crate rand;

    use rand::Rng;

    use super::{mean, sum, sum_of_squares, variance, std};

    #[test]
    fn test_sum() {
        let elements = vec![1, 5, 3, 2, 7, 100, 1234, 234, 12, 0, 1234];
        assert_eq!(elements.iter().sum::<i32>() as f64, sum(elements.iter()));
    }

    #[test]
    fn test_sum_of_squares() {
        let elements = vec![1, 5, 3, 2, 7, 100];
        assert_eq!(10088f64, sum_of_squares(elements.iter()));
    }

    #[test]
    fn test_mean() {
        let mut numbers = Vec::<i64>::with_capacity(100);
        let mut rng = rand::thread_rng();
        for _ in 0..1000000 {
            numbers.push(rng.gen_range(1, 21));
        }
        assert_eq!(numbers.iter().sum::<i64>() as f64 / numbers.len() as f64, mean(numbers.iter()));
    }

    #[test]
    fn test_variance() {
        let elements = vec![1, 5, 123, 5, -345, 467, 568, 1234, -123, -2343, 23];
        assert_eq!(768950.6, variance(elements.iter(), 1));
        assert_eq!(699046.0, variance(elements.iter(), 0));
    }

    #[test]
    fn test_std() {
        let elements = vec![1, 5, 3, 2, 7, 100, 1234, 234, 12, 0, 1234];
        assert_eq!(487.9947466185192, std(elements.iter(), 1));
        assert_eq!(465.28473464914003, std(elements.iter(), 0));
    }
}
