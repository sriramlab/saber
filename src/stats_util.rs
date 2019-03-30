use std::ops::Deref;

use num_traits::ToPrimitive;

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

pub fn sum_of_squares<'a, A, T: Iterator<Item=&'a A>>(element_iterator: T) -> f64
    where A: Copy + ToPrimitive + 'a, &'a A: Deref {
    kahan_sigma(element_iterator, Box::new(|a| {
        let a_f64 = a.to_f64().unwrap();
        a_f64 * a_f64
    }))
}

pub fn sum<'a, A, T: Iterator<Item=&'a A>>(element_iterator: T) -> f64
    where A: Copy + ToPrimitive + 'a, &'a A: Deref {
    kahan_sigma(element_iterator, Box::new(|a| a.to_f64().unwrap()))
}

pub fn mean<'a, T: Iterator<Item=&'a A>, A: ToPrimitive + 'a>(element_iterator: T) -> f64
    where &'a A: Deref {
    let mut count = 0usize;
    let mut sum = 0f64;
    let mut lower_bits = 0f64;
    for a in element_iterator {
        count += 1;
        let y = (*a).to_f64().unwrap() - lower_bits;
        let new_sum = sum + y;
        lower_bits = (new_sum - sum) - y;
        sum = new_sum;
    }
    sum / count as f64
}

#[cfg(test)]
mod tests {
    extern crate rand;

    use rand::Rng;

    use super::{mean, sum, sum_of_squares};

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
    fn test_sum() {
        let elements = vec![1, 5, 3, 2, 7, 100, 1234, 234, 12, 0, 1234];
        assert_eq!(elements.iter().sum::<i32>() as f64, sum(elements.iter()));
    }

    #[test]
    fn test_sum_of_squares() {
        let elements = vec![1, 5, 3, 2, 7, 100];
        assert_eq!(10088f64, sum_of_squares(elements.iter()));
    }
}
