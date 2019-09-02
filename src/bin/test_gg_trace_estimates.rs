use std::fmt;

use clap::{Arg, clap_app};
use ndarray::{Array, Axis, Ix2, s};
use ndarray_rand::RandomExt;
use num_traits::Float;
use rand::distributions::Uniform;

use saber::matrix_ops::get_gxg_dot_semi_kronecker_z_from_gz_and_ssq;
use saber::program_flow::OrExit;
use saber::simulation::sim_geno::get_gxg_arr;
use saber::trace_estimator::{estimate_gxg_dot_y_norm_sq, estimate_gxg_gram_trace, estimate_gxg_kk_trace};
use saber::util::{extract_numeric_arg, extract_optional_numeric_arg};
use saber::util::matrix_util::generate_plus_minus_one_bernoulli_matrix;
use saber::util::stats_util::{mean, n_choose_2, standard_deviation, sum_of_squares_f32};
use saber::util::timer::Timer;

#[inline]
fn get_error_ratio<T: Float>(estimated_value: T, true_value: T) -> T {
    (estimated_value - true_value) / true_value
}

struct ValueTracker<T> {
    pub values: Vec<T>
}

impl<T: Float> ValueTracker<T> {
    pub fn new() -> ValueTracker<T> {
        ValueTracker {
            values: Vec::<T>::new()
        }
    }

    #[inline]
    pub fn append(&mut self, value: T) {
        self.values.push(value);
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        mean(self.values.iter())
    }

    #[inline]
    pub fn std(&self) -> f64 {
        standard_deviation(self.values.iter(), 0)
    }

    #[inline]
    pub fn abs_mean(&self) -> f64 {
        mean(self.values.iter().map(|x| x.abs()).collect::<Vec<T>>().iter())
    }

    #[inline]
    pub fn abs_std(&self) -> f64 {
        standard_deviation(self.values.iter().map(|x| x.abs()).collect::<Vec<T>>().iter(), 0)
    }

    #[inline]
    pub fn to_percent_string(&self, sig_fig: usize) -> String {
        format!("mean: {:.*}%\nstd: {:.*}%\nabs_mean: {:.*}%\nabs_std: {:.*}%",
                sig_fig, self.mean() * 100., sig_fig, self.std() * 100.,
                sig_fig, self.abs_mean() * 100., sig_fig, self.abs_std() * 100.)
    }
}

impl<T: Float> fmt::Display for ValueTracker<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "mean: {}\nstd: {}\nabs_mean: {}\nabs_std: {}",
               self.mean(), self.std(), self.abs_mean(), self.abs_std())
    }
}

fn update_tracker_and_print<T: Float + fmt::Display>(real: T, estimate: T, tracker: &mut ValueTracker<T>, name: &str, sig_fig: usize) {
    let err_ratio = get_error_ratio(estimate, real);
    tracker.append(err_ratio);
    println!("\niter: {} {}: {}\nerror ratio: {:.*}%", tracker.values.len(), name, estimate, sig_fig, err_ratio * T::from(100.).unwrap());
}

fn main() {
    let mut app = clap_app!(test_gg_trace_estimates =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg num_rows: -r <NUM_ROWS> "Number of rows, i.e. individuals; required")
        (@arg num_cols: -c <NUM_COLUMNS> "Number of columns, i.e. SNPs; required")
        (@arg num_random_vecs: -n <NUM_RAND_VECS> "Number of random vectors; required")
        (@arg num_iter: -i [NUM_ITER] "Number of iterations to run")
    );
    app = app.arg(
        Arg::with_name("num_tr_kk_iter")
            .long("num-kk-iter").short("-k").takes_value(true).value_name("NUM_TR_KK_ITER")
            .help("Number of iterations to run for tr(KK), being kept separate from -i as this is more time consuming")
    );
    let matches = app.get_matches();

    let num_rows = extract_numeric_arg::<usize>(&matches, "num_rows")
        .unwrap_or_exit(Some("failed to parse num_rows"));

    let num_cols = extract_numeric_arg::<usize>(&matches, "num_cols")
        .unwrap_or_exit(Some("failed to parse num_cols"));

    let num_random_vecs = extract_numeric_arg::<usize>(&matches, "num_random_vecs")
        .unwrap_or_exit(Some("failed to parse num_random_vecs"));

    let num_iter = extract_optional_numeric_arg(&matches, "num_iter")
        .unwrap_or_exit(Some("failed to parse num_iter"))
        .unwrap_or(100);

    let num_tr_kk_iter = extract_optional_numeric_arg(&matches, "num_tr_kk_iter")
        .unwrap_or_exit(Some("failed to parse num_tr_kk_iter"))
        .unwrap_or(10);

    println!("num_rows: {}\nnum_cols: {}\nnum_random_vecs: {}\nnum_iter: {}\nnum_tr_kk_iter: {}",
             num_rows, num_cols, num_random_vecs, num_iter, num_tr_kk_iter);

    let gxg_basis = Array::random((num_rows, num_cols), Uniform::from(0..3))
        .mapv(|e| e as f32);

    println!("\n=> generating the GxG matrix");
    let gxg = get_gxg_arr(&gxg_basis);
    println!("GxG dim: {:?}", gxg.dim());

    let gxg_basis2 = Array::random((num_rows, num_cols), Uniform::from(0..3))
        .mapv(|e| e as f32);
    let gxg2 = get_gxg_arr(&gxg_basis2);

    let mut timer = Timer::new();
    let percent_sig_fig = 3usize;

    /*
    {
        println!("\n=> test estimate_gxg_dot_y_norm_sq");
        let y = Array::random(num_rows, Normal::new(0., 1.))
            .mapv(|e| e as f32);
        let mut gxg_dot_y = gxg.t().dot(&y);
        gxg_dot_y.par_iter_mut().for_each(|x| *x = (*x) * (*x));
        let gxg_dot_y_norm_sq = gxg_dot_y.sum() as f64;
        println!("gxg_dot_y_norm_sq true: {}", gxg_dot_y_norm_sq);

        let mut err_tracker = ValueTracker::new();
        for _ in 0..num_iter {
            let gxg_dot_y_norm_sq_est = estimate_gxg_dot_y_norm_sq(&gxg_basis, &y, num_random_vecs);
            update_tracker_and_print(gxg_dot_y_norm_sq, gxg_dot_y_norm_sq_est, &mut err_tracker, "gxg_dot_y_norm_sq_est", percent_sig_fig);
        }
        println!("\ngxg_dot_y_norm_sq error ratio stats:\n{}", err_tracker.to_percent_string(percent_sig_fig));
    }

    /*
    {
        println!("\n=> test estimate_tr_k_gxg_k");
        let rand_geno = Array::random((num_rows, num_cols), Uniform::from(4..7))
            .mapv(|e| e as f32);

        let k_gxg_k = rand_geno.t().dot(&gxg);
        let tr_k_gxg_k_true = sum_of_squares_f32(k_gxg_k.iter()) as f64 / (gxg.dim().1 * rand_geno.dim().1) as f64;
        println!("tr_k_gxg_k_true: {}", tr_k_gxg_k_true);

        let mut err_tracker = ValueTracker::new();
        for _ in 0..num_iter {
            let tr_k_gxg_k_est = estimate_tr_k_gxg_k(&rand_geno, &gxg_basis, num_random_vecs, None);
            update_tracker_and_print(tr_k_gxg_k_true, tr_k_gxg_k_est, &mut err_tracker, "tr_k_gxg_k_est", percent_sig_fig);
        }
        println!("\ntr_k_gxg_k error ratio stats:\n{}", err_tracker.to_percent_string(percent_sig_fig));
    }
    */

    timer.print();

    {
        println!("\n=> calculating tr_k_true");
        let tr_k_true = sum_of_squares_f32(gxg.iter()) as f64;
        println!("tr_k_true: {}", tr_k_true);
        timer.print();
        let mut err_tracker = ValueTracker::new();
        println!("\n=> estimating the trace of GxG.dot(GxG.T)");
        for _ in 0..num_iter {
            let tr_k_est = estimate_gxg_gram_trace(&gxg_basis, num_random_vecs).unwrap_or_exit(None::<String>);
            update_tracker_and_print(tr_k_true, tr_k_est, &mut err_tracker, "tr_k_est", percent_sig_fig);
        }
        println!("\ntr(K) estimate error ratio stats:\n{}", err_tracker.to_percent_string(percent_sig_fig));
    }
    timer.print();

*/
    {
        println!("\n=> calculating tr_kk_true");
        let k = gxg.dot(&gxg.t()) / gxg.dim().1 as f32;
        println!("k dim: {:?}", k.dim());
        let tr_kk_true = sum_of_squares_f32(k.iter()) as f64;
        println!("tr_kk_true: {}", tr_kk_true);
        timer.print();
        let mut err_tracker = ValueTracker::new();
        println!("\n=> computing tr_kk_est");
        for _ in 0..num_tr_kk_iter {
//            let tr_kk_est = estimate_gxg_kk_trace(&gxg_basis, num_random_vecs).unwrap_or_exit(None::<String>);
            let tr_kk_est = test_double_vec_tr_gxg_kk_est(&gxg_basis, num_random_vecs);
            println!("true: {} tr_kk_est: {}", tr_kk_true, tr_kk_est);
            update_tracker_and_print(tr_kk_true, tr_kk_est, &mut err_tracker, "tr_kk_est", percent_sig_fig);
        }
        println!("\ntr(KK) estimate error ratio stats:\n{}", err_tracker.to_percent_string(percent_sig_fig));
    }
    timer.print();

    {
        println!("\n=> calculating tr_ki_kj_true");
        let ki = gxg.dot(&gxg.t()) / gxg.dim().1 as f32;
        let kj = gxg2.dot(&gxg2.t()) / gxg2.dim().1 as f32;
        let mut tr = 0.;
        for i in 0..gxg.dim().0 {
            tr += ki.slice(s![i, ..]).dot(&kj.slice(s![.., i]));
        }
        let tr_ki_kj_true = tr as f64;
        println!("tr_ki_kj_true: {}", tr_ki_kj_true);
        timer.print();
        let mut err_tracker = ValueTracker::new();
        println!("\n=> computing tr_kk_est");
        for _ in 0..num_tr_kk_iter {
            let tr_est = test_double_vec_tr_gxg_ki_gxg_kj_est(&gxg_basis, &gxg_basis2, num_random_vecs, num_random_vecs);
            println!("true: {} tr_kk_est: {}", tr_ki_kj_true, tr_est);
            update_tracker_and_print(tr_ki_kj_true, tr_est, &mut err_tracker, "tr_ki_kj_est", percent_sig_fig);
        }
        println!("\ntr(Ki Kj) estimate error ratio stats:\n{}", err_tracker.to_percent_string(percent_sig_fig));
    }
}

fn test_double_vec_tr_gxg_kk_est(gxg_basis: &Array<f32, Ix2>, num_random_vecs: usize) -> f64 {
    let (num_people, num_snps) = gxg_basis.dim();
    let m = n_choose_2(num_snps);
    let z1 = generate_plus_minus_one_bernoulli_matrix(num_snps, num_random_vecs);
    let z2 = generate_plus_minus_one_bernoulli_matrix(num_snps, num_random_vecs);
    let ssq: Vec<f32> = gxg_basis.axis_iter(Axis(0)).map(|row| sum_of_squares_f32(row.iter())).collect();
    let ssq = Array::from_shape_vec(num_people, ssq).unwrap();
    let gz1 = gxg_basis.dot(&z1);
    let gz2 = gxg_basis.dot(&z2);

    let gz1 = get_gxg_dot_semi_kronecker_z_from_gz_and_ssq(gz1, &ssq);
    let gz2 = get_gxg_dot_semi_kronecker_z_from_gz_and_ssq(gz2, &ssq);
    sum_of_squares_f32(gz1.t().dot(&gz2).iter()) as f64 / (m * m * num_random_vecs * num_random_vecs) as f64
}

fn test_double_vec_tr_gxg_ki_gxg_kj_est(gxg_basis_1: &Array<f32, Ix2>, gxg_basis_2: &Array<f32, Ix2>,
                                        num_rand_vecs_1: usize, num_rand_vecs_2: usize) -> f64 {
    let (num_people, num_basis_snps_1) = gxg_basis_1.dim();
    let num_basis_snps_2 = gxg_basis_2.dim().1;
    let m1 = n_choose_2(num_basis_snps_1);
    let m2 = n_choose_2(num_basis_snps_2);
    let z1 = generate_plus_minus_one_bernoulli_matrix(num_basis_snps_1, num_rand_vecs_1);
    let z2 = generate_plus_minus_one_bernoulli_matrix(num_basis_snps_2, num_rand_vecs_2);
    let ssq_1: Vec<f32> = gxg_basis_1.axis_iter(Axis(0)).map(|row| sum_of_squares_f32(row.iter())).collect();
    let ssq_2: Vec<f32> = gxg_basis_2.axis_iter(Axis(0)).map(|row| sum_of_squares_f32(row.iter())).collect();
    let ssq_1 = Array::from_shape_vec(num_people, ssq_1).unwrap();
    let ssq_2 = Array::from_shape_vec(num_people, ssq_2).unwrap();
    let gz1 = gxg_basis_1.dot(&z1);
    let gz2 = gxg_basis_2.dot(&z2);
    let gz1 = get_gxg_dot_semi_kronecker_z_from_gz_and_ssq(gz1, &ssq_1);
    let gz2 = get_gxg_dot_semi_kronecker_z_from_gz_and_ssq(gz2, &ssq_2);
    sum_of_squares_f32(gz1.t().dot(&gz2).iter()) as f64 / (m1 * m2 * num_rand_vecs_1 * num_rand_vecs_2) as f64
}
