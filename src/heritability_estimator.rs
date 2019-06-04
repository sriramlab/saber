use ndarray::{Array, Ix1, Ix2};
use ndarray_linalg::Solve;
use crate::timer::Timer;
use crate::matrix_util::{generate_plus_minus_one_bernoulli_matrix, mean_center_vector,
                         normalize_matrix_row_wise_inplace, normalize_matrix_columns_inplace};
use crate::stats_util::{sum_of_squares, std, n_choose_2};
use crate::trace_estimators::{estimate_gxg_kk_trace, estimate_gxg_gram_trace, estimate_gxg_dot_y_norm_sq,
                              estimate_tr_k_gxg_k, estimate_tr_kk, estimate_tr_gxg_i_gxg_j};
use colored::Colorize;

fn bold_print(msg: &String) {
    println!("{}", msg.bold());
}

pub fn estimate_heritability(mut geno_arr: Array<f32, Ix2>, mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize) -> Result<f64, String> {
    println!("\n=> creating the genotype ndarray and starting the timer for profiling");
    let mut timer = Timer::new();
    // geno_arr is num_snps x num_people
    let (num_rows, num_cols) = geno_arr.dim();
    println!("geno_arr dim: {:?}", geno_arr.dim());
    timer.print();

    println!("\n=> normalizing the genotype matrix row-wise");
    geno_arr = normalize_matrix_row_wise_inplace(geno_arr, 1);
    timer.print();

    println!("\n=> mean centering the phenotype vector");
    pheno_arr = mean_center_vector(pheno_arr);
    timer.print();

    println!("\n=> generating random estimators");
    let rand_mat = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    timer.print();

    println!("\n=> MatMul geno_arr{:?} with rand_mat{:?}", geno_arr.dim(), rand_mat.dim());
    let xz_arr = geno_arr.dot(&rand_mat);
    println!("xz_arr: {:?}", xz_arr.dim());
    timer.print();

    println!("\n=> MatMul geno_arr{:?}.T with xz_arr{:?}", geno_arr.dim(), xz_arr.dim());
    let xxz = geno_arr.t().dot(&xz_arr);
    println!("xxz dim: {:?}", xxz.dim());
    timer.print();

    println!("\n=> calculating trace estimate through L2 squared");
    let trace_kk_est = sum_of_squares(xxz.iter()) / (num_rows * num_rows * num_random_vecs) as f64;
    println!("trace_kk_est: {}", trace_kk_est);
    timer.print();

    println!("\n=> calculating Xy");
    let xy = geno_arr.dot(&pheno_arr);
    timer.print();

    println!("\n=> calculating yKy and yy");
    let yky = sum_of_squares(xy.iter()) / num_rows as f64;
    let yy = sum_of_squares(pheno_arr.iter());
    timer.print();

    println!("\n=> solving for heritability");
    let a = array![[trace_kk_est, (num_cols - 1) as f64],[(num_cols - 1) as f64, num_cols as f64]];
    let b = array![yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    println!("sig_sq: {:?}", sig_sq);
    let s_y_sq = yy / (num_cols - 1) as f64;
    let heritability = sig_sq[0] as f64 / s_y_sq;
    println!("heritability: {}  s_y^2: {}", heritability, s_y_sq);

    let standard_error = (2. / (trace_kk_est - num_cols as f64)).sqrt();
    println!("standard error: {}", standard_error);

    Ok(heritability)
}

pub fn estimate_joint_heritability(mut geno_arr: Array<f32, Ix2>, mut le_snps_arr: Array<f32, Ix2>,
    mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize,
) -> Result<(f64, f64, f64), String> {
    let (num_people, num_snps) = geno_arr.dim();
    let num_independent_snps = le_snps_arr.dim().1;
    println!("\n=> estimating heritability due to G and GxG\nnum_people: {}\nnum_snps: {}\nnum_independent_snps: {}",
             num_people, num_snps, num_independent_snps);

    println!("\n=> normalizing the genotype matrices");
    normalize_matrix_columns_inplace(&mut geno_arr, 0);
    normalize_matrix_columns_inplace(&mut le_snps_arr, 0);

    println!("\n=> normalizing the phenotype vector");
    pheno_arr = mean_center_vector(pheno_arr);
    pheno_arr /= std(pheno_arr.iter(), 0) as f32;

    println!("\n=> estimating traces related to the G matrix");
    let num_rand_z = 100usize;
    let tr_kk_est = estimate_tr_kk(&geno_arr, num_rand_z);
    println!("tr_kk_est: {}", tr_kk_est);
    let xy = geno_arr.t().dot(&pheno_arr);
    let yky = sum_of_squares(xy.iter()) / num_snps as f64;
    let yy = sum_of_squares(pheno_arr.iter());

    println!("\n=> estimating traces related to the GxG matrix");
    let mm = n_choose_2(num_independent_snps) as f64;

    let gxg_tr_kk_est = estimate_gxg_kk_trace(&le_snps_arr, num_random_vecs)?;
    let gxg_tr_k_est = estimate_gxg_gram_trace(&le_snps_arr, num_random_vecs)? / mm;

    println!("gxg_tr_k_est: {}", gxg_tr_k_est);
    println!("gxg_tr_kk_est: {}", gxg_tr_kk_est);

    let gxg_yky = estimate_gxg_dot_y_norm_sq(&le_snps_arr, &pheno_arr, num_random_vecs) / mm;
    println!("gxg_yky: {}", gxg_yky);

    let tr_gk_est = estimate_tr_k_gxg_k(&geno_arr, &le_snps_arr, num_random_vecs);
    println!("tr_gk_est: {}", tr_gk_est);

    let n = num_people as f64;
    let a = array![[tr_kk_est, tr_gk_est, n], [tr_gk_est, gxg_tr_kk_est, gxg_tr_k_est], [n, gxg_tr_k_est, n]];
    let b = array![yky, gxg_yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    println!("variance estimates: {:?}", sig_sq);
    Ok((sig_sq[0], sig_sq[1], sig_sq[2]))
}

pub fn estimate_multi_gxg_heritability(mut geno_arr: Array<f32, Ix2>, mut le_snps_arr: Vec<Array<f32, Ix2>>,
    mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize,
) -> Result<Vec<f64>, String> {
    let (num_people, num_snps) = geno_arr.dim();
    let num_gxg_components = le_snps_arr.len();
    println!("\n=> estimating heritability due to G and GxG\nnum_people: {}\nnum_snps: {}\nnumber of GxG components: {}",
             num_people, num_snps, num_gxg_components);
    for (i, arr) in le_snps_arr.iter().enumerate() {
        println!("GxG component [{}/{}]: {} LE SNPs", i + 1, num_gxg_components, arr.dim().1);
    }

    println!("\n=> normalizing the genotype matrix");
    normalize_matrix_columns_inplace(&mut geno_arr, 0);
    for (i, arr) in le_snps_arr.iter_mut().enumerate() {
        println!("=> normalizing GxG component [{}/{}]", i + 1, num_gxg_components);
        normalize_matrix_columns_inplace(arr, 0);
    }

    println!("\n=> normalizing the phenotype vector");
    pheno_arr = mean_center_vector(pheno_arr);
    pheno_arr /= std(pheno_arr.iter(), 0) as f32;

    let mut a = Array::<f64, Ix2>::zeros((num_gxg_components + 2, num_gxg_components + 2));
    let mut b = Array::<f64, Ix1>::zeros(num_gxg_components + 2);

    println!("\n=> estimating traces related to the G matrix");
    let num_rand_z = 100usize;
    let tr_kk_est = estimate_tr_kk(&geno_arr, num_rand_z);
    let xy = geno_arr.t().dot(&pheno_arr);
    let yky = sum_of_squares(xy.iter()) / num_snps as f64;
    let yy = sum_of_squares(pheno_arr.iter());
    a[[0, 0]] = tr_kk_est;
    b[0] = yky;
    b[num_gxg_components + 1] = yy;
    println!("tr_kk_est: {}\nyky: {}\nyy: {}", tr_kk_est, yky, yy);

    println!("\n=> estimating traces related to the GxG component pairs");
    for i in 0..num_gxg_components {
        for j in i + 1..num_gxg_components {
            a[[1 + i, 1 + j]] = estimate_tr_gxg_i_gxg_j(&le_snps_arr[i], &le_snps_arr[j], num_random_vecs);
            a[[1 + j, 1 + i]] = a[[1 + i, 1 + j]];
            println!("tr(gxg_k{} gxg_k{}) est: {}", i, j, a[[1 + i, 1 + j]]);
        }
    }

    println!("\n=> estimating traces related to the GxG components");
    for i in 0..num_gxg_components {
        println!("\nGXG component {}", i + 1);
        let mm = n_choose_2(le_snps_arr[i].dim().1) as f64;

        let gxg_tr_kk_est = estimate_gxg_kk_trace(&le_snps_arr[i], num_random_vecs)?;
        a[[1 + i, 1 + i]] = gxg_tr_kk_est;
        println!("gxg_tr_kk{}_est: {}", i, gxg_tr_kk_est);

        let gxg_tr_k_est = estimate_gxg_gram_trace(&le_snps_arr[i], num_random_vecs)? / mm;
        a[[num_gxg_components + 1, 1 + i]] = gxg_tr_k_est;
        a[[1 + i, num_gxg_components + 1]] = gxg_tr_k_est;
        println!("gxg_tr_k{}_est: {}", i, gxg_tr_k_est);

        let tr_gk_est = estimate_tr_k_gxg_k(&geno_arr, &le_snps_arr[i], num_random_vecs);
        a[[0, 1 + i]] = tr_gk_est;
        a[[1 + i, 0]] = tr_gk_est;
        println!("tr_gk{}_est: {}", i, tr_gk_est);

        println!("estimate_gxg_dot_y_norm_sq using {} random vectors", num_random_vecs * 2);
        let gxg_yky = estimate_gxg_dot_y_norm_sq(&le_snps_arr[i], &pheno_arr, num_random_vecs * 2) / mm;
        b[1 + i] = gxg_yky;
        println!("gxg{}_yky_est: {}", i, gxg_yky);
    }

    let n = num_people as f64;
    a[[num_gxg_components + 1, 0]] = n;
    a[[0, num_gxg_components + 1]] = n;
    a[[num_gxg_components + 1, num_gxg_components + 1]] = n;
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    println!("variance estimates: {:?}", sig_sq);
    let mut var_estimates = Vec::new();
    for i in 0..num_gxg_components + 2 {
        var_estimates.push(sig_sq[i]);
    }
    Ok(var_estimates)
}

pub fn estimate_gxg_heritability(geno_arr: Array<f32, Ix2>, mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize) -> Result<f64, String> {
    println!("\n=> estimate_gxg_heritability");
    let (num_people, num_snps) = geno_arr.dim();
    println!("num_people: {}\nnum_snps: {}", num_people, num_snps);
    let mm = n_choose_2(num_snps) as f64;

    println!("\n=> normalizing the phenotype vector");
    pheno_arr = mean_center_vector(pheno_arr);
    pheno_arr /= std(pheno_arr.iter(), 0) as f32;

    let gxg_kk_trace_est = estimate_gxg_kk_trace(&geno_arr, num_random_vecs)?;
    let gxg_k_trace_est = estimate_gxg_gram_trace(&geno_arr, num_random_vecs)? / mm;

    println!("gxg_k_trace_est: {}", gxg_k_trace_est);
    println!("gxg_kk_trace_est: {}", gxg_kk_trace_est);

    let yky = estimate_gxg_dot_y_norm_sq(&geno_arr, &pheno_arr, num_random_vecs) / mm;
    let yy = sum_of_squares(pheno_arr.iter());
    println!("yky: {}", yky);
    println!("yy: {}", yy);

    let a = array![[gxg_kk_trace_est, gxg_k_trace_est],[gxg_k_trace_est, num_people as f64]];
    let b = array![yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    let sig_sq_g = sig_sq[0] as f64;
    let sig_sq_e = sig_sq[1] as f64;
    println!("\nsig_sq: {} {}", sig_sq_g, sig_sq_e);
    let heritability = sig_sq_g / (sig_sq_g + sig_sq_e);
    bold_print(&format!("heritability: {}", heritability));

    Ok(heritability)
}
