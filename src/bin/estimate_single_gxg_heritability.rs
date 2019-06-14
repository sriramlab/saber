extern crate saber;

#[macro_use]
extern crate clap;

#[cfg(feature = "cuda")]
use co_blas::transpose::Transpose;

#[macro_use]
extern crate ndarray;

use ndarray::{Array, Ix1};

use bio_file_reader::plink_bed::PlinkBed;

#[cfg(feature = "cuda")]
use estimate_heritability_cublas as estimate_heritability;
#[cfg(not(feature = "cuda"))]
use saber::heritability_estimator::estimate_g_and_multi_gxg_heritability;

use saber::util::matrix_util::{generate_plus_minus_one_bernoulli_matrix, mean_center_vector,
                               normalize_matrix_row_wise_inplace, row_mean_vec, row_std_vec};
use saber::program_flow::OrExit;
use saber::util::stats_util::sum_of_squares;
use saber::util::timer::Timer;
use saber::util::{extract_str_arg, get_pheno_arr, get_bed_bim_fam_path, extract_numeric_arg};

#[cfg(feature = "cuda")]
use saber::cublas::linalg::mul_xtxz_f32;

#[cfg(feature = "cuda")]
fn estimate_heritability_cublas(mut geno_arr: Array<f32, Ix2>, mut pheno_arr: Array<f32, Ix1>,
    num_random_vecs: usize) -> Result<f64, String> {
    println!("\nusing cuBLAS\n=> creating the genotype ndarray and starting the timer for profiling");
    let mut timer = Timer::new();
    // geno_arr is num_snps x num_people
    let (num_rows, num_cols) = geno_arr.dim();
    println!("geno_arr dim: {:?}", geno_arr.dim());
    timer.print();

    println!("\n=> normalizing the genotype matrix row-wise");
    geno_arr = normalize_matrix_row_wise_inplace(geno_arr, 1);
    timer.print();

    println!("\n=> mean centering the phenotype vector");
    mean_center_vector(&mut pheno_arr);
    timer.print();

    println!("\n=> generating random estimators");
    let rand_mat = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    timer.print();

    println!("\n=> getting xxz");
    let xxz = Array::from_shape_vec((num_cols, num_random_vecs),
                                    mul_xtxz_f32(geno_arr.as_slice().unwrap(), rand_mat.as_slice().unwrap(),
                                                 num_rows, num_random_vecs, num_cols)).unwrap();
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

fn main() {
    let matches = clap_app!(estimate_single_gxg_heritability =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg bfile: --bfile <BFILE> "required; the prefix for x.bed, x.bim, x.fam is x")
        (@arg le_snps_path: --le <LE_SNPS> "required; plink file prefix to the SNPs in linkage equilibrium")
        (@arg pheno_path: --pheno <PHENO> "required; each row is one individual containing one phenotype value")
        (@arg num_le_snps_to_use: -n +takes_value "number of independent SNPs to use; required")
        (@arg num_random_vecs: --nrv +takes_value "number of random vectors used to estimate traces; required")
    ).get_matches();

    let bfile = extract_str_arg(&matches, "bfile");
    let le_snps_path = extract_str_arg(&matches, "le_snps_path");
    let pheno_path = extract_str_arg(&matches, "pheno_path");

    let [bed_path, bim_path, fam_path] = get_bed_bim_fam_path(&bfile);
    let [le_snps_bed_path, le_snps_bim_path, le_snps_fam_path] = get_bed_bim_fam_path(&le_snps_path);

    let num_le_snps_to_use = extract_numeric_arg::<usize>(&matches, "num_le_snps_to_use")
        .unwrap_or_exit(Some("failed to parse num_le_snps_to_use"));

    let num_random_vecs = extract_numeric_arg::<usize>(&matches, "num_random_vecs")
        .unwrap_or_exit(Some("failed to parse num_random_vecs"));

    println!("PLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}",
             bed_path, bim_path, fam_path);
    println!("LE SNPs bed path: {}\nLE SNPs bim path: {}\nLE SNPs fam path: {}",
             le_snps_bed_path, le_snps_bim_path, le_snps_fam_path);
    println!("pheno_filepath: {}", pheno_path);
    println!("num_le_snps_to_use: {}\nnum_random_vecs: {}", num_le_snps_to_use, num_random_vecs);

    println!("\n=> generating the phenotype array and the genotype matrix");

    let pheno_arr = get_pheno_arr(&pheno_path)
        .unwrap_or_exit(None::<String>);

    let mut bed = PlinkBed::new(&bed_path, &bim_path, &fam_path)
        .unwrap_or_exit(None::<String>);
    let geno_arr = bed.get_genotype_matrix()
                      .unwrap_or_exit(Some("failed to get the genotype matrix"));

    let mut le_snps_bed = PlinkBed::new(&le_snps_bed_path, &le_snps_bim_path, &le_snps_fam_path)
        .unwrap_or_exit(None::<String>);
    let mut le_snps_arr = le_snps_bed.get_genotype_matrix()
                                     .unwrap_or_exit(Some("failed to get the le_snps genotype matrix"));
    le_snps_arr = le_snps_arr.slice(s![.., ..num_le_snps_to_use]).to_owned();

    match estimate_g_and_multi_gxg_heritability(geno_arr,
                                                vec![le_snps_arr],
                                                pheno_arr,
                                                num_random_vecs) {
        Ok((_a, _b, h, _, _, _)) => {
            println!("\nvariance estimates on the normalized phenotype:\nG variance: {}\nGxG variance: {}\nnoise variance: {}",
                     h[0], h[1], h[2]);
        }
        Err(why) => {
            eprintln!("{}", why);
            return ();
        }
    };
}

#[cfg(test)]
mod tests {
    extern crate rand;

    use std::collections::HashSet;

    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::distributions::StandardNormal;

    use crate::generate_plus_minus_one_bernoulli_matrix;
    use saber::util::stats_util::sum_of_squares;

    #[test]
    fn test_trace_estimator() {
        let n = 1000;
        let num_random_vecs = 40;
        let x = Array::random(
            (n, n),
            StandardNormal).mapv(|e| e as i32 as f32);
        // want to estimate the trace of x.t().dot(&x)
        let true_trace = sum_of_squares(x.iter());
        println!("true trace: {}", true_trace);

        let rand_mat = generate_plus_minus_one_bernoulli_matrix(n, num_random_vecs);

        let trace_est = sum_of_squares(x.dot(&rand_mat).iter()) / num_random_vecs as f64;
        println!("trace_est: {}", trace_est);
    }

    #[test]
    fn test_bernoulli_matrix() {
        let n = 1000;
        let num_random_vecs = 100;
        let rand_mat = generate_plus_minus_one_bernoulli_matrix(n, num_random_vecs);
        assert_eq!((n, num_random_vecs), rand_mat.dim());
        let mut value_set = HashSet::<i32>::new();
        for a in rand_mat.iter() {
            value_set.insert(*a as i32);
        }
        // almost certainly this will contain the two values 1 and -1
        assert_eq!(2, value_set.len());
        assert_eq!(true, value_set.contains(&-1));
        assert_eq!(true, value_set.contains(&1));
    }
}
