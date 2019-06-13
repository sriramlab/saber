#[macro_use]
extern crate clap;
#[macro_use]
extern crate ndarray;
extern crate saber;

use bio_file_reader::plink_bed::PlinkBed;
use saber::heritability_estimator::estimate_g_and_multi_gxg_heritability;
use saber::program_flow::OrExit;
use saber::simulation::sim_pheno::{generate_gxg_pheno_arr_from_gxg_basis, generate_pheno_arr};
use saber::util::extract_str_arg;
use saber::util::matrix_util::normalize_matrix_columns_inplace;

fn main() {
    let matches = clap_app!(estimate_heritability_sim_pheno =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg plink_filename_prefix: --bfile <BFILE> "required; the prefix for x.bed, x.bim, x.fam is x")
        (@arg le_snps_filename: --le <LE_SNPS> "required; plink file prefix to the SNPs in linkage equilibrium")
        (@arg num_le_snps_to_use: -n +takes_value "number of independent SNPs to use; required")
        (@arg g_var: --g +takes_value "G variance; required")
        (@arg gxg_var: --gxg +takes_value "GxG variance; required")
        (@arg num_random_vecs: --nrv +takes_value "number of random vectors used to estimate traces; required")
    ).get_matches();

    let plink_filename_prefix = extract_str_arg(&matches, "plink_filename_prefix");
    let le_snps_filename = extract_str_arg(&matches, "le_snps_filename");

    let plink_bed_path = format!("{}.bed", plink_filename_prefix);
    let plink_bim_path = format!("{}.bim", plink_filename_prefix);
    let plink_fam_path = format!("{}.fam", plink_filename_prefix);

    let le_snps_bed_path = format!("{}.bed", le_snps_filename);
    let le_snps_bim_path = format!("{}.bim", le_snps_filename);
    let le_snps_fam_path = format!("{}.fam", le_snps_filename);

    let num_le_snps_to_use = extract_str_arg(&matches, "num_le_snps_to_use")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_le_snps_to_use"));

    let g_var = extract_str_arg(&matches, "g_var")
        .parse::<f64>()
        .unwrap_or_exit(Some("failed to parse g_var"));

    let gxg_var = extract_str_arg(&matches, "gxg_var")
        .parse::<f64>()
        .unwrap_or_exit(Some("failed to parse gxg_var"));

    let num_random_vecs = extract_str_arg(&matches, "num_random_vecs")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_random_vecs"));

    println!("PLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}\n",
             plink_bed_path, plink_bim_path, plink_fam_path);
    println!("LE SNPs bed path: {}\nLE SNPs bim path: {}\nLE SNPs fam path: {}",
             le_snps_bed_path, le_snps_bim_path, le_snps_fam_path);
    println!("num_le_snps_to_use: {}\ng_var: {} \ngxg_var: {}\nnum_random_vecs: {}",
             num_le_snps_to_use, g_var, gxg_var, num_random_vecs);

    println!("\n=> generating the genotype matrix");

    let mut bed = PlinkBed::new(&plink_bed_path,
                                &plink_bim_path,
                                &plink_fam_path).unwrap_or_exit(None::<String>);
    let mut geno_arr = bed.get_genotype_matrix()
                          .unwrap_or_exit(Some("failed to get the genotype matrix"));

    let mut le_snps_bed = PlinkBed::new(&le_snps_bed_path,
                                        &le_snps_bim_path,
                                        &le_snps_fam_path).unwrap_or_exit(None::<String>);
    let mut le_snps_arr = le_snps_bed.get_genotype_matrix()
                                     .unwrap_or_exit(Some("failed to get the le_snps genotype matrix"));
    le_snps_arr = le_snps_arr.slice(s![.., ..num_le_snps_to_use]).to_owned();

    println!("geno_arr.dim: {:?}\nle_snps_arr.dim: {:?}", geno_arr.dim(), le_snps_arr.dim());

    println!("\n=> normalizing genotype matrices before simulating phenotypes");
    normalize_matrix_columns_inplace(&mut geno_arr, 0);
    normalize_matrix_columns_inplace(&mut le_snps_arr, 0);

    println!("\n=> simulating phenotypes");
    let pheno_arr;
    if gxg_var == 0. {
        println!("\n=> gxg variance is 0. Generating the phenotypes using only the geno_arr");
        pheno_arr = generate_pheno_arr(&geno_arr, g_var, 1. - g_var);
    } else {
        pheno_arr = generate_gxg_pheno_arr_from_gxg_basis(&geno_arr, &le_snps_arr,
                                                          g_var, gxg_var, 1. - g_var - gxg_var);
    }

    match estimate_g_and_multi_gxg_heritability(geno_arr,
                                                vec![le_snps_arr],
                                                pheno_arr,
                                                num_random_vecs) {
        Ok((_a, _b, h)) => {
            println!("\nvariance estimates on the normalized phenotype:\nG variance: {}\nGxG variance: {}\nnoise variance: {}",
                     h[0], h[1], h[2]);
        }
        Err(why) => {
            eprintln!("{}", why);
            return ();
        }
    };
}

