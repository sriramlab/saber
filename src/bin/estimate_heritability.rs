#[macro_use]
extern crate clap;
#[macro_use]
extern crate ndarray;
extern crate saber;

use bio_file_reader::plink_bed::PlinkBed;
use saber::heritability_estimator::estimate_heritability;
use saber::program_flow::OrExit;
use saber::util::{extract_str_arg, get_pheno_arr};

fn main() {
    let matches = clap_app!(estimate_heritability =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg plink_filename_prefix: --bfile <BFILE> "required; the prefix for x.bed, x.bim, x.fam is x")
        (@arg pheno_filename: --pheno <PHENO> "required; each row is one individual containing one phenotype value")
        (@arg num_random_vecs: --nrv +takes_value "number of random vectors used to estimate traces; required")
    ).get_matches();

    let plink_filename_prefix = extract_str_arg(&matches, "plink_filename_prefix");
    let pheno_filename = extract_str_arg(&matches, "pheno_filename");

    let plink_bed_path = format!("{}.bed", plink_filename_prefix);
    let plink_bim_path = format!("{}.bim", plink_filename_prefix);
    let plink_fam_path = format!("{}.fam", plink_filename_prefix);

    let num_random_vecs = extract_str_arg(&matches, "num_random_vecs")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_random_vecs"));

    println!("PLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}",
             plink_bed_path, plink_bim_path, plink_fam_path);
    println!("pheno_filepath: {}\nnum_random_vecs: {}", pheno_filename, num_random_vecs);

    let pheno_arr = get_pheno_arr(&pheno_filename)
        .unwrap_or_exit(None::<String>);

    let mut bed = PlinkBed::new(&plink_bed_path,
                                &plink_bim_path,
                                &plink_fam_path)
        .unwrap_or_exit(None::<String>);

    let geno_arr = bed.get_genotype_matrix()
                      .unwrap_or_exit(Some("failed to get the genotype matrix"));

    match estimate_heritability(geno_arr,
                                pheno_arr,
                                num_random_vecs) {
        Ok(h) => {
            println!("\nheritability estimate: {}", h);
        }
        Err(why) => {
            eprintln!("{}", why);
            return ();
        }
    };
}
