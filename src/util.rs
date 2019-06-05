use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};
use clap::ArgMatches;

use ndarray::{Array, Ix1, Ix2};

pub fn extract_str_arg(matches: &ArgMatches, arg_name: &str) -> String {
    match matches.value_of(arg_name) {
        Some(filename) => filename.to_string(),
        None => {
            eprintln!("the argument {} is required", arg_name);
            std::process::exit(1);
        }
    }
}

pub fn get_pheno_arr(pheno_filename: &String) -> Result<Array<f32, Ix1>, String> {
    let buf = match OpenOptions::new().read(true).open(pheno_filename.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", pheno_filename, why)),
        Ok(f) => BufReader::new(f)
    };
    let pheno_vec: Vec<f32> = buf.lines().map(|l| l.unwrap().parse::<f32>().unwrap()).collect();
    Ok(Array::from_vec(pheno_vec))
}

pub fn get_line_count(filepath: &String) -> Result<usize, String> {
    let buf = match OpenOptions::new().read(true).open(filepath.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", filepath, why)),
        Ok(f) => BufReader::new(f)
    };
    Ok(buf.lines().count())
}

pub fn get_plink_covariate_arr(covariate_filepath: &String) -> Result<Array<f32, Ix2>, String> {
    let num_people = get_line_count(covariate_filepath)? - 1;
    println!("num_people: {}", num_people);

    let mut buf = match OpenOptions::new().read(true).open(covariate_filepath.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", covariate_filepath, why)),
        Ok(f) => BufReader::new(f)
    };

    // get rid of the header
    let mut header = String::new();
    let _ = buf.read_line(&mut header);
    println!("\n{} header:\n{}", covariate_filepath, header);

    let covariate_vec: Vec<f32> = buf.lines().flat_map(|l| {
        l.unwrap()
         .split_whitespace()
         .skip(2)
         .map(|s| s.parse::<f32>().unwrap())
         .collect::<Vec<f32>>()
    }).collect();

    assert_eq!(covariate_vec.len() % num_people, 0,
               "total number of elements {} is not divisible by num_people {}", covariate_vec.len(), num_people);
    let arr = Array::<f32, Ix2>::from_shape_vec((num_people, covariate_vec.len() / num_people), covariate_vec).unwrap();
    Ok(arr)
}
