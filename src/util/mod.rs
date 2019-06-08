use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader};

use clap::ArgMatches;
use ndarray::{Array, Ix1, Ix2};

pub mod histogram;
pub mod matrix_util;
pub mod stats_util;

pub fn get_line_count(filepath: &String) -> Result<usize, String> {
    let buf = match OpenOptions::new().read(true).open(filepath.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", filepath, why)),
        Ok(f) => BufReader::new(f)
    };
    Ok(buf.lines().count())
}

pub fn extract_str_arg(matches: &ArgMatches, arg_name: &str) -> String {
    match matches.value_of(arg_name) {
        Some(filename) => filename.to_string(),
        None => {
            eprintln!("the argument {} is required", arg_name);
            std::process::exit(1);
        }
    }
}

fn validate_header(header: &String, expected_first_n_tokens: Vec<String>) -> Result<(), String> {
    let header_toks: Vec<String> = header.split_whitespace().map(|t| t.to_owned()).collect();
    for (i, (actual, expected)) in header_toks.into_iter().zip(expected_first_n_tokens).enumerate() {
        if actual != expected {
            return Err(format!("expected the header field at position {} to be {}, received {}", i, expected, actual));
        }
    }
    Ok(())
}

fn read_and_validate_plink_header(buf: &mut BufReader<File>) -> Result<String, String> {
    let mut header = String::new();
    let _ = buf.read_line(&mut header);
    header = header.trim_end().to_string();
    validate_header(&header, vec!["FID".to_string(), "IID".to_string()])?;
    Ok(header)
}

/// The first line of the file is FID IID pheno
/// Each of the remaining lines have the three corresponding fields
///
/// returns an array containing only the phenotype values in the order listed in the file
pub fn get_pheno_arr(pheno_path: &String) -> Result<Array<f32, Ix1>, String> {
    let mut buf = match OpenOptions::new().read(true).open(pheno_path.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", pheno_path, why)),
        Ok(f) => BufReader::new(f)
    };

    let header = read_and_validate_plink_header(&mut buf)?;
    println!("\n{} header:\n{}", pheno_path, header);

    let pheno_vec = buf.lines().map(|l|
        l.unwrap()
         .split_whitespace()
         .nth(2).unwrap()
         .parse::<f32>().unwrap()
    ).collect();

    Ok(Array::from_vec(pheno_vec))
}

/// The first line of the file is FID IID pheno
/// Each of the remaining lines have the three corresponding fields
///
/// returns (header, FID vector, IID vector, pheno vector) where the vectors are in the order listed in the file
pub fn get_plink_pheno_data(pheno_path: &String) -> Result<(String, Vec<String>, Vec<String>, Array<f32, Ix1>), String> {
    let mut buf = match OpenOptions::new().read(true).open(pheno_path.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", pheno_path, why)),
        Ok(f) => BufReader::new(f)
    };

    let header = read_and_validate_plink_header(&mut buf)?;
    println!("\n{} header:\n{}", pheno_path, header);

    let mut pheno_vec = Vec::new();
    let mut fid_vec = Vec::new();
    let mut iid_vec = Vec::new();
    for l in buf.lines() {
        let toks: Vec<String> = l.unwrap().split_whitespace().map(|t| t.to_string()).collect();
        fid_vec.push(toks[0].to_owned());
        iid_vec.push(toks[1].to_owned());
        pheno_vec.push(toks[2].parse::<f32>().unwrap());
    }
    Ok((header, fid_vec, iid_vec, Array::from_vec(pheno_vec)))
}

/// The first line of the file starts with FID IID, followed by any number of covariate names.
/// Each of the remaining lines of the file has the corresponding fields.
pub fn get_plink_covariate_arr(covariate_path: &String) -> Result<Array<f32, Ix2>, String> {
    let num_people = get_line_count(covariate_path)? - 1;
    println!("\n{} contains {} people", covariate_path, num_people);

    let mut buf = match OpenOptions::new().read(true).open(covariate_path.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", covariate_path, why)),
        Ok(f) => BufReader::new(f)
    };

    let header = read_and_validate_plink_header(&mut buf)?;
    println!("\n{} header:\n{}", covariate_path, header);

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

#[cfg(test)]
mod tests {
    use super::validate_header;

    #[test]
    fn test_validate_header() {
        assert_eq!(Ok(()), validate_header(&"FID IID".to_string(), vec!["FID".to_string(), "IID".to_string()]));
        assert_eq!(Ok(()), validate_header(&"FID IID pheno".to_string(), vec!["FID".to_string(), "IID".to_string()]));
        assert!(validate_header(&"FID WRONG pheno".to_string(), vec!["FID".to_string(), "IID".to_string()]).is_err());
        assert!(validate_header(&"FID IID".to_string(), Vec::new()).is_ok());
        assert!(validate_header(&"".to_string(), Vec::new()).is_ok());
    }
}

