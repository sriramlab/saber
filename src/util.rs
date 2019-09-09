use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};

use ndarray::{Array, Ix1, Ix2, ShapeBuilder};

pub mod matrix_util;
pub mod timer;

pub fn get_line_count(filepath: &str) -> Result<usize, String> {
    let buf = match OpenOptions::new().read(true).open(filepath) {
        Err(why) => return Err(format!("failed to open {}: {}", filepath, why)),
        Ok(f) => BufReader::new(f)
    };
    Ok(buf.lines().count())
}

pub fn get_bed_bim_fam_path(bfile: &str) -> [String; 3] {
    [format!("{}.bed", bfile), format!("{}.bim", bfile), format!("{}.fam", bfile)]
}

pub fn load_trace_estimates(load_path: &str) -> Result<Array<f64, Ix2>, String> {
    let num_rows = get_line_count(load_path)?;
    let buf = match OpenOptions::new().read(true).open(load_path) {
        Err(why) => return Err(format!("failed to read the trace estimates from file {}: {}", load_path, why)),
        Ok(f) => BufReader::new(f)
    };
    let trace_vec: Vec<f64> = buf.lines().flat_map(|l|
        l.unwrap()
         .split_whitespace()
         .map(|val| val.parse::<f64>().unwrap())
         .collect::<Vec<f64>>()
    ).collect();
    let num_cols = trace_vec.len() / num_rows;
    Ok(Array::from_shape_vec((num_rows, num_cols).strides((num_cols, 1)), trace_vec).unwrap())
}

pub fn write_trace_estimates(trace_estimates: &Array<f64, Ix2>, out_path: &str) -> Result<(), String> {
    let mut buf = match OpenOptions::new().truncate(true).create(true).write(true).open(out_path) {
        Err(why) => return Err(format!("failed to write the trace estimates to file {}: {}", out_path, why)),
        Ok(f) => BufWriter::new(f)
    };
    for row in trace_estimates.genrows() {
        for val in row.iter() {
            if let Err(why) = buf.write_fmt(format_args!("{} ", val)) {
                return Err(format!("failed to write the trace estimates to file {}: {}", out_path, why));
            }
        }
        if let Err(why) = buf.write_fmt(format_args!("\n")) {
            return Err(format!("failed to write the trace estimates to file {}: {}", out_path, why));
        }
    }
    Ok(())
}

fn validate_header(header: &str, expected_first_n_tokens: Vec<String>) -> Result<(), String> {
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
pub fn get_pheno_arr(pheno_path: &str) -> Result<Array<f32, Ix1>, String> {
    let mut buf = match OpenOptions::new().read(true).open(pheno_path) {
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
pub fn get_plink_pheno_data(pheno_path: &str) -> Result<(String, Vec<String>, Vec<String>, Array<f32, Ix1>), String> {
    let mut buf = match OpenOptions::new().read(true).open(pheno_path) {
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

enum PhenoVal<T> {
    Missing,
    Present(T),
}

/// The first line of the file is FID IID pheno
/// Each of the remaining lines have the three corresponding fields
///
/// returns (header, FID vector, IID vector, pheno vector) where the vectors are in the order listed in the file
/// and the missing phenotype values are replaced with the mean for the returned phenotype vector
pub fn get_plink_pheno_data_replace_missing_with_mean(pheno_path: &str, missing_reps_vec: &Vec<String>)
    -> Result<(String, Vec<String>, Vec<String>, Array<f32, Ix1>), String> {
    let missing_reps: HashSet<String> = missing_reps_vec.iter().cloned().collect();

    let mut buf = match OpenOptions::new().read(true).open(pheno_path) {
        Err(why) => return Err(format!("failed to open {}: {}", pheno_path, why)),
        Ok(f) => BufReader::new(f)
    };

    let header = read_and_validate_plink_header(&mut buf)?;
    println!("\n{} header:\n{}", pheno_path, header);

    let mut pheno = Vec::new();
    let mut fid_vec = Vec::new();
    let mut iid_vec = Vec::new();
    for l in buf.lines() {
        let toks: Vec<String> = l.unwrap().split_whitespace().map(|t| t.to_string()).collect();
        fid_vec.push(toks[0].to_owned());
        iid_vec.push(toks[1].to_owned());
        if missing_reps.contains(&toks[2]) {
            pheno.push(PhenoVal::Missing);
        } else {
            pheno.push(PhenoVal::Present(toks[2].parse::<f32>().unwrap()));
        }
    }
    let non_missing_count_sum = pheno.iter().fold((0usize, 0.), |(count, sum), val| {
        match val {
            PhenoVal::Missing => (count, sum),
            PhenoVal::Present(val) => (count + 1, sum + *val)
        }
    });

    let pheno_mean = non_missing_count_sum.1 / non_missing_count_sum.0 as f32;
    println!("\n[{}/{}] non-missing phenotype values, with mean: {}", non_missing_count_sum.0, pheno.len(), pheno_mean);
    let pheno_vec = pheno.iter().map(|v| match v {
        PhenoVal::Missing => pheno_mean,
        PhenoVal::Present(v) => *v
    }).collect();

    Ok((header, fid_vec, iid_vec, Array::from_vec(pheno_vec)))
}

/// The first line of the file starts with FID IID, followed by any number of covariate names.
/// Each of the remaining lines of the file has the corresponding fields.
pub fn get_plink_covariate_arr(covariate_path: &str) -> Result<Array<f32, Ix2>, String> {
    let num_people = get_line_count(covariate_path)? - 1;
    println!("\n{} contains {} people", covariate_path, num_people);

    let mut buf = match OpenOptions::new().read(true).open(covariate_path) {
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
    use std::io::Write;

    use ndarray::Array;
    use tempfile::NamedTempFile;

    use crate::util::{load_trace_estimates, validate_header, write_trace_estimates};

    #[test]
    fn test_validate_header() {
        assert_eq!(Ok(()), validate_header("FID IID", vec!["FID".to_string(), "IID".to_string()]));
        assert_eq!(Ok(()), validate_header("FID IID pheno", vec!["FID".to_string(), "IID".to_string()]));
        assert!(validate_header("FID WRONG pheno", vec!["FID".to_string(), "IID".to_string()]).is_err());
        assert!(validate_header("FID IID", Vec::new()).is_ok());
        assert!(validate_header("", Vec::new()).is_ok());
    }

    #[test]
    fn test_load_trace_estimates() {
        let mut file = NamedTempFile::new().unwrap();
        let arr = vec![vec![2., 123., 0.003, 23., -409.], vec![-0., 1.23, -2.43, 0., -9.]];
        for row in arr.iter() {
            for val in row.iter() {
                write!(file, "{} ", val).unwrap();
            }
            write!(file, "\n").unwrap();
        }
        let estimates = load_trace_estimates(
            file.path().as_os_str().to_str().unwrap()).unwrap();
        let true_estimates = Array::from_shape_vec(
            (2, 5),
            arr.into_iter().flat_map(|a| a).collect::<Vec<f64>>()).unwrap();
        assert_eq!(estimates, true_estimates);
    }

    #[test]
    fn test_write_trace_estimates() {
        let file = NamedTempFile::new().unwrap();
        let path = file.into_temp_path().to_str().unwrap().to_string();
        let estimates = Array::from_shape_vec((2, 5),
                                              vec![2., 123., 0.003, 23., -409.,
                                                   -0., 1.23, -2.43, 0., -9.]).unwrap();
        write_trace_estimates(&estimates, &path).unwrap();

        let loaded_estimates = load_trace_estimates(&path).unwrap();

        assert_eq!(loaded_estimates, estimates);
    }
}

