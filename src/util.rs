use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};

use biofile::plink_bed::{PlinkBed, PlinkSnpType};
use biofile::plink_bim::PlinkBim;
use ndarray::{Array, Ix1, Ix2, ShapeBuilder};
use biofile::util::get_buf;

pub mod matrix_util;
pub mod timer;

pub fn get_line_count(filepath: &str) -> Result<usize, String> {
    let buf = match OpenOptions::new().read(true).open(filepath) {
        Err(why) => return Err(format!("failed to open {}: {}", filepath, why)),
        Ok(f) => BufReader::new(f)
    };
    Ok(buf.lines().count())
}

pub fn get_bed_bim_fam_path(bfile: &str) -> (String, String, String) {
    (format!("{}.bed", bfile), format!("{}.bim", bfile), format!("{}.fam", bfile))
}

pub fn get_bed_bim_from_prefix_and_partition(
    plink_filename_prefixes: &Vec<String>,
    plink_dominance_prefixes: &Option<Vec<String>>,
    partition_filepath: &Option<String>,
) -> Result<(PlinkBed, PlinkBim), biofile::error::Error> {
    let bfile_prefix_snptype_list = {
        let mut list = plink_filename_prefixes
            .iter()
            .map(|p| (p.to_string(), PlinkSnpType::Additive))
            .collect::<Vec<(String, PlinkSnpType)>>();
        if let Some(dominance_prefixes) = plink_dominance_prefixes {
            list.append(
                &mut dominance_prefixes
                    .iter()
                    .map(|p| (p.to_string(), PlinkSnpType::Dominance))
                    .collect::<Vec<(String, PlinkSnpType)>>()
            );
        }
        list
    };
    let bed_bim_fam_snptype_list: Vec<(String, String, String, PlinkSnpType)> =
        bfile_prefix_snptype_list
            .iter()
            .map(|(prefix, snp_type)| {
                let (bed, bim, fam) = get_bed_bim_fam_path(prefix);
                (bed, bim, fam, *snp_type)
            })
            .collect();
    let bed = PlinkBed::new(&bed_bim_fam_snptype_list)?;

    let bim_path_list: Vec<String> = bed_bim_fam_snptype_list
        .iter()
        .map(|t| t.1.to_string())
        .collect();

    let bim = match partition_filepath {
        Some(partition_filepath) => PlinkBim::new_with_partition_file(
            bim_path_list,
            partition_filepath,
        )?,
        None => PlinkBim::new(bim_path_list)?
    };
    Ok((bed, bim))
}

pub fn get_fid_iid_list(
    fam_file_path: &str
) -> Result<Vec<(String, String)>, biofile::error::Error> {
    Ok(get_buf(fam_file_path)?
        .lines()
        .map(|l| {
            let toks: Vec<String> =
                l.unwrap()
                 .split_whitespace()
                 .map(|t| t.to_string())
                 .collect();
            (toks[0].to_owned(), toks[1].to_owned())
        })
        .collect()
    )
}

pub fn get_file_lines(filepath: &str) -> Result<Vec<String>, std::io::Error> {
    Ok(
        BufReader::new(OpenOptions::new().read(true).open(filepath)?)
            .lines()
            .map(|l| l
                .unwrap()
                .to_string()
            )
            .collect()
    )
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

pub fn get_pheno_path_to_arr(
    pheno_path_vec: &Vec<String>
) -> Result<HashMap<String, Array<f32, Ix1>>, String> {
    pheno_path_vec
        .iter()
        .map(|p| Ok((p.to_string(), get_pheno_arr(p)?)))
        .collect::<Result<HashMap<String, Array<f32, Ix1>>, String>>()
}

pub fn get_pheno_matrix(
    pheno_path_vec: &Vec<String>
) -> Result<Array<f32, Ix2>, String> {
    let v: Vec<f32> = pheno_path_vec
        .iter()
        .map(|p| Ok(get_pheno_arr(p)?.to_vec()))
        .collect::<Result<Vec<Vec<f32>>, String>>()?
        .into_iter()
        .flat_map(|v| v)
        .collect();
    let num_pheno_types = pheno_path_vec.len();
    let num_rows = v.len() / num_pheno_types;
    Ok(
        Array::from_shape_vec((num_rows, num_pheno_types).strides((1, num_rows)), v).unwrap()
    )
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
    use std::io::{Write, BufWriter};

    use ndarray::Array;
    use tempfile::NamedTempFile;

    use crate::util::{load_trace_estimates, validate_header, write_trace_estimates, get_fid_iid_list};
    use std::fs::OpenOptions;

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

    #[test]
    fn test_get_fid_iid_list() {
        let fam_path = NamedTempFile::new().unwrap().into_temp_path();
        {
            let mut writer = BufWriter::new(
                OpenOptions::new().create(true).truncate(true).write(true).open(
                    fam_path.to_str().unwrap()
                ).unwrap()
            );
            writer.write_fmt(
                format_args!(
                    "1532 1532\n\
                0924 0924\n\
                1254 1323\n\
                123 123\n"
                )
            ).unwrap();
        }
        let fid_iid_list = get_fid_iid_list(fam_path.to_str().unwrap()).unwrap();
        let mut iter = fid_iid_list.into_iter();
        assert_eq!(iter.next(), Some(("1532".to_string(), "1532".to_string())));
    }
}

