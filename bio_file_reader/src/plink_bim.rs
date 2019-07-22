use std::collections::{HashSet, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};

use crate::error::Error;
use crate::interval::{ClosedIntegerIntervalCollector, IntervalOverIntegers};

pub const CHROM_FIELD_INDEX: usize = 0;
pub const VARIANT_ID_FIELD_INDEX: usize = 1;
pub const COORDINATE_FIELD_INDEX: usize = 3;
pub const FIRST_ALLELE_FIELD_INDEX: usize = 4;
pub const SECOND_ALLELE_FIELD_INDEX: usize = 5;

pub struct PlinkBim {
    pub filepath: String,
    buf: BufReader<File>,
}

impl PlinkBim {
    pub fn new(filepath: &str) -> Result<PlinkBim, Error> {
        let buf = BufReader::new(OpenOptions::new().read(true).open(filepath)?);
        Ok(PlinkBim {
            filepath: filepath.to_string(),
            buf,
        })
    }

    pub fn reset_buf(&mut self) -> Result<(), Error> {
        self.buf.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    pub fn get_all_chroms(&mut self) -> Result<HashSet<String>, Error> {
        self.reset_buf()?;
        return Ok(self.buf.by_ref().lines().filter_map(|l|
            l.unwrap().split_whitespace().nth(CHROM_FIELD_INDEX).and_then(|s| Some(s.to_string()))
        ).collect());
    }

    pub fn get_chrom_fileline_positions(&mut self, chrom: &str) -> Result<Vec<IntervalOverIntegers<usize>>, Error> {
        let mut collector = ClosedIntegerIntervalCollector::new();
        self.reset_buf()?;
        for (i, l) in self.buf.by_ref().lines().enumerate() {
            if l.unwrap()
                .split_whitespace()
                .nth(CHROM_FIELD_INDEX).unwrap() == chrom {
                collector.append_larger_point(i)?;
            }
        }
        Ok(collector.get_intervals())
    }

    pub fn get_chrom_to_fileline_positions(&mut self) -> Result<HashMap<String, Vec<IntervalOverIntegers<usize>>>, Error> {
        let mut chrom_to_positions = HashMap::new();
        for chrom in self.get_all_chroms()? {
            let positions = self.get_chrom_fileline_positions(&chrom)?;
            chrom_to_positions.insert(chrom, positions);
        }
        Ok(chrom_to_positions)
    }
}

#[cfg(test)]
mod tests {
    use super::PlinkBim;
    use std::io::{BufWriter, Write};
    use tempfile::NamedTempFile;
    use std::collections::{HashSet, HashMap};
    use crate::interval::{IntervalOverIntegers};

    fn write_bim_line<W: Write>(buf_writer: &mut BufWriter<W>, chrom: &str, id: &str, coordinate: u64,
        first_allele: char, second_allele: char) {
        buf_writer.write_fmt(format_args!("{} {} {} {} {} {}\n", chrom, id, 0, coordinate, first_allele, second_allele)).unwrap();
    }

    #[test]
    fn test_get_chrom_to_positions() {
        let file = NamedTempFile::new().unwrap();
        {
            let mut writer = BufWriter::new(&file);
            for (chrom, coord) in &[
                (1, 2), (1, 3), (1, 4), (1, 7), (1, 8), (1, 20),
                (3, 4), (3, 5), (3, 6), (3, 10),
                (4, 100),
                (5, 1), (5, 10),
                (3, 32), (3, 2), (3, 1),
                (5, 4), (5, 8),
            ] {
                write_bim_line(&mut writer, &chrom.to_string(), "id", *coord, 'A', 'C');
            }
        }
        let mut bim = PlinkBim::new(file.into_temp_path().to_str().unwrap()).unwrap();
        let positions = bim.get_chrom_to_fileline_positions().unwrap();
        let expected: HashMap<String, Vec<IntervalOverIntegers<usize>>> = vec![
            ("1".to_string(), vec![IntervalOverIntegers::new(0, 5)]),
            ("3".to_string(), vec![
                IntervalOverIntegers::new(6, 9),
                IntervalOverIntegers::new(13, 15)
            ]),
            ("4".to_string(), vec![IntervalOverIntegers::new(10, 10)]),
            ("5".to_string(), vec![
                IntervalOverIntegers::new(11, 12),
                IntervalOverIntegers::new(16, 17),
            ])
        ].into_iter().collect();
        assert_eq!(positions, expected);
    }

    #[test]
    fn test_get_all_chroms() {
        let file = NamedTempFile::new().unwrap();
        {
            let mut writer = BufWriter::new(&file);
            for chrom in &[1, 2, 1, 3, 2, 5] {
                write_bim_line(&mut writer, &chrom.to_string(), "id", 0, 'A', 'C');
            }
        }
        let mut bim = PlinkBim::new(file.into_temp_path().to_str().unwrap()).unwrap();
        let chrom_set = bim.get_all_chroms().unwrap();
        let expected: HashSet<String> = vec!["1", "2", "3", "5"].into_iter().map(|x| x.to_string()).collect();
        assert_eq!(chrom_set, expected);
    }
}
