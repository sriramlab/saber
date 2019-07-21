use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};

use crate::error::Error;
use crate::interval::{ClosedIntervalCollector, Interval};

const CHROM_FIELD_INDEX: usize = 0;
const VARIANT_ID_FIELD_INDEX: usize = 1;
const COORDINATE_FIELD_INDEX: usize = 3;
const FIRST_ALLELE_FIELD_INDEX: usize = 4;
const SECOND_ALLELE_FIELD_INDEX: usize = 5;

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

    pub fn get_chrom_positions(&mut self, chrom: &str) -> Result<Vec<Interval<usize>>, Error> {
        let mut collector = ClosedIntervalCollector::new();
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
}
