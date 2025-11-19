use crate::training_data::{
    db::{Database, ImageMetadata, Split},
    DEFAULT_DB_PATH,
};
use anyhow::Context;
use arrow_array::{cast::AsArray, types::Int64Type, BinaryArray};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::{prelude::*, ThreadPoolBuilder};
use std::io::Read;

const NUM_FILES_TRAINING: usize = 40;
const NUM_FILES_VALIDATION: usize = 2;
const NUM_FILES_TEST: usize = 3;

const BASE_URL: &str =
    "https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256/resolve/main/data";

pub fn download_data() -> anyhow::Result<()> {
    let agent = ureq::AgentBuilder::new().redirects(5).build();

    let db = Database::create(DEFAULT_DB_PATH)?;
    let db = &db;

    const MAX_DOWNLOAD_PARALLELISM: usize = 4;
    let compute_pool = ThreadPoolBuilder::new().build()?;
    let io_pool = ThreadPoolBuilder::new()
        .num_threads(MAX_DOWNLOAD_PARALLELISM)
        .build()?;

    compute_pool.scope(move |compute_scope| {
        for (split, split_name, split_count) in [
            (Split::Training, "train", NUM_FILES_TRAINING),
            (Split::Validation, "validation", NUM_FILES_VALIDATION),
            (Split::Test, "test", NUM_FILES_TEST),
        ] {
            io_pool.install(|| {
                (0..split_count).into_par_iter().try_for_each(|file_index| {
                    let url = format!(
                        "{BASE_URL}/{split_name}-{file_index:05}-of-{split_count:05}.parquet"
                    );
                    tracing::info!("Downloading {url}");
                    let mut bytes = Vec::new();
                    agent
                        .get(&url)
                        .call()?
                        .into_reader()
                        .read_to_end(&mut bytes)?;
                    let bytes = Bytes::from(bytes);

                    compute_scope.spawn(move |_| {
                        extract_parquet_to_db(bytes, db, split)
                            .expect("failed to extract parquet into DB")
                    });
                    Result::<_, anyhow::Error>::Ok(())
                })
            })?;
        }
        Result::<_, anyhow::Error>::Ok(())
    })?;

    Ok(())
}

fn extract_parquet_to_db(bytes: Bytes, db: &Database, split: Split) -> anyhow::Result<()> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)?.build()?;

    let mut tx = db.begin_transaction()?;
    let mut count = 0;
    for batch in reader {
        let batch = batch?;
        let images: &BinaryArray = batch
            .column_by_name("image")
            .context("missing image column")?
            .as_struct()
            .column_by_name("bytes")
            .context("missing image.bytes column")?
            .as_binary();
        let labels = batch
            .column_by_name("label")
            .context("missing label column")?
            .as_primitive::<Int64Type>();

        for (image, label) in images.iter().zip(labels) {
            let image = image.context("image null")?;
            let label = label.context("label null")?;

            tx.insert_image(
                ImageMetadata {
                    split,
                    label: label as i32,
                },
                image,
            )?;
            count += 1;
        }
    }
    tx.commit()?;

    tracing::debug!(
        "Extracted {:.1}K images for split {split:?}",
        count as f64 / 1000.0
    );

    Ok(())
}

pub fn compact() -> anyhow::Result<()> {
    Database::open(DEFAULT_DB_PATH)?.compact()?;
    Ok(())
}
