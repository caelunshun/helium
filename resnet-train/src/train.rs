use crate::{
    model::{Model, NUM_CLASSES},
    train::preview::{Example, Previewer},
    training_data::{
        augmentation::augment_training_image,
        db::{Database, Split},
        DEFAULT_DB_PATH,
    },
};
use flume::{Receiver, Sender};
use helium::{
    bf16,
    loss::cross_entropy_loss,
    modules::batch_norm::ForwardMode,
    optimizer::{sgd::Sgd, Optimizer},
    DataType, Device, Module, Tensor,
};
use image::RgbImage;
use parking_lot::RwLock;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::{
    sync::{Arc, Barrier},
    thread,
    thread::available_parallelism,
};
use uuid::Uuid;

mod preview;

static IMAGE_SIZE: usize = 256;

#[derive(Clone)]
struct Item {
    image: Arc<RgbImage>,
    label: usize,
}

impl Item {
    pub fn from_db(db: &Database, id: Uuid) -> anyhow::Result<Self> {
        let metadata = db.get_metadata(id)?;
        let bytes = db.get_encoded_image_data(id)?;
        let image: RgbImage = image::load_from_memory(bytes.value())?.into();
        Ok(Self {
            image: Arc::new(image),
            label: metadata
                .label
                .try_into()
                .expect("label out of bounds; perhaps this is a test item?"),
        })
    }
}

#[derive(Clone)]
struct Batch {
    items: Vec<Item>,
    image_tensor: Tensor<4>,
    /// One-hot encoding
    label_tensor: Tensor<2>,
}

impl Batch {
    pub fn new(items: Vec<Item>, device: Device) -> Self {
        let image_data = items
            .iter()
            .flat_map(|item| {
                item.image
                    .iter()
                    .copied()
                    .map(|x| ((x as f32 / u8::MAX as f32) - 0.5) * 2.0)
            })
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let label_data = items
            .iter()
            .flat_map(|item| {
                let mut one_hot = [0.0f32; NUM_CLASSES];
                one_hot[item.label] = 1.0;
                one_hot
            })
            .map(bf16::from_f32)
            .collect::<Vec<_>>();

        let image_tensor =
            Tensor::from_slice(image_data, [items.len(), IMAGE_SIZE, IMAGE_SIZE, 3], device);
        let label_tensor = Tensor::from_slice(label_data, [items.len(), NUM_CLASSES], device);

        Self {
            items,
            image_tensor,
            label_tensor,
        }
    }
}

enum ItemOrEpochEnd {
    Item(Item),
    EpochEnd,
}

fn start_data_loader_threads(
    db: &Arc<Database>,
    split: Split,
    rng: &mut impl Rng,
) -> anyhow::Result<flume::Receiver<ItemOrEpochEnd>> {
    let num_threads = available_parallelism()?;
    let (tx, rx) = flume::bounded(num_threads.get() * 4);

    let unshuffled_ids = db.list_image_ids(split)?;

    tracing::info!("Found {} items for split {split:?}", unshuffled_ids.len());

    // Shuffled by leader thread at start of every epoch
    let shuffled_ids = Arc::new(RwLock::new(unshuffled_ids));

    fn run_thread(
        thread_index: usize,
        num_threads: usize,
        shuffled_ids: &RwLock<Vec<Uuid>>,
        db: &Database,
        tx: Sender<ItemOrEpochEnd>,
        epoch_sync: &Barrier,
        seed: u128,
        split: Split,
    ) {
        let mut rng = Pcg64Mcg::from_seed(seed.to_le_bytes());
        loop {
            epoch_sync.wait();

            if thread_index == 0 {
                shuffled_ids.write().shuffle(&mut rng);
            }

            epoch_sync.wait();

            let shuffled_ids = shuffled_ids.read();
            for i in (thread_index..shuffled_ids.len()).step_by(num_threads) {
                let mut item = Item::from_db(db, shuffled_ids[i]).expect("failed to load item");

                if split == Split::Training {
                    item.image = Arc::new(augment_training_image(&item.image, &mut rng));
                }

                if tx.send(ItemOrEpochEnd::Item(item)).is_err() {
                    return;
                }
            }

            epoch_sync.wait();
            if thread_index == 0 {
                tx.send(ItemOrEpochEnd::EpochEnd).unwrap();
            }
        }
    }

    let epoch_sync = Arc::new(Barrier::new(num_threads.get()));
    let seed: u128 = rng.gen();
    for thread_index in 0..num_threads.get() {
        let tx = tx.clone();
        let shuffled_ids = shuffled_ids.clone();
        let db = db.clone();
        let epoch_sync = epoch_sync.clone();
        thread::Builder::new()
            .name(format!("{split:?}-loader-{thread_index}"))
            .spawn(move || {
                run_thread(
                    thread_index,
                    num_threads.get(),
                    &shuffled_ids,
                    &db,
                    tx,
                    &epoch_sync,
                    seed,
                    split,
                );
            })?;
    }

    Ok(rx)
}

fn recv_batch(receiver: &Receiver<ItemOrEpochEnd>, batch_size: usize) -> Option<Vec<Item>> {
    let mut items = Vec::new();
    for _ in 0..batch_size {
        match receiver.recv().unwrap() {
            ItemOrEpochEnd::Item(item) => items.push(item),
            ItemOrEpochEnd::EpochEnd => return None,
        }
    }
    Some(items)
}

fn start_batcher_thread(
    receiver: Receiver<ItemOrEpochEnd>,
    batch_size: usize,
    device: Device,
) -> anyhow::Result<flume::Receiver<Batch>> {
    let (tx, rx) = flume::bounded(2);
    thread::Builder::new()
        .name("batcher".to_owned())
        .spawn(move || loop {
            match recv_batch(&receiver, batch_size) {
                Some(items) => {
                    let batch = Batch::new(items, device);
                    if tx.send(batch).is_err() {
                        return;
                    }
                }
                None => return,
            }
        })?;
    Ok(rx)
}

fn top_1_accuracy(batch: &Batch, probs: &[f32]) -> f32 {
    let mut num_correct = 0;
    for (item, probs) in batch.items.iter().zip(probs.chunks_exact(NUM_CLASSES)) {
        let top1 = probs
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        if top1 == item.label {
            num_correct += 1;
        }
    }
    num_correct as f32 / batch.items.len() as f32 * 100.0
}

pub fn train() -> anyhow::Result<()> {
    let device = Device::Cuda(0);
    let mut rng = Pcg64Mcg::seed_from_u64(66);

    let mut model = Model::new(&mut rng, device);
    let mut optimizer = Sgd::new_with_momentum(0.9);

    let db = Arc::new(Database::open_read_only(DEFAULT_DB_PATH)?);

    let items_training = start_data_loader_threads(&db, Split::Training, &mut rng)?;
    let items_validation = start_data_loader_threads(&db, Split::Validation, &mut rng)?;

    let num_epochs = 200;
    let batch_size = 64;
    let mut lr = 1e-1;
    let lr_gamma = 0.99;

    let previewer = Previewer::new()?;

    thread::scope(|s| {
        for epoch in 0..num_epochs {
            tracing::info!(epoch, "begin epoch");
            let mut batch_idx = 0;

            let batches_training =
                start_batcher_thread(items_training.clone(), batch_size, device)?;

            struct TrainingResult {
                loss: Tensor<1>,
                probs: Tensor<2>,
                batch: Batch,
                index: usize,
            }

            let (results_tx, results_rx) = flume::bounded::<TrainingResult>(2);
            let previewer = &previewer;
            let mut rng_fork = Pcg64Mcg::from_rng(&mut rng).unwrap();
            let results_thread = s.spawn(move || {
                for TrainingResult {
                    loss,
                    probs,
                    batch,
                    index,
                } in results_rx
                {
                    let probs = probs.to_vec::<f32>();
                    let loss = loss.to_scalar::<f32>();
                    let accuracy = top_1_accuracy(&batch, &probs);

                    // Sample some examples
                    for ((i, item), probs) in batch
                        .items
                        .iter()
                        .enumerate()
                        .zip(probs.chunks_exact(NUM_CLASSES))
                    {
                        if rng_fork.gen_bool(0.005) {
                            previewer.add_example(Example {
                                epoch,
                                iteration: index + i,
                                image: Arc::clone(&item.image),
                                predictions: probs.try_into().expect("wrong slice size"),
                                label: item.label,
                            });
                        }
                    }

                    previewer.add_train_iteration(loss, accuracy);
                    tracing::debug!(epoch, loss, accuracy, "step");
                }
            });

            for batch in batches_training {
                let logits = model
                    .forward(&batch.image_tensor, ForwardMode::Train)
                    .to_data_type(DataType::F32);
                let probs = logits.log_softmax().exp();
                let loss = cross_entropy_loss(logits, batch.label_tensor.clone());

                loss.async_start_eval();

                let grads = loss.backward();
                optimizer.step(&mut model, &grads, lr);
                model.async_start_eval();

                results_tx
                    .send(TrainingResult {
                        loss,
                        probs,
                        batch,
                        index: batch_idx * batch_size,
                    })
                    .unwrap();
                batch_idx += 1;
            }
            drop(results_tx);
            results_thread.join().unwrap();

            let batches_validation =
                start_batcher_thread(items_validation.clone(), batch_size, device)?;
            tracing::info!(epoch, "begin validation");
            for batch in batches_validation {
                let logits = model
                    .forward(&batch.image_tensor, ForwardMode::Train)
                    .to_data_type(DataType::F32);
                let probs = logits.log_softmax().exp();
                let loss_tensor = cross_entropy_loss(logits, batch.label_tensor.clone());

                let loss = loss_tensor.to_scalar::<f32>();
                let probs = probs.to_vec::<f32>();
                let accuracy = top_1_accuracy(&batch, &probs);

                previewer.add_validation_iteration(loss, accuracy);
                tracing::debug!(loss, accuracy, "validation batch");
                batch_idx += 1;
            }
            lr *= lr_gamma;
        }
        Result::<_, anyhow::Error>::Ok(())
    })?;

    Ok(())
}
