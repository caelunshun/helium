use crate::model::NUM_CLASSES;
use image::RgbImage;
use indexmap::IndexMap;
use parking_lot::Mutex;
use plotters::prelude::*;
use rayon::prelude::*;
use serde::Serialize;
use std::{
    array, fs,
    fs::File,
    io::BufWriter,
    mem,
    sync::{Arc, OnceLock},
    thread,
};

static EPOCHS_DIR: &str = "preview/epochs";
static LOSS_PLOT_FILE: &str = "preview/loss.png";
static ACCURACY_PLOT_FILE: &str = "preview/accuracy.png";

pub struct Previewer {
    train_data: Mutex<LossData>,
    validation_data: Mutex<LossData>,
    example_queue: Mutex<Vec<Example>>,
}

impl Previewer {
    pub fn new() -> anyhow::Result<Arc<Self>> {
        fs::create_dir_all(EPOCHS_DIR)?;
        let previewer = Arc::new(Previewer {
            train_data: Mutex::new(LossData::default()),
            validation_data: Mutex::new(LossData::default()),
            example_queue: Mutex::new(Vec::new()),
        });

        let previewer2 = previewer.clone();
        thread::Builder::new()
            .name("previewer".to_owned())
            .spawn(move || {
                run_processor_thread(&previewer2);
            })?;

        Ok(previewer)
    }

    pub fn add_train_iteration(&self, loss: f32, accuracy: f32) {
        let mut data = self.train_data.lock();
        data.loss.push(loss);
        data.accuracy.push(accuracy);
    }

    pub fn add_validation_iteration(&self, loss: f32, accuracy: f32) {
        let mut data = self.validation_data.lock();
        data.loss.push(loss);
        data.accuracy.push(accuracy);
    }

    pub fn add_example(&self, example: Example) {
        self.example_queue.lock().push(example);
    }
}

fn run_processor_thread(previewer: &Previewer) {
    loop {
        // Emit plots and preview examples in parallel
        rayon::join(
            || {
                let train_data = previewer.train_data.lock().to_smoothed();
                let validation_data = previewer.validation_data.lock().to_smoothed();

                generate_plot(
                    "loss",
                    &train_data.loss,
                    &validation_data.loss,
                    LOSS_PLOT_FILE,
                )
                .expect("failed to generate loss plot");
                generate_plot(
                    "accuracy",
                    &train_data.accuracy,
                    &validation_data.accuracy,
                    ACCURACY_PLOT_FILE,
                )
                .expect("failed to generate accuracy plot");
            },
            || {
                let examples = mem::take(&mut *previewer.example_queue.lock());
                examples
                    .into_par_iter()
                    .for_each(|ex| ex.write_to_disk().expect("failed to save example to disk"));
            },
        );

        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

#[derive(Debug, Clone, Default)]
pub struct LossData {
    loss: Vec<f32>,
    accuracy: Vec<f32>,
}

impl LossData {
    pub fn to_smoothed(&self) -> LossData {
        let mut target = LossData {
            loss: Vec::with_capacity(self.loss.len()),
            accuracy: Vec::with_capacity(self.accuracy.len()),
        };
        let mut moving_average_loss = None;
        let mut moving_average_accuracy = None;
        for (loss, accuracy) in self.loss.iter().copied().zip(self.accuracy.iter().copied()) {
            moving_average_loss = Some(match moving_average_loss {
                Some(avg) => 0.8 * avg + 0.2 * loss,
                None => loss,
            });
            moving_average_accuracy = Some(match moving_average_accuracy {
                Some(avg) => 0.8 * avg + 0.2 * accuracy,
                None => accuracy,
            });
            target.loss.push(moving_average_loss.unwrap());
            target.accuracy.push(moving_average_accuracy.unwrap());
        }
        target
    }
}

pub struct Example {
    pub epoch: usize,
    pub iteration: usize,
    pub image: Arc<RgbImage>,
    pub predictions: [f32; NUM_CLASSES],
    pub label: usize,
}

impl Example {
    fn write_to_disk(&self) -> anyhow::Result<()> {
        let path = format!("{EPOCHS_DIR}/{:04}/{:06}", self.epoch, self.iteration);

        if let Some(parent) = std::path::Path::new(&path).parent() {
            fs::create_dir_all(parent).ok();
        }

        self.image.save(format!("{path}.png"))?;

        #[derive(Serialize)]
        struct Metadata {
            top_predictions: IndexMap<&'static str, String>,
            actual: &'static str,
        }

        let mut predictions: [(f32, &'static str); NUM_CLASSES] =
            array::from_fn(|i| (self.predictions[i], name_for_label_index(i)));
        predictions.sort_unstable_by(|(a, _), (b, _)| a.total_cmp(b));
        let mut top_predictions = IndexMap::new();
        for (prob, label) in predictions.iter().rev().take(5) {
            top_predictions.insert(*label, format!("{:.2}%", *prob as f64 * 100.0));
        }

        let metadata = Metadata {
            top_predictions,
            actual: name_for_label_index(self.label),
        };

        serde_json::to_writer_pretty(
            BufWriter::new(File::create(format!("{path}.meta.json"))?),
            &metadata,
        )?;

        Ok(())
    }
}

fn generate_plot(
    name: &str,
    data_train: &[f32],
    data_valid: &[f32],
    output_path: &str,
) -> anyhow::Result<()> {
    let root = BitMapBackend::new(output_path, (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_len = data_train.len() as f32;
    let y_min = data_train
        .iter()
        .chain(data_valid.iter())
        .copied()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);
    let y_max = data_train
        .iter()
        .chain(data_valid.iter())
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 24).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0f32..x_len, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            data_train
                .iter()
                .copied()
                .enumerate()
                .map(|(i, y)| (i as f32, y)),
            &BLACK,
        ))?
        .label(format!("{name} (train)"));
    chart
        .draw_series(LineSeries::new(
            data_valid.iter().copied().enumerate().map(|(i, y)| {
                (
                    i as f32 / data_valid.len() as f32 * data_train.len() as f32,
                    y,
                )
            }),
            &RED,
        ))?
        .label(format!("{name} (validation)"));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn name_for_label_index(label: usize) -> &'static str {
    static MAPPING: OnceLock<Vec<String>> = OnceLock::new();

    &MAPPING.get_or_init(|| {
        static URL: &str = "https://raw.githubusercontent.com/formigone/tf-imagenet/refs/heads/master/LOC_synset_mapping.txt";
        let data = ureq::get(URL).call().expect("failed to fetch labels from web")
            .into_string().expect("failed to fetch labels from web");

        let mut names = Vec::new();
        for line in data.lines() {
            let line = line.trim();
            if !line.is_empty() {
                let start = line.char_indices()
                    .find_map(|(i, c)| if c == ' ' { Some(i) } else { None })
                    .unwrap() + 1;
                names.push(line[start..].to_owned());
            }
        }
        names
    })
    [label]
}
