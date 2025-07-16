use birefnet_burn::{BiRefNetDataset, ModelConfig};
use burn::backend::ndarray::NdArray;
use burn::data::Dataset;

type TestBackend = NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test configuration
    let mut config = ModelConfig::new();

    // Set the dataset path to our test dataset
    config.path.data_root_dir = std::path::PathBuf::from("dataset");

    // Create device
    let device = Default::default();

    // Try to create a dataset with the test data
    println!("Attempting to create dataset with test1 split...");
    let dataset = BiRefNetDataset::<TestBackend>::new(&config, "test1", &device)?;

    println!("Dataset created successfully!");
    println!("Dataset length: {}", dataset.len());

    // Try to get the first item
    if let Some(item) = dataset.get(0) {
        println!("First item loaded successfully!");
        println!("Image shape: {:?}", item.image.shape());
        println!("Mask shape: {:?}", item.mask.shape());
    } else {
        println!("No items in dataset or failed to load first item");
        return Ok(());
    }

    // Test loading multiple items
    println!("Testing loading multiple items...");
    let test_indices = [0, 1, 2, 10, 50, 100];

    for &idx in &test_indices {
        if idx < dataset.len() {
            match dataset.get(idx) {
                Some(item) => {
                    println!(
                        "Item {}: Image shape: {:?}, Mask shape: {:?}",
                        idx,
                        item.image.shape(),
                        item.mask.shape()
                    );
                }
                None => {
                    println!("Failed to load item {}", idx);
                }
            }
        }
    }

    // Test iteration over first few items
    println!("Testing iteration over first 5 items...");
    for i in 0..std::cmp::min(5, dataset.len()) {
        if let Some(item) = dataset.get(i) {
            println!(
                "Item {}: Image shape: {:?}, Mask shape: {:?}",
                i,
                item.image.shape(),
                item.mask.shape()
            );
        } else {
            println!("Failed to load item {}", i);
        }
    }

    Ok(())
}
