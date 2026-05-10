import time
from pathlib import Path
from utils.test_perplexity import PerplexityTester
import shutil

def run_benchmark():
    # Create mock dataset directory
    test_dir = Path("mock_data_for_benchmark")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Create 5000 mock dataset subdirectories with test.txt
    print("Creating mock datasets...")
    for i in range(5000):
        ds_dir = test_dir / f"dataset_{i}"
        ds_dir.mkdir()
        test_file = ds_dir / "test.txt"
        test_file.write_text("Hello World!")

    print("Running benchmark...")

    # We need dummy values for binary paths and model path
    dummy_bin = test_dir / "dummy_bin"
    dummy_bin.touch()
    dummy_model = test_dir / "dummy_model"
    dummy_model.touch()

    # Patch print to suppress output
    import builtins
    original_print = builtins.print
    builtins.print = lambda *args, **kwargs: None

    try:
        tester = PerplexityTester(
            model_path=dummy_model,
            llama_perplexity_bin=dummy_bin,
            quantize_bin=dummy_bin,
            data_dir=test_dir,
            output_dir=test_dir / "output"
        )

        start_time = time.time()
        for _ in range(5):  # Run a few times to get a stable average
            tester.find_datasets()
        end_time = time.time()
    finally:
        builtins.print = original_print
        shutil.rmtree(test_dir)

    avg_time = (end_time - start_time) / 5
    print(f"Average time per find_datasets() call: {avg_time:.4f} seconds")

if __name__ == '__main__':
    run_benchmark()
