# scripts/gan/

GAN (Generative Adversarial Network) training and post-processing for synthetic data augmentation.

## Files

| File | Description |
|------|-------------|
| `train_gan.py` | Train CTGAN/TVAE on parametric data (requires GPU / Colab) |
| `fix_gan_output.py` | Post-process GAN output (fix IDs, recompute round amounts) |

## Usage

```bash
# Train GAN (requires GPU — do NOT run on dev server)
python3 scripts/gan/train_gan.py --model ctgan --rows 500000 --epochs 300 --batch-size 2000

# Fix GAN output
python3 scripts/gan/fix_gan_output.py
```

## Output

- Raw: `data/generated/gan/pantau_gan_ctgan_500k.csv`
- Cleaned: `data/generated/gan/pantau_gan_ctgan.csv`
