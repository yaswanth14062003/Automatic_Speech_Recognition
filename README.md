# Telugu ASR Cross-Validation Study

## Overview

This repository benchmarks two leading speech-to-text models on Telugu language data using $k$-fold cross-validation ($k=2,3,4,5$):

- **Wav2Vec2.0 XLSR-53-large** (Meta/Facebook)
- **Whisper-small** (OpenAI)

The aim is to evaluate how well these models transcribe Telugu speech, using rigorous cross-validation to assess their generalization and reliability.

---

## Dataset

- **Source:** [OpenSLR]
- **Preprocessing:** Audio is resampled to 16kHz, normalized, and transcripts are cleaned (punctuation removed, lowercased, etc.).
- **Splitting:** For each $k$-fold experiment, the dataset is divided into $k$ stratified folds to ensure balanced speaker and utterance distribution.

---

## Models Compared

### Wav2Vec2.0 XLSR-53-large

- Transformer-based, pre-trained on 53 languages.
- Fine-tuned on Telugu data with CTC loss.
- Uses a character-level tokenizer from training transcripts.
- Training: AdamW optimizer, learning rate scheduler, batch size 8-16, 10-20 epochs, early stopping on validation WER.

### Whisper-small

- Encoder-decoder transformer trained on multilingual data.
- Fine-tuned for Telugu ASR using Hugging Face Transformers.
- Built-in Whisper tokenizer set to Telugu.
- Training: AdamW optimizer, batch size 8-16, 10-20 epochs, early stopping on validation WER.

---

## Cross-Validation Procedure

For each $k$ in {2, 3, 4, 5}:
- Split data into $k$ folds.
- For each fold:
  - Train on $k-1$ folds, validate on the remaining fold.
  - Log training/validation loss, WER, and CER.
- Average results across all folds.

Additionally, each model is trained on the full dataset without cross-validation for baseline comparison.

---

## Metrics

- **WER (Word Error Rate):** Percentage of words incorrectly predicted.
- **CER (Character Error Rate):** Percentage of characters incorrectly predicted.
- **Evaluation Loss:** CTC or cross-entropy loss on the validation set.

---

## Robustness to Noisy Data

The models were also evaluated on a noisy test set containing background sounds, distortions, and varied recording quality. The best model (Wav2Vec2.0 XLSR-53-large) showed only a 7% drop in accuracy compared to clean data, demonstrating strong robustness to real-world audio conditions.

---

## Usage

1. **Install dependencies:**
   ```bash
   pip install torch transformers datasets numpy matplotlib jupyter
   ```
2. **Prepare the data:**
   - Place audio files and transcripts in the `data/` directory.
   - Adjust file paths in notebooks if needed.

3. **Run experiments:**
   - Open the relevant notebook (`wav2vec_k*.ipynb` or `whisper_k*.ipynb`) in JupyterLab.
   - Run all cells to train and evaluate the model for the chosen $k$.

4. **View results:**
   - Training logs, tables, and plots (loss, WER, CER) are generated for each fold.
   - The bar chart `base_vs_best.pdf` compares base and best-fold errors.

---

## Example: Running a 5-Fold CV Experiment

1. Open `wav2vec_k5.ipynb` or `whisper_k5.ipynb`.
2. Run all cells.
3. Outputs include per-fold WER/CER, average metrics, and plots for training/validation loss and error rates.

---

## Key Results

| Model                      | Base WER/CER | Best CV WER/CER ($k=5$) |
|----------------------------|--------------|-------------------------|
| Wav2Vec2.0 XLSR-53-large   | 0.2567/0.0453| 0.2362/0.0412           |
| Whisper-small              | 0.2776/0.0515| 0.2867/0.0548           |

**Observation:** Wav2Vec2.0 outperforms Whisper-small, especially after cross-validation. See `base_vs_best.pdf` for a visual comparison.

---

## Directory Structure

```
Telugu_ASR_K-Fold/
├── data/                # Audio files and transcripts
├── wav2vec_k2.ipynb     # 2-fold CV for Wav2Vec2.0
├── wav2vec_k3.ipynb     # 3-fold CV for Wav2Vec2.0
├── wav2vec_k4.ipynb     # 4-fold CV for Wav2Vec2.0
├── wav2vec_k5.ipynb     # 5-fold CV for Wav2Vec2.0
├── whisper_k2.ipynb     # 2-fold CV for Whisper
├── whisper_k3.ipynb     # 3-fold CV for Whisper
├── whisper_k4.ipynb     # 4-fold CV for Whisper
├── whisper_k5.ipynb     # 5-fold CV for Whisper
├── base_vs_best.pdf     # Bar chart comparing results
└── README.md
```

---

## Contributing

Contributions are welcome! You can:
- Suggest new models or folds
- Report issues or bugs
- Propose improvements to data, training, or evaluation
- Add new metrics or visualizations

Open an issue or submit a pull request to participate.

---

## References

- [Wav2Vec2.0 Paper](https://arxiv.org/abs/2006.11477)
- [Whisper Paper](https://cdn.openai.com/papers/whisper.pdf)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
