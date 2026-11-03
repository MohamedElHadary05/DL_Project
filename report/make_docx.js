// Build the Milestone 1 report as a Word document (report/report.docx) from the
// figures in report/figures/. Run from the project root:  node report/make_docx.js
// Text markup: **bold**, *italic*, `code`, x_{sub}, x^{sup}, {fig:key} / {tab:key} / {sec:key}.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, BorderStyle, AlignmentType, HeadingLevel, ShadingType, LevelFormat,
  TableOfContents, PageBreak, Footer, PageNumber, VerticalAlign,
} = require("docx");

const FIG = path.join(__dirname, "figures");
const OUT = path.join(__dirname, "report.docx");
const PAGE_W = 11906, MARGIN = 1247;           // A4, 2.2 cm margins (DXA)
const TEXT_W = PAGE_W - 2 * MARGIN;            // 9412 DXA
const TEXT_PX = Math.round(TEXT_W / 15);       // image width budget at 96 dpi
const FONT = "Calibri";

// ---------------------------------------------------------------------------
// Report content. Each block: [kind, ...args]
// ---------------------------------------------------------------------------
const CONTENT = [
  ["h1", "Introduction", "intro"],
  ["p", "The project builds one visual encoder and reuses it for three tasks: (1) image classification, (2) colorization of grayscale images and (3) caption generation. This report covers Milestone 1: understanding and preparing the data, and training the classification encoder."],
  ["p", "**Dataset choice.** We use the default dataset, CUB-200-2011 [1] with the captions of Reed et al. [2], loaded from Hugging Face (`Multimodal-Fatima/CUB_train` and `CUB_test`). It fits all three tasks: every image has a species label (classification), the photos are of good quality and can be turned into grayscale/colour pairs (colorization), and every image has ten human-written descriptions (captioning). At about 1.1 GB it also fits the free Kaggle T4 GPU we trained on."],
  ["p", "**Why it is hard.** CUB is a *fine-grained* benchmark: 200 species, many of which differ only in small details (a wing bar, the colour of a crown), with only about 30 training images per species. This makes it a good test of how much a model can learn from little data, and of the value of transfer learning."],
  ["p", "**Code.** All final code is pure Python under `src/` and is run through one command-line interface (`python -m src.cli`) with the sub-commands `train`, `evaluate`, `predict`, `tune`, `explain` and `compare`. Data, models, training, evaluation and visualization live in separate modules; models are created by a registry (`src/models/factory.py`), so the trainer never names a specific architecture. The exact commands are in the `README.md`."],

  ["h1", "Data: Analysis, Cleaning and Preprocessing", "data"],
  ["h2", "Overview"],
  ["table", "data", "Dataset overview (computed by `src/visualization/eda_plots.py` and `eda_extra.py`).",
    ["", "Train", "Test"],
    [["Images", "5,994", "5,794"],
     ["Classes", "200", "200"],
     ["Images per class (min / mean / max)", "29 / 29.97 / 30", "11 / 28.97 / 30"],
     ["Grayscale images (mode L)", "4", "4"],
     ["Corrupted / unreadable images", "0", "0"],
     ["Median image size (W × H)", "500 × 375", "500 × 375"],
     ["Captions per image", "10 (all)", "10 (all)"],
     ["Caption length in words (mean / median / max)", "15.2 / 14 / 65", "15.2 / 14 / 62"]],
    [5212, 2100, 2100]],
  ["fig", "samples", "samples_train.jpg", 0.95, "Training samples with their species labels."],

  ["h2", "Class distribution"],
  ["p", "The training split is almost perfectly balanced: 194 species have 30 images and 6 have 29 ({fig:classdist}). The test split is less even (11 to 30 images per class, 52 classes below 30), but no class is missing. Because the training classes are balanced we use plain cross-entropy without class weights, and we report *macro*-averaged precision, recall and F1 so that every species counts equally in the evaluation."],
  ["fig2", "classdist", [["class_dist_train.png", "Train."], ["class_dist_test.png", "Test."]], "Number of images per class (histogram over the 200 classes)."],

  ["h2", "Image sizes and quality"],
  ["p", "Images are at most 500 pixels on their longer side; the median is 500×375 (aspect ratio 1.33), and the smallest is about 120 pixels on each side ({fig:imgstats}). Since the models use 128 or 224 pixel crops, every image has enough resolution. All 11,788 images decode without errors. Eight images (four per split) are grayscale; the dataset class converts every image to RGB, which handles them without dropping data."],
  ["p", "We also measured brightness, contrast and sharpness (variance of the Laplacian) for every image ({fig:quality}). The distributions are wide but contain no unusable outliers. The eight “blurriest” images ({fig:blur}) are birds in flight or slightly out of focus, but several are sharp birds against plain sky or water: the Laplacian measures texture, so smooth backgrounds also score as blurry. All of these are realistic inputs, so we keep them."],
  ["fig", "imgstats", "image_stats_train.png", 0.8, "Image width vs. height and aspect ratio (train)."],
  ["fig", "quality", "quality_train.png", 1.0, "Brightness, contrast and blur (Laplacian variance) of the training images; the last bin of each histogram collects values above the 99th percentile."],
  ["fig", "blur", "blurriest_train.jpg", 0.75, "The eight blurriest training images."],

  ["h2", "Duplicates and train/test leakage"],
  ["p", "We hashed every image twice: an exact MD5 hash of the pixels and a 64-bit perceptual difference hash (dHash) that survives resizing and re-compression. We found one near-duplicate pair across the splits: test image 661 matches training image 774. We confirmed it visually and removed the test image, so the test set we report on has **5,793** images. Leaving it in would leak a training image into the test score."],

  ["h2", "Caption analysis"],
  ["p", "The `description` field holds the ten captions of an image as one string separated by newlines. Our findings on the training split (59,940 captions):"],
  ["ul", [
    "Every image has exactly ten captions.",
    "Captions are short: 15.2 words on average, 95% have at most 24 words ({fig:captions}, left). A few outliers are 1–2 words long (27 captions) or longer than 40 words (106 captions).",
    "The raw vocabulary has 4,077 words, of which 1,875 occur only once. Many of these rare words are typos: 989 rare words have a close match among the common words (e.g. *agrayish* vs. *grayish*).",
    "27 captions contain non-ASCII characters (mostly typographic quotes and accents).",
    "9,463 captions are exact duplicates of another caption, such as generic descriptions (“this bird is black”) written for several birds.",
    "After function words, the most frequent words are colours and body parts (*black*, *white*, *brown*, *beak*, *belly*, *wings*, *yellow*; {fig:captions}, right), which shows that captions describe exactly the visual cues a classifier needs.",
  ]],
  ["fig2", "captions", [["caption_len_train.png", "Caption length in words."], ["top_words_train.png", "Most frequent words."]], "Caption statistics (train)."],

  ["h2", "Caption cleaning"],
  ["p", "Captions are cleaned in `src/data/preprocessing.py` and `src/data/captions.py`:"],
  ["ol", [
    "Split the description on newlines into ten captions.",
    "Replace non-ASCII characters with spaces (replacing instead of deleting avoids gluing two words together), lower-case, keep only letters, digits and basic punctuation, collapse repeated whitespace.",
    "Drop captions shorter than 3 or longer than 40 words: they are either uninformative or rambling outliers. This removes 111 of 59,940 training captions (0.19%).",
    "Tokenize into words and build the vocabulary on the *training split only* (no validation or test words), keeping words seen at least 5 times. Rarer words, mostly typos, map to `<unk>`.",
  ]],
  ["p", "The resulting vocabulary has 1,127 tokens (including `<pad>`, `<start>`, `<end>`, `<unk>`), and only 0.53% of training words become `<unk>`. Exact duplicate captions are kept: each one is a valid description of its own image. The captions are used in Milestone 2."],

  ["h2", "Train / validation / test split"],
  ["p", "The official training split is divided into training (80%) and validation (20%) with a stratified split (scikit-learn, seed 42), so every species keeps its proportion: **4,795 training and 1,199 validation images**. The validation set chooses the best epoch and the hyperparameters; the test set (5,793 images) is used only once per final model. A fingerprint of the validation indices is printed by `python -m src.check_env` so all team members can confirm they use identical splits."],

  ["h2", "Normalization and augmentation"],
  ["p", "Images are normalized with the ImageNet mean and standard deviation. The pretrained ResNet requires these statistics, and we use the same ones for the scratch model so both see identical inputs."],
  ["p", "The training augmentation (`src/data/transforms.py`, {fig:aug}) is designed around what does *not* change a bird's species:"],
  ["ul", [
    "**RandomResizedCrop** (scale 0.6–1.0): birds appear at different sizes and positions; the minimum scale of 0.6 keeps most of the bird in view so the label stays correct.",
    "**Horizontal flip**: a bird facing left is the same species. Vertical flips and large rotations are not used because birds are almost always upright.",
    "**Colour jitter** on brightness and contrast (0.2) and a little saturation (0.1): covers different lighting. **Hue is not changed**, because colour is one of the most important cues for telling species apart (e.g. Blue Grosbeak vs. Indigo Bunting).",
  ]],
  ["p", "Validation and test images are resized to 1.15× the crop size and centre-cropped, with no randomness."],
  ["fig", "aug", "augmentation_examples.jpg", 0.75, "Four random augmentations of each of four training images."],

  ["h1", "Models", "models"],
  ["p", "All models share one structure: an *encoder* that turns an image into a spatial feature map, followed by a classification *head* (global average pooling, dropout 0.3, and a linear layer to 200 classes). The encoder is the part that Milestone 2 reuses for colorization (as the downsampling half of a U-Net) and captioning (frozen features fed to a decoder)."],
  ["h2", "CNN from scratch (baseline)"],
  ["p", "The scratch encoder (`src/models/scratch_cnn.py`) has five blocks. Each block is two 3×3 convolutions, each followed by batch normalization [5] and ReLU, then 2×2 max pooling. The channel widths are 32, 64, 128, 256 and 512, so a 128×128 image becomes a 512×4×4 feature map. The whole classifier has 4.82 M parameters, all randomly initialised. We use 128-pixel inputs to keep training fast; this is a deliberate trade-off against detail."],
  ["h2", "ResNet-18 transfer learning"],
  ["p", "The transfer encoder (`src/models/pretrained.py`) is a ResNet-18 [3] with ImageNet weights [4] (`IMAGENET1K_V1` from torchvision), without its average pooling and 1000-class layer. With 224-pixel inputs it produces a 512×7×7 feature map. It exposes the same interface as the scratch encoder, so the same head, trainer and Grad-CAM code work unchanged. We use it in two ways:"],
  ["ul", [
    "**Frozen (feature extraction):** the encoder weights are fixed and only the head trains (102,600 parameters). Batch-normalization statistics are frozen too, so the pretrained features do not drift.",
    "**Fine-tuned:** all 11.28 M parameters train. The pretrained encoder uses a 10× smaller learning rate (0.0001) than the new head (0.001), so it adapts to birds without losing its ImageNet features.",
  ]],
  ["h2", "Training setup"],
  ["p", "All models are implemented in PyTorch [14] and trained by the same loop (`src/training/trainer.py`): AdamW [6], a cosine learning-rate schedule [7] down to zero over the run, cross-entropy with label smoothing 0.1 [8], and mixed precision on the GPU. After every epoch the model is evaluated on the validation set, and the checkpoint with the best validation accuracy is kept. All runs use seed 42 and were trained on a Kaggle NVIDIA T4 GPU. Configurations are YAML files in `configs/`."],
  ["table", "configs", "Training configurations of the final models.",
    ["Model", "Input", "Batch", "LR (head / encoder)", "Weight decay", "Epochs", "Trainable params"],
    [["Scratch CNN (tuned)", "128", "64", "0.0003 / 0.0003", "0.005", "60", "4.82 M"],
     ["ResNet-18 frozen", "224", "32", "0.001 / —", "0.0001", "15", "0.10 M"],
     ["ResNet-18 fine-tuned", "224", "32", "0.001 / 0.0001", "0.0001", "15", "11.28 M"]],
    [2212, 800, 800, 1800, 1300, 900, 1600]],

  ["h1", "Experiments", "experiments"],
  ["p", "{tab:runs} lists every classification run, and {fig:valcurves} compares their validation curves."],
  ["table", "runs", "Classification runs. “Gap” is training minus validation accuracy at the best epoch; training accuracy is measured on augmented images.",
    ["Run", "Best epoch", "Val acc.", "Val top-5", "Val F1", "Train acc.", "Gap", "Minutes"],
    [["Scratch CNN (default)", "29 / 30", "31.9%", "62.2%", "0.306", "50.4%", "18.5", "5.8"],
     ["Scratch CNN (tuned)", "59 / 60", "47.4%", "73.0%", "0.465", "96.2%", "48.8", "11.6"],
     ["ResNet-18 frozen", "15 / 15", "60.7%", "87.2%", "0.601", "77.6%", "16.9", "3.9"],
     ["ResNet-18 fine-tuned", "14 / 15", "72.4%", "92.1%", "0.720", "99.1%", "26.7", "4.1"]],
    [2212, 1000, 950, 1000, 850, 1100, 1300, 1000]],
  ["fig", "valcurves", "val_curves_all.png", 0.8, "Validation accuracy per epoch of the four classification runs."],

  ["h2", "Baseline vs. tuned scratch CNN"],
  ["p", "The default scratch run (learning rate 0.001, 30 epochs) reached 31.9%. Its validation accuracy was still rising when the cosine schedule ended. Tuning ({sec:tuning}) moved the learning rate to 0.0003, which gave 39.2% in 30 epochs, and training the tuned model for 60 epochs gave **47.4%**: +15.5 points over the baseline. The longer run converges (validation accuracy is flat from about epoch 46) but overfits heavily: the model fits 96% of the (augmented) training images while validation accuracy stays at 47% ({fig:curves_scratch}). With about 24 training images per species, the network can memorise the training set."],
  ["h2", "Scratch vs. transfer learning"],
  ["p", "Transfer learning dominates. After a single epoch, the fine-tuned ResNet-18 (43.3% validation accuracy) is almost level with the scratch CNN after 60 epochs, and it finishes at 72.4% in 15 epochs and about 4 minutes."],
  ["ul", [
    "**Frozen** features already reach 60.7% with only a linear head trained. ImageNet features locate the bird and its parts well, but they are not specialised for distinguishing similar species, so the frozen model plateaus ({fig:curves_resnet}, left).",
    "**Fine-tuning** lets the encoder adapt to fine-grained details and adds 11.7 points. It also memorises the training set (99% training accuracy); validation loss keeps decreasing, but the 27-point gap between training and validation accuracy shows that it overfits as well.",
  ]],
  ["fig", "curves_scratch", "scratch_tuned_curves.png", 0.85, "Training vs. validation loss and accuracy, tuned scratch CNN."],
  ["fig2", "curves_resnet", [["resnet18_frozen_curves.png", "Frozen."], ["resnet18_finetune_curves.png", "Fine-tuned."]], "Training vs. validation curves of the ResNet-18 models."],
  ["p", "The loss values in the curves include label smoothing, so they cannot reach zero: with smoothing 0.1 over 200 classes the lowest possible loss is about 0.85, which is why the fine-tuned model's training loss stops near 1.1."],

  ["h1", "Hyperparameter Tuning", "tuning"],
  ["p", "Tuning was done on the scratch CNN (`src/training/tuning.py`, `python -m src.cli tune`), because it is the model most sensitive to its settings and is cheap to train (about 6 minutes per 30-epoch trial). Every trial uses the base configuration with 30 epochs and the same seed, and is scored by its best validation accuracy."],
  ["p", "**Search space.** We tuned the learning rate and the weight decay of AdamW, the two settings that most directly control how fast and how freely the network fits the small training set."],
  ["ul", [
    "**Grid search:** learning rate ∈ {0.0003, 0.001, 0.003} × weight decay ∈ {0.0005, 0.005, 0.05}, 9 trials.",
    "**Random search [9]:** 9 trials, both parameters drawn log-uniformly from the same ranges (0.0003–0.003 and 0.0005–0.05), seed 0.",
  ]],
  ["p", "Both methods get the same budget and the same bounds, so the comparison is fair."],
  ["table", "grid", "Grid search: best validation accuracy of each trial.",
    ["Learning rate \\ weight decay", "0.0005", "0.005", "0.05"],
    [["0.0003", "38.3%", "**39.2%**", "38.9%"],
     ["0.001", "31.9%", "34.4%", "33.9%"],
     ["0.003", "28.4%", "26.6%", "28.3%"]],
    [3412, 2000, 2000, 2000]],
  ["table", "random", "Random search trials, in the order they were run.",
    ["Trial", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    [["LR (×10^{-3})", "2.10", "0.79", "0.97", "1.82", "0.90", "2.43", "0.57", "1.25", "2.44"],
     ["WD (×10^{-3})", "16.4", "1.6", "3.2", "2.0", "7.3", "5.1", "16.2", "1.6", "46.2"],
     ["Val acc. (%)", "29.5", "33.5", "33.4", "31.9", "33.1", "29.8", "**35.6**", "32.6", "31.9"]],
    [1756, 850, 850, 850, 850, 850, 850, 850, 850, 856]],
  ["table", "tunecompare", "Grid search vs. random search (`outputs/tuning/comparison.csv`).",
    ["Method", "Trials", "Best val acc.", "Best found at", "Minutes to best", "Total minutes", "Mean val acc."],
    [["Grid", "9", "**39.2%**", "2nd trial", "**11.8**", "53.0", "33.3%"],
     ["Random", "9", "35.6%", "7th trial", "40.4", "52.2", "32.4%"]],
    [1312, 900, 1400, 1400, 1500, 1400, 1500]],
  ["fig2", "tuning", [["tuning_best_so_far.png", "Best accuracy found vs. time spent."], ["grid_heatmap.png", "Grid search results."]], "Hyperparameter tuning results."],
  ["p", "**Findings.**"],
  ["ul", [
    "**The learning rate dominates.** Each row of the grid is clearly separated (about 39%, 34% and 28%), and in both searches accuracy rises steadily as the learning rate falls.",
    "**Weight decay barely matters.** Within a row, values differ by 1–3 points, which is close to the noise level of the validation set (about ±1.4 points for 1,199 images).",
    "**Grid search won here**, both in best score (39.2% vs. 35.6%) and in speed (best result after 11.8 instead of 40.4 minutes). Random search is usually better when several hyperparameters matter, because it tries more distinct values of each one [9]. Here only one parameter matters, and its best value lies exactly on the lowest grid point; random search, by chance, drew no learning rate below 0.00057, and its best trial was exactly its lowest learning rate.",
  ]],
  ["p", "**Coarse-to-fine follow-up.** Because the best learning rate lay on the edge of the range, we ran a second grid below it with weight decay fixed at 0.005: learning rate 0.0001 gave 30.4%, 0.00015 gave 36.4% and 0.0002 gave 37.9%. Accuracy therefore peaks at 0.0003 and falls on both sides, confirming it as the optimum rather than an artefact of the range. The lowest learning rates were not unstable but too slow: they had not converged within 30 epochs."],
  ["p", "**Final choice.** Learning rate 0.0003 and weight decay 0.005, trained for 60 instead of 30 epochs because the best tuning trials were still improving at epoch 30. Hyperparameter tuning was required for Milestone 1 only; the ResNet models use standard fine-tuning settings."],

  ["h1", "Experiment Tracking", "tracking"],
  ["p", "Every run is logged to MLflow [13] (SQLite backend, `mlflow.db`): all hyperparameters, the device, the total and trainable parameter counts, per-epoch training and validation loss, accuracy, macro precision, recall, F1 and top-5 accuracy, the learning rate, the training-curve plot, and the best checkpoint. Classification runs are in the experiment `classification` and the 21 tuning trials in `tuning_scratch`, each tagged with its search method. Any run can be opened with `mlflow ui --backend-store-uri sqlite:///mlflow.db`. {tab:runs}–{tab:random} and {fig:valcurves} are exported from these runs."],
  ["fig", "mlflow", "mlflow_ui.png", 0.75, "MLflow UI comparing the four classification runs: run details, logged parameters and final metrics."],

  ["h1", "Evaluation", "evaluation"],
  ["h2", "Metrics"],
  ["p", "With TP_{c}, FP_{c} and FN_{c} the true positives, false positives and false negatives of class c, and C = 200 classes:"],
  ["ul", [
    "**Accuracy** — the fraction of images whose top prediction is the correct species. It answers “how often is the model right?”.",
    "**Macro precision** = (1/C) Σ_{c} TP_{c} / (TP_{c} + FP_{c}) — when the model predicts a species, how often it is correct, averaged so every species counts equally.",
    "**Macro recall** = (1/C) Σ_{c} TP_{c} / (TP_{c} + FN_{c}) — of the images of a species, how many the model finds.",
    "**Macro F1** — the average over classes of the harmonic mean of precision and recall. It is low if a model ignores some classes, even when overall accuracy looks good.",
    "**Top-5 accuracy** — the fraction of images whose correct species is among the five highest-scoring predictions. For fine-grained data it shows whether errors are “near misses” between similar species.",
    "**Confusion matrix** — counts of true vs. predicted class; it shows which species are confused with each other.",
  ]],
  ["h2", "Test results"],
  ["table", "test", "Final models on the test set (5,793 images, evaluated once each with `python -m src.cli evaluate`). Full table: `outputs/eval/model_comparison.csv`.",
    ["Model", "Val acc.", "Test acc.", "Precision", "Recall", "F1", "Top-5"],
    [["Scratch CNN (tuned)", "47.4%", "49.6%", "0.504", "0.499", "0.495", "76.6%"],
     ["ResNet-18 frozen", "60.7%", "64.3%", "0.649", "0.647", "0.643", "89.1%"],
     ["ResNet-18 fine-tuned", "72.4%", "**74.4%**", "**0.751**", "**0.746**", "**0.743**", "**92.3%**"]],
    [2512, 1150, 1150, 1150, 1150, 1150, 1150]],
  ["p", "**Interpretation.**"],
  ["ul", [
    "**Ranking.** Fine-tuned > frozen > scratch, by 10.1 and 14.7 points. The fine-tuned model identifies three out of four birds correctly among 200 species; the scratch model one in two, which is still about 100 times better than chance (0.5%).",
    "**Balance across species.** For every model, macro precision, recall and F1 are within about one point of accuracy. No model reaches its accuracy by mastering a few easy species and ignoring others.",
    "**Near misses.** Top-5 accuracy is far above top-1 (92.3% vs. 74.4% for the fine-tuned model): most errors put the true species among the top candidates. {sec:gradcam} shows these are mostly confusions between very similar species.",
    "**Test vs. validation.** All three models score 2–4 points *higher* on test than on validation. The validation set is small (about 6 images per species, ±1.4 points of noise), and all models are measured on the same validation images, so a slightly harder validation sample lowers all three together. The larger test set is the more reliable estimate, and the result also shows that choosing the best epoch on validation did not inflate the test scores.",
    "The test loss uses plain cross-entropy, while training and validation losses include label smoothing, so the loss values are not comparable between them; we compare models by accuracy and F1.",
  ]],
  ["fig2", "confusion", [["scratch_tuned_confusion.png", "Scratch CNN (tuned)."], ["resnet18_finetune_confusion.png", "ResNet-18 fine-tuned."]], "Normalized test confusion matrices (200×200; rows are true classes). The fine-tuned model has a much sharper diagonal. Its errors form small blocks next to the diagonal: the class list is ordered by family, so these are confusions within a family, most visibly gulls (classes 58–65), sparrows (112–132), terns (140–146) and warblers (157–181)."],

  ["h1", "Explainability: Grad-CAM", "gradcam"],
  ["p", "Grad-CAM [10] (`src/visualization/gradcam.py`, `python -m src.cli explain`) shows which image regions pushed the model towards a class. For the class score y^{c} and the feature maps A^{k} of the last convolutional stage, each channel gets the weight α^{c}_{k} = (1/Z) Σ_{i,j} ∂y^{c} / ∂A^{k}_{ij} (the average gradient), and the heatmap is L^{c} = ReLU(Σ_{k} α^{c}_{k} A^{k}), upsampled to the image size. The last stage is used because it has the most class-specific features that still keep a spatial layout. The heatmaps are coarse because that layer is small (4×4 for the scratch CNN, 7×7 for ResNet-18)."],
  ["fig", "gradcam_compare", "gradcam_comparison.jpg", 0.7, "The same validation images explained by the three final models. Green labels are correct predictions, red labels wrong ones."],
  ["p", "**Comparison of the models ({fig:gradcam_compare}).**"],
  ["ul", [
    "The **fine-tuned ResNet-18** attends to the bird in every row, usually its head and breast, with high confidence (76–96%).",
    "The **frozen ResNet-18** focuses on small, specific regions (the Seaside Sparrow's throat, the Vireo's chest). Its error is telling: it calls a Blue Grosbeak an Indigo Bunting, a very similar blue bird, because generic ImageNet features find the right region but cannot separate close species. For the Horned Grebe its heatmap is empty: no region supports the predicted class strongly, a known limitation of Grad-CAM.",
    "The **scratch CNN** gets five of the six images right and often looks at the bird (the Kingfisher's crest, the Sparrow's head), but its heatmaps are broad and spill into the background. For the Horned Grebe it looks at the *pebbles* under the bird and is wrong. The untuned baseline relied on background even more: it predicted waterbirds from the water around them.",
  ]],
  ["fig", "gradcam_scratch", "gradcam_scratch_tuned.jpg", 0.95, "Scratch CNN (tuned): six correct and six wrong validation predictions."],
  ["p", "**Scratch CNN, correct vs. wrong ({fig:gradcam_scratch}).** Correct predictions mostly centre on the bird (the Scott Oriole at 95%, the Laysan Albatross at 85%). Wrong predictions fall into two groups. *Background shortcuts:* the Heermann Gull is called a Brown Pelican while the heatmap sits on the water, the Great Crested Flycatcher becomes a Myrtle Warbler with the heatmap on the branch, and the Crested Auklet is judged from the rock beside it. *Similar species:* Henslow Sparrow vs. Clay-colored Sparrow, and a Shiny Cowbird called a White-necked Raven (both all-black). The wrong predictions also have low confidence (17–53%)."],
  ["fig", "gradcam_resnet", "gradcam_resnet18_finetune.jpg", 0.95, "ResNet-18 fine-tuned: six correct and six wrong validation predictions."],
  ["p", "**Fine-tuned ResNet-18, correct vs. wrong ({fig:gradcam_resnet}).** Correct predictions focus tightly on the head and breast, where the species-defining marks are (the White-crowned Sparrow's head stripes, the Meadowlark's black V). Most errors are between *near-identical species* while the model still looks at the bird: Herring Gull vs. California Gull, Hooded Warbler vs. Wilson's Warbler (with 89% confidence), Red-faced vs. Brandt's Cormorant. The remaining errors come from *hard images*: a small flying Yellow-breasted Chat with almost no heatmap, and a Mangrove Cuckoo mostly hidden by leaves, where the model looks at the branches. One correct prediction (Ovenbird, 11%) is right with very low confidence while looking at the edge of the bird rather than its distinctive head and breast, so not every correct answer is correct for a convincing reason."],

  ["h1", "Benchmarking", "bench"],
  ["p", "**Published results.** Our test numbers use the official CUB test split (minus the one leaked image) and top-1 accuracy, the standard metric for this dataset, so they are comparable in split and metric. Reported results on CUB-200-2011:"],
  ["table", "literature", "Published top-1 accuracy on the CUB-200-2011 test set.",
    ["Method", "Backbone / input", "Top-1"],
    [["ResNet-50 fine-tuned baseline [15]", "ResNet-50, 448 px", "84.5–84.9%"],
     ["Bilinear CNN [11]", "VGG-D + VGG-M, 448 px", "84.1%"],
     ["TransFG [12]", "ViT-B/16 (ImageNet-21k), 448 px", "91.7%"],
     ["**Ours: ResNet-18 fine-tuned**", "ResNet-18, 224 px", "**74.4%**"]],
    [3812, 3600, 2000]],
  ["p", "Our fine-tuned ResNet-18 (74.4%) is about 10 points below the ResNet-50 baseline. The difference has clear causes: a smaller backbone (11 M vs. 24 M parameters), half the input resolution (224 vs. 448 pixels, which matters for small details such as eye rings and wing bars), only 15 epochs, and no fine-grained techniques. Methods above 90% add attention to discriminative parts and transformer backbones pretrained on much larger data."],
  ["p", "**Another team.** We also compared against another team's validation results on the same dataset (their own random 80/20 split, about 1,200 validation images):"],
  ["table", "otherteam", "Validation accuracy compared with another team (different splits; indicative only).",
    ["Setting", "Other team", "Ours"],
    [["Scratch CNN", "9.4% (0.44 M params, 20 epochs)", "**47.4%** (4.82 M params, tuned, 60 epochs)"],
     ["Frozen encoder", "58.3% (ResNet-50)", "**60.7%** (ResNet-18)"],
     ["Partially fine-tuned", "72.3% (ResNet-50)", "—"],
     ["Fully fine-tuned", "**74.3%** (ResNet-50)", "72.4% (ResNet-18)"],
     ["Top-5, fully fine-tuned", "92.5%", "92.1%"]],
    [2612, 3200, 3600]],
  ["p", "The comparison is not fully fair — different validation images, unknown input sizes and no test results on their side — so differences of one or two points are within noise. Two conclusions still hold. First, their scratch CNN underfits (10.9% training accuracy), while ours reaches 96%; model capacity, batch normalization and the learning-rate tuning make the difference. Second, ResNet-50 is only about 2 points better than our ResNet-18 at twice the size, suggesting that at 224 pixels resolution and pretraining matter more than depth."],

  ["h1", "Final Model Selection", "selection"],
  ["p", "**Classification encoder: ResNet-18, fine-tuned.** It has the best score on every metric (74.4% test accuracy, 0.743 macro F1, 92.3% top-5), it is the only model whose Grad-CAM consistently looks at the bird, and it trains in about 4 minutes. Its 512-channel 7×7 feature map is the encoder we plan to reuse in Milestone 2. The tuned scratch CNN (49.6%) remains the from-scratch baseline."],
  ["p", "Colorization and captioning models will be selected in Milestone 2."],

  ["h1", "Failures and Limitations", "limits"],
  ["ul", [
    "**Overfitting.** The scratch CNN reaches 96% training vs. 47% validation accuracy, and the fine-tuned ResNet-18 99% vs. 72%. With about 24 images per species, both models memorise the training set.",
    "**Background shortcuts.** Grad-CAM shows the scratch model still uses water, rocks and branches as cues. Object-centred cropping (the original CUB bounding boxes) could reduce this.",
    "**The default learning rate was too high.** Our first scratch run used 0.001 and reached 31.9%; tuning found 0.0003 (+7 points). The best value was also at the edge of the first search range, which we only discovered and fixed with the coarse-to-fine search.",
    "**Random search underperformed** for this two-parameter problem because its draws missed the low learning-rate region.",
    "**Small validation set.** ±1.4 points of noise makes small differences (e.g. between weight-decay values) inconclusive.",
    "**Low resolution.** 128 pixels for the scratch model and 224 for ResNet-18 lose fine details; published results use 448.",
    "**Single seed.** Every result comes from one run (seed 42); we did not measure run-to-run variance.",
  ]],

  ["h1", "Feedback and Plan for Milestone 2", "plan"],
  ["p", "The teaching assistant's review of this milestone confirmed the main weakness above: our models overfit strongly (training–validation gaps of 49 points for the scratch CNN and 27 points for ResNet-18). Other teams' ResNet models exceeded 90% top-1 accuracy, so our 74.4% leaves clear room for improvement. Since Milestone 1 is not graded, we will address this at the start of Milestone 2, before the encoder is reused for colorization and captioning:"],
  ["ul", [
    "**Regularization against overfitting:** stronger augmentation (random erasing, a wider crop range, mixup/CutMix), higher dropout and weight decay, and early stopping on validation loss.",
    "**Higher resolution:** train at 448 pixels, the standard for fine-grained recognition, so small marks such as eye rings and wing bars stay visible.",
    "**Stronger backbone:** a deeper ResNet (ResNet-50) or a transformer backbone, plugged in through the existing model registry (`src/models/factory.py`).",
    "**Partial fine-tuning** as a middle ground between frozen and full fine-tuning, and several seeds per final model to measure run-to-run variance.",
  ]],
  ["p", "The target is a classification encoder above 90% top-1 accuracy on the test set with a much smaller generalization gap."],

  ["refs", [
    "C. Wah, S. Branson, P. Welinder, P. Perona and S. Belongie. The Caltech-UCSD Birds-200-2011 Dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.",
    "S. Reed, Z. Akata, H. Lee and B. Schiele. Learning Deep Representations of Fine-Grained Visual Descriptions. In *CVPR*, 2016.",
    "K. He, X. Zhang, S. Ren and J. Sun. Deep Residual Learning for Image Recognition. In *CVPR*, 2016.",
    "J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In *CVPR*, 2009.",
    "S. Ioffe and C. Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In *ICML*, 2015.",
    "I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. In *ICLR*, 2019.",
    "I. Loshchilov and F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. In *ICLR*, 2017.",
    "C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens and Z. Wojna. Rethinking the Inception Architecture for Computer Vision. In *CVPR*, 2016.",
    "J. Bergstra and Y. Bengio. Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research*, 13:281–305, 2012.",
    "R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. In *ICCV*, 2017.",
    "T.-Y. Lin, A. RoyChowdhury and S. Maji. Bilinear CNN Models for Fine-grained Visual Recognition. In *ICCV*, 2015.",
    "J. He, J.-N. Chen, S. Liu, A. Kortylewski, C. Yang, Y. Bai and C. Wang. TransFG: A Transformer Architecture for Fine-Grained Recognition. In *AAAI*, 2022.",
    "MLflow: An open source platform for the machine learning lifecycle. https://mlflow.org",
    "A. Paszke et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *NeurIPS*, 2019.",
    "Y. Gao, X. Han, X. Wang, W. Huang and M. Scott. Channel Interaction Networks for Fine-Grained Image Categorization. In *AAAI*, 2020.",
  ]],
];

const ABSTRACT = "We build the first stage of an end-to-end multimodal system on CUB-200-2011 with natural-language captions: a visual encoder trained for 200-way bird species classification, which later becomes the backbone for colorization and captioning. We analyse and clean the data, design an augmentation pipeline, train a CNN from scratch as a baseline, and compare it with ImageNet-pretrained ResNet-18 encoders used frozen and fine-tuned. The scratch CNN is tuned with grid search, random search and a coarse-to-fine follow-up. On the official test split the tuned scratch CNN reaches 49.6% top-1 accuracy, the frozen ResNet-18 64.3%, and the fine-tuned ResNet-18 74.4% (92.3% top-5). Grad-CAM shows that the scratch model still relies partly on background cues, while the fine-tuned model attends to the bird and fails mainly on visually near-identical species.";

const AUTHORS = [
  ["Mohamed Wael Ahmed ElHadary", "23P0043"],
  ["Kerolos Bassem", "23P0136"],
  ["Andrea Aziz Fathy", "23P0379"],
];

// ---------------------------------------------------------------------------
// Numbering of sections, figures and tables (resolved before rendering)
// ---------------------------------------------------------------------------
const refs = {};
{
  let s = 0, f = 0, t = 0;
  for (const b of CONTENT) {
    if (b[0] === "h1") { s++; if (b[2]) refs["sec:" + b[2]] = `Section ${s}`; }
    if (b[0] === "fig" || b[0] === "fig2") refs["fig:" + b[1]] = `Figure ${++f}`;
    if (b[0] === "table") refs["tab:" + b[1]] = `Table ${++t}`;
  }
}
const resolve = (s) => s.replace(/\{(fig|tab|sec):([a-z_]+)\}/g, (m, k, n) => {
  const r = refs[`${k}:${n}`];
  if (!r) throw new Error("undefined reference " + m);
  return r;
});

// Inline markup -> TextRuns
function runs(text, base = {}) {
  const out = [];
  const re = /\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|_\{(.+?)\}|\^\{(.+?)\}/g;
  let last = 0, m;
  text = resolve(text);
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    if (m[1] !== undefined) out.push(...runs(m[1], { ...base, bold: true }));
    else if (m[2] !== undefined) out.push(...runs(m[2], { ...base, italics: true }));
    else if (m[3] !== undefined) out.push(new TextRun({ text: m[3], ...base, font: "Consolas", size: 19 }));
    else if (m[4] !== undefined) out.push(new TextRun({ text: m[4], ...base, subScript: true }));
    else out.push(new TextRun({ text: m[5], ...base, superScript: true }));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const para = (text, opts = {}) => new Paragraph({
  children: runs(text), alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 }, ...opts,
});

function imageRun(file, widthPx) {
  const p = path.join(FIG, file);
  const buf = fs.readFileSync(p);
  const { w, h } = imageSize(buf);
  return new ImageRun({
    type: file.endsWith(".png") ? "png" : "jpg",
    data: buf,
    transformation: { width: Math.round(widthPx), height: Math.round(widthPx * h / w) },
  });
}

// PNG/JPEG dimensions without extra dependencies
function imageSize(buf) {
  if (buf[0] === 0x89) return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
  let i = 2;
  while (i < buf.length) {
    const marker = buf[i + 1], len = buf.readUInt16BE(i + 2);
    if (marker >= 0xc0 && marker <= 0xc3) return { h: buf.readUInt16BE(i + 5), w: buf.readUInt16BE(i + 7) };
    i += 2 + len;
  }
  throw new Error("unknown image format");
}

const caption = (label, text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240 }, keepLines: true,
  children: [new TextRun({ text: label + ". ", bold: true, size: 19 }), ...runs(text, { size: 19 })],
});

const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NO_BORDERS = { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE };

function figure(key, file, frac, text) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 },
      children: [imageRun(file, TEXT_PX * frac)] }),
    caption(refs["fig:" + key], text),
  ];
}

function figure2(key, items, text) {
  const half = Math.floor(TEXT_W / 2);
  const cell = ([file, sub], i) => new TableCell({
    width: { size: half, type: WidthType.DXA }, borders: NO_BORDERS, verticalAlign: VerticalAlign.BOTTOM,
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [imageRun(file, TEXT_PX / 2 - 8)] }),
      new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 40 },
        children: runs(`(${"ab"[i]}) ${sub}`, { size: 18 }) }),
    ],
  });
  return [
    new Table({ width: { size: half * 2, type: WidthType.DXA }, columnWidths: [half, half],
      borders: NO_BORDERS, rows: [new TableRow({ cantSplit: true, children: items.map(cell) })] }),
    caption(refs["fig:" + key], text),
  ];
}

// Booktabs-style table: thick rules top and bottom, thin rule under the header
function table(key, text, header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const thick = { style: BorderStyle.SINGLE, size: 12, color: "000000" };
  const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  const mk = (cells, r) => new TableRow({
    cantSplit: true, tableHeader: r === 0,
    children: cells.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      shading: r === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: "E8EEF4" } : undefined,
      borders: {
        top: r === 0 ? thick : NONE,
        bottom: r === 0 ? thin : r === rows.length ? thick : NONE,
        left: NONE, right: NONE,
      },
      children: [new Paragraph({
        alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, keepNext: true,
        children: runs(c, { size: 19, bold: r === 0 || undefined }),
      })],
    })),
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 80 },
      children: [new TextRun({ text: refs["tab:" + key] + ". ", bold: true, size: 19 }), ...runs(text, { size: 19 })] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
      rows: [header, ...rows].map(mk) }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ---------------------------------------------------------------------------
// Assemble
// ---------------------------------------------------------------------------
let olCount = 0;
const body = [];
let sec = 0, sub = 0;
for (const b of CONTENT) {
  switch (b[0]) {
    case "h1":
      sec++; sub = 0;
      body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(`${sec}  ${b[1]}`)] }));
      break;
    case "h2":
      sub++;
      body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(`${sec}.${sub}  ${b[1]}`)] }));
      break;
    case "p": body.push(para(b[1])); break;
    case "ul":
      for (const item of b[1]) body.push(para(item, { numbering: { reference: "bullets", level: 0 }, spacing: { after: 60, line: 276 } }));
      break;
    case "ol": {
      olCount++;
      for (const item of b[1]) body.push(para(item, { numbering: { reference: "numbers", level: 0, instance: olCount }, spacing: { after: 60, line: 276 } }));
      break;
    }
    case "fig": body.push(...figure(b[1], b[2], b[3], b[4])); break;
    case "fig2": body.push(...figure2(b[1], b[2], b[3])); break;
    case "table": body.push(...table(b[1], b[2], b[3], b[4], b[5])); break;
    case "refs":
      body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("References")] }));
      b[1].forEach((r, i) => body.push(new Paragraph({
        spacing: { after: 80 }, indent: { left: 500, hanging: 500 },
        children: [new TextRun(`[${i + 1}]\t`), ...runs(r)],
        tabStops: [{ type: "left", position: 500 }],
      })));
      break;
    default: throw new Error("unknown block " + b[0]);
  }
}

const authorRows = AUTHORS.map(([name, id]) => new TableRow({ children: [
  new TableCell({ width: { size: 4000, type: WidthType.DXA }, borders: NO_BORDERS,
    children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: name, size: 26 })] })] }),
  new TableCell({ width: { size: 2000, type: WidthType.DXA }, borders: NO_BORDERS,
    margins: { left: 300 },
    children: [new Paragraph({ children: [new TextRun({ text: id, size: 26, color: "444444" })] })] }),
] }));

const titlePage = [
  new Paragraph({ spacing: { before: 2400 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "From Pixels to Language", bold: true, size: 56, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 },
    children: [new TextRun({ text: "CSE 485 / CSE 488 Deep Learning — Final Project Report", size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1F3864", space: 12 } },
    children: [new TextRun({ text: "Milestone 1: Data Understanding and Image Classification", size: 28, italics: true })] }),
  new Paragraph({ spacing: { before: 600 }, children: [] }),
  new Table({ width: { size: 6000, type: WidthType.DXA }, columnWidths: [4000, 2000], alignment: AlignmentType.CENTER,
    borders: NO_BORDERS, rows: authorRows }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600 },
    children: [new TextRun({ text: "Ain Shams University, Faculty of Engineering", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "November 2026", size: 24, color: "444444" })] }),
  new Paragraph({ spacing: { before: 1200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Abstract", bold: true, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 600, right: 600 }, spacing: { line: 276 },
    children: runs(ABSTRACT, { size: 21 }) }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 32, color: "1F3864" })], spacing: { after: 200 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
];

const doc = new Document({
  creator: "Mohamed Wael Ahmed ElHadary, Kerolos Bassem, Andrea Aziz Fathy",
  title: "From Pixels to Language — Milestone 1 Report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "1F3864", font: FONT },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E5496", font: FONT },
        paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
      titlePage: true },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "666666" })] })] }),
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
    },
    children: [...titlePage, ...body],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT} (${(buf.length / 1e6).toFixed(1)} MB)`);
});
