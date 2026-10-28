// Study notes, one chapter per week. Light markdown: "## " section, "### " sub-heading, "- " bullet,
// "1. " numbered step, "> " exam tip, **bold**, `code`, blank line = new paragraph.
const NOTES = [
{ w: 1, md: `
## Big picture
- **AI** = intelligence demonstrated by machines. **ML** = the subset of AI where the machine learns its parameters from data. **Deep learning** = ML with multi-layer neural networks. AI ⊃ ML ⊃ DL.
- Many ML ideas (least squares, Bayes) date from the 1800s; they just weren't called ML.
- **GOFAI / symbolism**: a human expert writes rules (decision logic) or an analytical physics model with few fitted parameters. **Connectionism / ML**: the machine learns thresholds (classification) or parameters of a generic model (regression) from data.
- Connectionism was limited by slow computers and lack of data, then enabled by PCs, the Internet (data) and GPUs.

## Tasks and paradigms
- **Classification**: predict a discrete class (spam / not spam, pneumonia / healthy, 1 of 1000 objects).
- **Regression**: predict a continuous value (house price, krill density, heart rate).
- **Supervised**: labelled pairs (x, y). **Unsupervised**: only x; find clusters or structure. Labelling is expensive and is a big business.

## ML as "creating maps"
- An image is just numbers: 4000 × 3000 × 3 = 36 million values.
- Layers of additions and multiplications map those numbers to a **location on a map** (a vector): the **embedding / latent variable**. Similar inputs land close together.
- Classification: read which region the location falls in. **Generative AI**: pick a location and decode it into new data. A trajectory on the map gives a movie. ChatGPT guesses the next location (token).
- Works for any data that can be converted to numbers, and locations can be aligned across data types (text ↔ image).

## History: two paradigm shifts
- 1872: impressionism (Monet, *Impression, Sunrise*) moved painting from accurate models to data-driven perception.
- 2012: CNNs won ImageNet (14M images, 20,000+ categories), moving AI from models to data. Triggered by GPU compute plus big data.

## What could go wrong (repeated in weeks 5 and 12)
1. **Unknown data**: a case unlike the training data, so the model fails to generalise.
2. **Data drift**: after deployment the data changes (new camera, new circumstances).
3. **Bias**: the mapping is wrong for a subgroup (e.g. black patients mapped near healthy white volunteers). Sources: class imbalance, sampling, human labelling.

## Computers vs humans (Kahneman)
- **Traditional computer**: exact, vast memory; fast; reliable; only solves problems coded by humans.
- **System 1**: automated, unconscious; solves known problems; requires learning; relatively fast, **multitask**.
- **System 2**: conscious thinking; solves **unknown** problems by reasoning; relatively slow, **one task**.
- Current ML ≈ System 1 skills. Reasoning on never-seen problems is still open.

## Intelligence, AGI, ASI
- **AGI** = human-level general intelligence. **ASI** = beyond human. Questions: can scaling alone reach AGI? Would ASI be aligned with us?
- Efficiency gap: the brain uses ~20 W and learns from few examples.
- Ethical issues: jobs, responsibility, control, bias, concentration of power.

> Exam traps: System 1 is the **multitask** one, and data bias can come from **all three** sources.
`},
{ w: 2, md: `
## Supervised ML recipe
1. Get data and inspect it. 2. Choose the simplest possible model. 3. Is there an analytical solution? 4. Can the parameter space be inspected (grid search)? 5. Otherwise optimise (gradient descent). 6. Analyse the results on test data.
- Assumption: a relationship y = f(x) + noise exists. "All models are wrong, but some are useful" (George Box).

## Correlation
- **Pearson** r = cov(x, y) / (σₓσᵧ) ∈ [−1, 1] measures linear correlation. **Spearman** = Pearson on ranks, so it captures monotonic non-linear relationships.
- Check correlation before regression: no correlation means a line explains nothing.

## Linear regression and OLS
- Model: y = mx + c. Parameters: m, c. Residual: eᵢ = yᵢ − ŷᵢ.
### How to derive OLS
1. Stack the samples: A = [[x₁, 1], …, [x_N, 1]], W = [m, c]ᵀ, B = [y₁ … y_N]ᵀ, so AW ≈ B.
2. Cost E(W) = ‖AW − B‖² = (AW − B)ᵀ(AW − B).
3. Derivative: ∂E/∂W = 2Aᵀ(AW − B) = 0.
4. Normal equations: AᵀAW = AᵀB, so **W = (AᵀA)⁻¹AᵀB** (the pseudo-inverse).
- **Multiple variables**: add a column per feature. **Polynomial**: columns 1, x, x², … The model is still linear in W, so the same formula applies.

## Assumptions and limitations
- Linearity; **independence** of residuals; **homoscedasticity** (constant residual variance); **normality** of residuals (mean 0).
- Limitations: can't capture complex relationships; sensitive to outliers; Simpson's paradox (a trend reverses when groups are pooled).
- Polynomial: curvilinear relationship; the **order is the hyperparameter**; a high order overfits; a 1st-order polynomial = linear regression.

## Three ways to find parameters
- **OLS** (closed form). **Grid search**: evaluate the loss on a grid of (m, c) and keep the best. Cost is kᵖ evaluations.
- **Gradient descent**: w ← w − η∇E. On un-normalised data (one variable ~0.05, another ~30) it zig-zags, so **normalise** first (z = (x − μ)/σ).

## Experimental design
- **Train** fits parameters, **validation** picks hyperparameters, **test** is secret, used once, ideally from a different source.
- **k-fold CV**: split into K folds and rotate the validation fold. Use it when data is small or expensive.
- **Overfitting**: low training error, high validation error.

## kNN
1. Normalise. 2. Store the training data. 3. Compute distances. 4. Take the K closest. 5. Majority vote (classification) or mean (regression).
- Pick K on validation data. Pros: interpretable, no training. Cons: slow and memory-heavy at prediction, sensitive to scale and high dimension.

## Decision trees and random forests
- A sequence of axis-aligned rules (income < $35k?). Assumes variables can be split to separate classes.
- Pros: **interpretable**, handle mixed features. Cons: overfit, blocky boundaries, unstable.
- **Pruning**: pre-pruning (max depth, min samples) or post-pruning (remove branches that don't help validation).
- **Random forest**: many trees on bootstrap samples and random features, combined by vote. Lower variance.

## Confusion matrix
- Rows = ground truth, columns = prediction. The diagonal is correct; off-diagonal cells are specific confusions. Multi-class: focus on one class to get TP / FP / FN / TN.
`},
{ w: 3, md: `
## From linear classifier to neuron
- Score **z = w·x + b**. The boundary is z = 0; **w is perpendicular** to it (for ax + by + c = 0 the normal is (a, b)). b shifts the line.
- The score is a scaled signed distance: sign gives the class, magnitude gives "confidence".
- **Neuron** = linear classifier + activation f(z). One neuron gives one linear boundary. A neuron is a projection (dot product) onto w.

## Layers as geometric transformations
- Each layer maps one feature space to another (linear map plus non-linearity) so the classes become linearly separable for the next layer. The **last layer is a linear classifier** on learned features.
- **Without activations**, W₂(W₁x + b₁) + b₂ = Wx + b, so any depth collapses to one linear layer.
- 2-layer ANN (one hidden layer) = universal approximator.

## Loss functions
- Regression: **MSE**. Classification: **cross-entropy** (softmax loss) or **hinge** (multiclass SVM).
- **Softmax**: pᵢ = exp(sᵢ)/Σexp(sⱼ). **CE** = −log p_correct (maximise the softmax score of the correct class).
- **Hinge**: Σ over j ≠ y of max(0, sⱼ − s_y + Δ), which is 0 once the margin is met. CE never reaches exactly 0.
### How to compute a softmax / CE example
1. Scores s = [2, 1, 0]. exp gives [7.39, 2.72, 1.00], with sum 11.11.
2. p = [0.665, 0.245, 0.090]. If the correct class is 0: CE = −ln 0.665 = 0.41.

## Gradient descent and variants
- Update: **w ← w − η ∇L(w)**. Initialise weights randomly (not all equal, because of symmetry).
- **Batch GD** (all data), **SGD** (1 sample), **mini-batch** (B samples, the usual choice).
- **Momentum**: v ← βv + ∇L, w ← w − ηv (heavy ball, damps oscillation).
- **AdaGrad** (accumulates g², so steps shrink), **RMSProp** (EMA of g²), **Adam** = momentum (β₁ ≈ 0.9) + RMSProp (β₂ ≈ 0.999), the most used optimiser.
- **EMA**: yₜ = αyₜ₋₁ + (1 − α)xₜ stores only one value, whereas a moving average stores 2N + 1.
- **Learning rate**: too small is slow; too large oscillates or diverges. Start large, then decay. **Epoch** = a full pass; **batch size** = samples per update.

## Deep learning
- DL = ML with **more than one hidden layer**. **Depth** = number of layers; **width** = units per layer.
- **Backpropagation** = the chain rule, from output back to input. Forward pass computes the loss; backward pass computes the gradients.
- Activations: **sigmoid, tanh, ReLU**. Sigmoid saturates, σ' ≤ 0.25, so gradients **vanish** through many layers.

## Overfitting in deep nets
- More parameters means more capacity to overfit. Identify it with training vs validation curves.
- Mitigate with **regularisation** (weight decay), **dropout** (remove random connections during training), **early stopping**, data augmentation.

## How to count MLP parameters
- Layer (n_in → n_out): **n_in·n_out + n_out**.
- Example 2 → 10 → 1: (2·10 + 10) + (10·1 + 1) = 30 + 11 = **41**.
`},
{ w: 4, md: `
## Computer vision tasks
- **Classification**: one label per image (assumes one object). **Semantic segmentation**: a class per pixel (no objects, only pixels). **Object detection**: class + box per object (0 or more objects). **Instance segmentation**: class + mask per object.
- Dataset assumptions: the dataset distribution matches the application, and the ground truth is unambiguous. These are often violated, so an 87% benchmark score may not transfer.

## Why convolution
- An MLP ignores spatial order (shuffled pixels give the same result) and needs huge weight matrices.
- Convolution = a sliding **match filter**: high response where the image matches the template (a "nose"). **Local connectivity + weight sharing** give few parameters and detect anywhere.
- **Translation** invariant (equivariant); **not** rotation or scale invariant, so use augmentation.

## How to compute conv output size and parameters
- **Output = (N − F + 2P)/S + 1**. N = input, F = filter, P = padding, S = stride.
- Examples: 7×7, F = 3, S = 1 gives 5×5; S = 2 gives 3×3; S = 3 gives 2.33, which doesn't fit.
- "Same" size: F = 3, P = 1, S = 1 (or F = 5, P = 2).
- **Params = (F·F·C_in + 1)·C_out**. The filter depth always equals C_in.
- Example: Conv2d(3, 5, k = 3) gives (3·3·3 + 1)·5 = **140**. 6 filters 5×5 on RGB gives (75 + 1)·6 = **456**.
- Tensor layout in PyTorch: **[B, C, H, W]**, e.g. [100, 3, 224, 224].
### Worked chain (practice exam Q10)
1. [3, 64, 64] → Conv 16@3×3 P1 → [16, 64, 64] → ReLU (same) → MaxPool 2×2/2 → [16, 32, 32].
2. Conv 32@3×3 P1 → [32, 32, 32] → ReLU → MaxPool → **[32, 16, 16]**.

## Pooling and downsampling
- Max or average pooling reduces H and W. **No learnable parameters**; 3 hyperparameters (type, size, stride). A strided conv is a learnable alternative. A transpose conv upsamples.

## What CNNs learn
- Early layers learn edges and contrast; deeper layers learn textures, parts, objects. The receptive field grows with depth.
- Generalisation is hard because of viewpoint, illumination, deformation, occlusion, background clutter and intra-class variation.

## Regularisation and normalisation
- **Dropout**: randomly zero activations (train only). **Input normalisation** per channel: (x − μ)/σ.
- **BatchNorm**: normalise activations with batch statistics plus learnable γ, β. Speeds up training and is robust to bad initialisation.
- **Data augmentation**: random flips, crops, rotations, colour jitter (torchvision transforms).

## Segmentation: UNet
- Without pooling, a 3×3 conv adds 2 pixels of receptive field, so ~112 layers are needed to cover 224 pixels. That's too deep, and bigger kernels explode the parameter count.
- **UNet**: an encoder (downsampling, context) and a decoder (upsampling) with **skip connections** that keep detail. Multiscale. Ubiquitous in medical imaging and diffusion.

## Very deep networks: ResNet
- The chain rule over many layers makes gradients vanish. A **residual block** computes y = F(x) + x, so the identity path carries the gradient. Allows 50-152 layers.
- **Transfer learning**: pre-trained ResNets (ImageNet) in torchvision; replace the head and fine-tune.
`},
{ w: 5, md: `
## Train / validation / test, again
- Hyperparameters are chosen on validation (or k-fold CV); the test set is used once.
- **model.train() vs model.eval()**: Dropout(0.1) zeroes 10% in train mode and nothing in eval. BatchNorm uses batch statistics in train and running statistics in eval. Testing in train mode, especially with batch size 1, gives wrong results.

## Two types of mistakes
- **False positive** = false alarm = **Type I** ("overdo a task"). **False negative** = miss = **Type II** ("not very good at a task").
- Costs are asymmetric (pregnancy test, fall detection for the elderly, cancer). **Accuracy mixes both**, so it's rarely a good target.

## How to compute the metrics
- From TP, FP, FN, TN:
- **Accuracy** = (TP + TN)/(all). **Precision** = TP/(TP + FP). **Recall = Sensitivity = TPR** = TP/(TP + FN). **Specificity** = TN/(TN + FP). **FPR** = 1 − specificity. **F1** = 2PR/(P + R).
### Worked example
1. TP = 40, FN = 10, FP = 20, TN = 930.
2. Precision = 40/60 = 0.667; Recall = 40/50 = 0.80; Specificity = 930/950 = 0.979; Accuracy = 970/1000 = 0.97.
3. The 97% accuracy hides that one in three alarms is false.
- Imbalance trap: 1% prevalence and "always healthy" gives 99% accuracy with 0% sensitivity.
- **Multi-class**: focus on one class (e.g. "Cold") and group the rest as negative.

## ROC curve
1. The model outputs scores. 2. Sweep a threshold t. 3. For each t, plot TPR (sensitivity) against FPR (1 − specificity).
- The diagonal is random and the top-left is perfect; report AUC or the point closest to the corner.
- **Screening**: minimise FN (high sensitivity). **Diagnosis**: minimise FP (high specificity), e.g. a PET scan (~$2,000) after cheap screening.
- **Low prevalence doesn't change the ROC**, but precision drops. Example: sensitivity = specificity = 98% at 1% prevalence gives PPV ≈ 33%.
- Small studies give noisy estimates; increase the sample size.

## Calibration
1. Store the confidence and correctness of each test prediction. 2. Bin by confidence. 3. Compute the accuracy per bin. 4. Plot against confidence.
- Calibrated means on the diagonal (the 50-60% bin is about 55% accurate). Deep networks are usually **over-confident**. Fixes: temperature scaling, ensembles.

## Robustness
- **Feature shift** (appearance changes) vs **class shift** (novel classes). Networks misclassify novel classes with **high confidence**: softmax must pick a known class, and far-from-boundary points get extreme scores.
- **Adversarial attack** (test time): x_adv = x + ε·sign(∇ₓL) (FGSM). Uses the gradient with respect to the input, giving the shortest vector across the boundary (panda → gibbon).
- **Backdoor attack** (training time): poisoned samples with a trigger mapped to a target label. Invisible on clean tests. A risk with public datasets and pre-trained models.
- **Ensembles**: average diverse models (random initialisation) for better accuracy, calibration and robustness.

## Explainability and ethics
- Decision trees are explainable; CNNs are black boxes. Tools: **gradient saliency** (∂score/∂image), **occlusion maps**, **Grad-CAM**, feature visualisation.
- Case study: a CNN predicts race from chest X-rays, which is a hidden bias risk.
- Responsible AI along the value chain; Australia's AI Ethics Principles; self-regulation vs government regulation; AI standards.
`},
{ w: 6, md: `
## Why go beyond supervised
- Supervised learning depends on the quantity, diversity and quality of labels; learns only discriminative features; and labels are expensive or ambiguous.
- **Unsupervised**: no labels. **Self-supervised**: create targets from the data. **Semi-supervised**: few labels plus many unlabelled samples.

## k-means
1. Pick K random centroids. 2. Assign each point to the nearest centroid. 3. Move each centroid to the mean of its points. 4. Repeat.
- K is a hyperparameter (not kNN's k). Pros: simple, efficient. Cons: choosing K, spherical clusters, local minima.

## Maximum likelihood: how to derive it for a Gaussian
1. Likelihood L(μ, σ) = Πᵢ (1/√(2πσ²)) exp(−(xᵢ − μ)²/(2σ²)).
2. NLL = (n/2) log(2πσ²) + Σᵢ (xᵢ − μ)²/(2σ²).
3. ∂/∂μ = 0 gives **μ̂ = (1/n)Σxᵢ**.
4. ∂/∂σ = 0 gives **σ̂² = (1/n)Σ(xᵢ − μ̂)²**. This is biased; for small n, use 1/(n − 1).
- With σ fixed at 1, the MLE of μ is least squares. NLL connects to MSE.

## GMM and EM
- p(x) = Σₖ πₖ N(x | μₖ, σₖ). History: Pearson (1894) on Weldon's crabs.
### EM steps
1. **E-step**: responsibility γᵢₖ = πₖN(xᵢ|μₖ, σₖ) / Σⱼ πⱼN(xᵢ|μⱼ, σⱼ).
2. **M-step**: Nₖ = Σᵢγᵢₖ; μₖ = Σᵢγᵢₖxᵢ/Nₖ; σₖ² = Σᵢγᵢₖ(xᵢ − μₖ)²/Nₖ; πₖ = Nₖ/N.
3. Repeat until the log-likelihood converges (it never decreases).
- k-means = EM with **hard** assignments. Application: brain MRI tissue segmentation.
- Clustering applications: anomaly detection, recommender systems, exploring unlabelled data.

## Visualisation: t-SNE and UMAP
- Move the data to form clusters (non-parametric). t-SNE: Gaussian similarities in high dimension, Student-t in 2D, minimise **KL(P‖Q)**. UMAP is similar but faster and keeps more global structure.
- **KL divergence** D_KL(p‖q) = ∫p log(p/q) ≥ 0, = 0 iff p = q, **not symmetric**. The lecture builds it from naive differences, then ratios, then log-ratios.

## Transfer learning
- Use a model trained on Dataset A (e.g. 1000 classes) and replace the head with 10 outputs for Dataset B.
- **Freezing**: train only the new head. **Fine-tuning**: update all layers with a small learning rate. It's a balance between new and old knowledge.

## Semi-supervised
- **Pseudo-labelling**: predict unlabelled data, keep confident predictions as labels, retrain. Risk: confirmation bias.
- **Consistency regularisation**: the same prediction under different augmentations. Key choices: which augmentations and the loss weight.

## Contrastive learning
1. Positives = augmentations of the same image; negatives = other images.
2. Encoder to a low-dimensional embedding.
3. Contrastive or margin loss pulls positives together and pushes negatives apart.
4. Use as pretraining for downstream tasks.
- Limitations: choosing augmentations and their magnitude; needs many negatives; long training.

## Foundation models
- Trained on huge unlabelled data (self-supervised), adaptable to many tasks. **DINOv2** for vision; LLMs.
- Advantages: general features, few labels needed. Dangers: inherited bias, opacity, cost and concentration of power.
`},
{ w: 7, md: `
## Sequence data
- **Time series**: ordered points over time; order matters. Text can be treated as a time series of tokens. Applications: forecasting (stocks, energy), prediction or classification (activity, speech, translation).

## Tokenisation
- Shakespeare: 65 symbols (26 × 2 letters + specials) and 1,115,000 characters.
- Characters give a small vocabulary and long sequences; words give a large vocabulary and short sequences; subwords (BPE, SentencePiece) are the compromise. LLM vocabularies are ~50,000 tokens.

## Classic language models
- Letter frequency gives a naive model. **Bigram**: P(next | current), counted or learned. Generates "not random but still gibberish". N-grams explode as V^N.
- **Bag of words**: count vector. Easy, but loses order and is high-dimensional.
- **Embeddings** (lookup table): each token is a learned dense vector with semantic meaning. Word2Vec. king − man + woman ≈ queen.

## RNN and LSTM
- RNN: hₜ = f(Wₓxₜ + Wₕhₜ₋₁ + b). Many-to-many (translation), many-to-one (sentiment), one-to-many (generation).
- RNN limitations: **can't parallelise**, **vanishing or exploding gradients**, **short memory**.
- LSTM: a **cell state** with **forget, input and output gates**, giving longer memory and robustness to vanishing gradients. Still sequential and slow.

## Attention
### How to compute it
1. Q = XW_q, K = XW_k, V = XW_v. For T tokens: Q, K, V are T × D.
2. Scores S = QKᵀ/√D give a **T × T** matrix.
3. Causal mask (decoder): set j > i to −∞.
4. A = softmax(S) row-wise, so **each row sums to 1**.
5. Output = AV (T × D), added to the embeddings (residual): tokens "nudged" by context.
- **Multi-head**: split D into h heads of size D/h (768/12 = 64), compute in parallel, concatenate, project.
- **Positional encoding**: attention ignores order ("not happy because raining" vs "happy because not raining"). Add PE(pos, 2i) = sin(pos/10000^(2i/D)), PE(pos, 2i + 1) = cos(…).

## Transformer block and LLMs
- Block: multi-head attention + add & norm + feed-forward + add & norm. Stack N blocks (GPT-3: **96**, DeepSeek-R1: **61**).
- Output: a linear layer over the vocabulary, then softmax, giving next-token probabilities. Generate autoregressively.
- Training: **self-supervised next-token prediction** (targets = inputs shifted by 1). BERT: masked tokens.
- Strengths vs RNN: parallel training; global context via self-attention. Limitations: **data hungry**, **O(T²)** attention cost, compute.
- Long context: **KV cache** (reuse past K and V), linear or windowed attention. Context windows of 128k+ tokens.

## Baby LM tutorial
- Addition model: '23+12=' completes to '35'. Character tokenizer, padding + EOS, transformer blocks, autoregressive generation, accuracy = % of correct digits.
`},
{ w: 8, md: `
## Discriminative vs generative
- Discriminative learns **p(y|x)**; generative learns **p(x)** or **p(x, y) = p(y|x)p(x)**. High p(y|x) but low p(x) means an uncertain decision.

## Manifold hypothesis
- A 3×32×32 image with 256 levels has 256³⁰⁷² possibilities, mostly noise. Real data lies on a low-dimensional manifold.
- Linear manifold means PCA; non-linear means an autoencoder; a probabilistic latent means a VAE.

## PCA: how to derive it
1. Centre the data: X ← X − mean. Covariance C = (1/N)XᵀX.
2. Maximise the variance uᵀCu subject to ‖u‖ = 1. The Lagrangian gives **Cu = λu**, so the directions are **eigenvectors** and λ is the variance along each.
3. Sort by λ (descending). Components are orthogonal; Σλ = total variance.
4. Project: z = Uᵀx; reconstruct: x̂ = UUᵀx (PCA as a linear autoencoder and denoiser).
- SVD of X is one way to compute it (optional). With N samples, at most ~N components (50,000 images in 10⁶ dimensions gives ≤ 50,000).

## Autoencoders
- Enc: x → z (bottleneck), Dec: z → x̂, loss ‖x − x̂‖². Self-supervised: the input is the target.
- **Denoising**: input x + noise, target x. **Masked**: hide patches and reconstruct. **Sparse**: a large latent (e.g. 784) with an L1 or top-k sparsity constraint, used for LLM interpretability.
- Encoder-decoders are everywhere: transformers, UNet, masked AEs, DALL-E, Stable Diffusion.
- As a generator an AE fails: the latent distribution is uncontrolled, so a random z gives rubbish.

## VAE: how to derive the loss
1. We want max log p(x), but p(x) = ∫p(x|z)p(z)dz is intractable.
2. Introduce the encoder q_φ(z|x): log p(x) = ELBO + KL(q_φ(z|x) ‖ p(z|x)) ≥ ELBO.
3. **ELBO = E_q[log p_θ(x|z)] − KL(q_φ(z|x) ‖ N(0, I))**.
4. With a Gaussian decoder: **Loss = MSE(x, x̂) + β·KL**.
- **KL closed form** (per latent dimension): KL = ½(μ² + σ² − ln σ² − 1). When μ = 0, σ = 1, KL = 0.
- **Reparameterisation**: z = μ + σ·ε, ε ~ N(0, 1). The encoder outputs μ and **ln σ²** (2 × latent dimension); σ = exp(0.5·logvar).
- Generation: z ~ N(0, I), x = Dec(z).

## Balancing β and failure modes
- β too high: perfect prior fit but poor reconstruction, up to posterior collapse (μ = 0, σ = 1). β too low: like an AE, poor prior fit.
- Mode collapse extremes: σ → 0 ("autoencoder mode") or z ignoring x.
- 2 classes give 2 manifolds and a ridge in z, solved by the **conditional VAE**: give c to the encoder and decoder. To generate a "7", set c = 7. Classify by trying each c and taking the lowest reconstruction error.
`},
{ w: 9, md: `
## High-dimensional geometry
- For z ~ N(0, I_D): **‖z‖ ≈ √D** (D = 10 gives 3.16; D = 400 gives 20). Samples live in a thin shell; a norm < 1 is almost impossible when D > 100.
- Two random samples are **nearly orthogonal** (cos ≈ 0 ± 1/√D). They're **not** aligned with the axes.
- **Interpolation**: the midpoint of two samples has norm ≈ 0.71√D, inside the shell (atypical). Use slerp.
- **Curse of dimensionality**: 3 bins per dimension gives 3^D cells; data becomes sparse and distances concentrate. Real data has structure.
### How to compute the typical norm
1. E‖z‖² = Σ E[zᵢ²] = D·1 = D. 2. So ‖z‖ ≈ √D. 3. Example: D = 3072 (CIFAR 3×32×32, normalised) gives ≈ 55.

## Normalisation
- Data: centre and scale to std 1. **BatchNorm**: per channel across the batch. **LayerNorm**: per sample across features (used in transformers). Both have learnable γ, β.

## GAN
- Generator G(z) makes x̃; discriminator D(x) gives P(real). Real = 1, fake = 0, **BCE loss**, alternating updates. Only G is needed after training.
- **cGAN**: the condition modulates features. **VAE-GAN**: a VAE plus a discriminator for sharper outputs. Problems: instability, mode collapse.

## Score-based and diffusion
- **Score** s(x) = ∇ₓ log p(x). **Langevin**: x ← x + (η/2)s(x) + √η·ε, a gradient walk with noise.
- Sparse regions have no score estimate, so **add noise at multiple scales**, starting with lots of noise.
### DDPM: how training and sampling work
1. Forward (no training): **xₜ = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε**, with ᾱₜ = Π(1 − βₛ). The **noise schedule** βₜ is linear or cosine.
2. Training: random t, sample ε, form xₜ, loss = **‖ε − ε_θ(xₜ, t)‖²**. Predicting the noise is equivalent to predicting the score.
3. Sampling: x_T ~ N(0, I), iteratively remove the predicted noise (plus small noise) down to t = 0.
- **Classifier guidance**: ∇log p(x|c) = ∇log p(x) + ∇log p(c|x) (needs a classifier). **Classifier-free**: ε̂ = ε(x, ∅) + w[ε(x, c) − ε(x, ∅)].

## Conditional flow matching
1. Pair noise x₀ with data x₁; path xₜ = (1 − t)x₀ + t·x₁; true velocity = x₁ − x₀.
2. Loss = **MSE(u_θ(xₜ, t), x₁ − x₀)**.
3. Generate: solve the **ODE** dx/dt = u_θ(x, t) with a solver. No random walk, but several steps (the field is non-linear). Rectified flow straightens the paths.

## Summary of generative models
- AE / VAE / GAN: a low-dimensional latent, one-shot decoding, but many random z are bad.
- Diffusion / flow: the latent has the same dimension as x, multi-step generation, and any random z is good.

## Measuring quality: FID
1. Real and generated images go through Inception-v3 (299×299 input) to pool3 features.
2. Fit Gaussians (μ, Σ) to each set.
3. **FID = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_rΣ_g)^½)**. Lower is better. Others: IS, KID, SWD.
`},
{ w: 10, md: `
## Overfitting as a feature
- Normally overfitting is bad. In **implicit neural representations** we deliberately overfit one signal; the weights become the representation.

## Universal approximation, constructively
- One hidden layer with enough units approximates any function, but more units are harder to optimise. Wider vs deeper: two or three layers of 100 units fit better.
- sigmoid(wx + b): **w = steepness**, **b = shift**. Two steep sigmoids (w = 50) make a box. Many boxes approximate any 1D function. ReLU units work similarly.

## 2D and 3D signals
- Binary image: MLP (x, y) → 0/1 gives smooth (anti-aliased) edges; the network doesn't know it's an image.
- Representations: equation (~200 bytes), pixels (10,000 bytes / jpg 1,900), MLP weights (~105 floats ≈ 450 bytes).
- **SDF**: closest distance to the shape, > 0 outside, < 0 inside. Threshold at 0 gives a mask; the gradient (via **autograd**) gives the direction. **Infinite resolution**: query any grid with the same parameters.
- Image intensity I(x, y): deep ReLU MLPs miss detail, so add **batch norm** and Fourier features.
- **Geometric transforms**: transform the coordinates, X' = (T + P + S)·R·X, then query the MLP.

## Graphics refresher
- **Rendering**: from a 3D scene to a 2D image. Extrinsic **[R|T]** takes world to camera; intrinsic **K** takes camera to screen.
- **Ray**: X = P + t·d, t > 0. The **rendering equation** is emitted light plus ∫BRDF·incoming·(n·s); **Lambert's law** gives the n·s factor.
- Materials: diffuse (all directions), specular (preferred direction), mirror (one direction).
- **Alpha compositing**: C = Σ cₖαₖ Π_(j<k)(1 − αⱼ). α = 1 is opaque.

## NeRF
- MLP F(x, y, z, θ, φ) → (RGB, σ). Learn 3D from 2D photos by volume rendering; loss = MSE on pixels.
### Volume rendering: how to compute it
1. Ray r(t) = o + t·d. The probability of hitting in [t, t + dt] is σ(t)dt.
2. Transmittance **T(t) = exp(−∫σ ds)** is the probability of reaching t unblocked.
3. Discrete: αᵢ = 1 − exp(−σᵢδᵢ), Tᵢ = Π_(j<i)(1 − αⱼ), **C = Σ Tᵢαᵢcᵢ**.
4. Coarse pass, then a fine pass (more samples where the weights are high).
- **Spectral bias**: MLPs learn low frequencies, so results are blobby. Fix with **Fourier features**, γ(p) = [sin(2ᵏπp), cos(2ᵏπp)] for k < L.
### How to compute the encoding dimension
- dim = components × 2L. Position: 3 × 2 × 10 = **60**. Direction (unit vector): 3 × 2 × 4 = **24** (16 with 2 angles).
- Camera poses: given, from **COLMAP** (SfM), or optimised jointly (**BARF**).
- SRN (Scene Representation Network): learn features plus a renderer, a complex optimisation.

## Gaussian splatting
- NeRF is dense (queried everywhere, but scenes are mostly air). Splatting is **sparse**: 3D Gaussians only where density is non-zero.
- Cull Gaussians outside the frustum, project 3D Gaussians to **2D Gaussians**, sort and alpha-composite. Real-time.
- Initialise from an **SfM point cloud**, then **clone and split** (and prune).
`},
{ w: 11, md: `
## Exam logistics (from the slides)
- ~50 MCQ + 2 short-answer questions; 2 h + 10 min perusal; one double-sided page of notes; calculator for simple arithmetic.
- Plan: ~1 min per MCQ (1 h), then 30 min per short-answer question. Covers lectures, tutorials and assessment.

## Vision Transformer (ViT)
- Same as an LLM, but tokens = **image patches**. Project each patch to an embedding and add a **2D positional encoding**, then transformer blocks.
- Transformer blocks are now common inside UNets.

## LLM and attention refresher
- Tokens (e.g. 9 of ~50,000) → embeddings + PE (9 × 16) → N blocks → ~50,000 next-token probabilities.
- A = softmax(QKᵀ) with a causal mask; rows sum to 1; output AV. Often D = dim_emb. GPT-3 has 96 layers; DeepSeek-R1 has 61.

## DINO (self-distillation)
- Two augmented views x₁, x₂. The teacher is the **EMA** of the student; **stop-gradient** on the teacher.
- **Centering** (avoids collapse to a constant) + **softmax sharpening**; **cross-entropy** between teacher and student.
- Best classification uses **kNN on the features**. DINOv3 uses a ViT.

## Stable diffusion / latent diffusion
- AE: one-shot; VAE: plus a prior; GAN: discriminator-driven. Diffusion (Langevin, DDPM) and flow matching (ODE) generate in several steps.
- VAE latent trade-off: a small latent (64) is blurry; a large one (2048) reconstructs sharply, but random samples are rubbish.
- **Most latent locations are invalid**, so use diffusion or flow to map low-dimensional noise to valid VAE latents in many small steps.
- Pipeline: VAE encoder → diffusion in latent space (UNet with transformers) → conditions (text, images, maps) encoded and concatenated with or attended by the UNet features → VAE decoder.

## Encoder-only and predictive models
- VAE baseline: the μ and z manifolds aren't identical, so inference can fall outside the learned manifold.
- LeCun: intelligence **doesn't need a decoder**; it's based on **prediction** in representation space.
- Contrastive learning: same object gives the same features; different objects give different features. It's hard: good pairs are scarce and training is long.
- **JEPA**: predict the target embedding s_y from the context s_x, **non-contrastive** (no negatives), possibly conditioned (future, world models).
`},
{ w: 12, md: `
## The whole unit on one page
- **ML vs GOFAI**: learned thresholds and parameters vs human-written rules and physics. Symbolism vs connectionism.
- **Failure modes**: unknown data, data drift, bias.
- **Brain**: computer (exact), System 1 (skills, fast, multitask), System 2 (reasoning, slow, one task).

## Classic methods: how each defines classes
- kNN: neighbours. k-means: centre points. GMM: Gaussians. Decision tree: conditions on each axis. Random forest: an ensemble of trees. SVM: a line with a **buffer zone**. Neuron: a line z = ax + by + c.
- Linear least squares: **W = (AᵀA)⁻¹AᵀB**; polynomial regression via the same formula.
- Experimental design: train / validation (90/10 or k-fold) / secret test from another source.

## Performance
- FN = miss = Type II; FP = false alarm = Type I. Accuracy mixes both costs.
- ROC: screening means high sensitivity; diagnosis means high specificity. Calibration: under- vs over-confident.

## ANN and CNN key concepts
- Backpropagation, gradient descent and optimisers (momentum, Adam), activation functions, overfitting and dropout.
- Convolution, pooling and stride, batch norm and layer norm, classic networks (convnet, VGG, ResNet, UNet), transfer learning, distillation, self-supervision and semi-supervision.
- Layer = mapping one feature space to another; the output is a 1D problem.
- Safety: adversarial attack, backdoor attack, saliency map, responsible AI.

## Sequences and transformers
- N-gram → Word2Vec → RNN → LSTM → attention. LLM: tokens → embeddings + PE → blocks → next-token probabilities. ViT: patches as tokens.

## Generative AI: learning p(x)
- p(x, y) = p(y|x)p(x): high p(y|x) with low p(x) means an uncertain decision.
- GMM (maximum likelihood via EM) → VAE (latent + q(z|x), ELBO) → diffusion (score, SDE) → flow matching (velocity, ODE).
- Latent diffusion: diffusion or flow maps low-dimensional noise to valid VAE latents.

## Implicit neural models
- An MLP approximates any function, models a 2D image or 3D scan, models density and colour from 2D projections (NeRF), or uses a sum of 3D Gaussians (splatting).

## Formula sheet (copy to your one page)
- Conv output **(N − F + 2P)/S + 1**; conv params **(F²·C_in + 1)·C_out**; linear params **n_in·n_out + n_out**; pooling has **0** params.
- Precision TP/(TP + FP); Recall = Sensitivity TP/(TP + FN); Specificity TN/(TN + FP).
- Softmax exp(sᵢ)/Σexp(sⱼ); CE −log p_y; GD w ← w − η∇L; EMA y ← αy + (1 − α)x.
- Gaussian MLE μ̂ = mean, σ̂² = (1/n)Σ(x − μ̂)². KL = ∫p log(p/q), not symmetric.
- Attention softmax(QKᵀ/√D)V, a T × T matrix; per-head dim = D/h.
- VAE: MSE + β·KL, KL = ½Σ(μ² + σ² − ln σ² − 1), z = μ + σε.
- High dimension: ‖z‖ ≈ √D, random vectors ⟂. DDPM: xₜ = √ᾱₜx₀ + √(1 − ᾱₜ)ε, loss ‖ε − ε_θ‖². Flow: target velocity x₁ − x₀.
- NeRF PE dimension = components × 2L (60 for position, 24 for direction). Volume rendering C = Σ Tᵢαᵢcᵢ.

> Exam strategy: about 1 minute per MCQ. For "combination" questions mark each statement T/F first. For "NOT true" questions, find the one false statement.
`}
];
