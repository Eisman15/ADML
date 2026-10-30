// Study notes, one chapter per week. Light markdown: "## " section, "### " sub-heading, "- " bullet,
// "1. " numbered step, "> " exam tip, **bold**, `code`, blank line = new paragraph.
const NOTES = [
{ w: 1, md: `
## Big picture: AI, ML, DL
- **AI** = intelligence demonstrated by machines: systems that do tasks that normally need human intelligence (perception, reasoning, language, planning).
- **ML** = the subset of AI where the machine **learns its parameters from data** instead of a human programming the rules.
- **Deep learning** = ML with multi-layer neural networks. **Generative AI / LLMs** sit inside DL. AI ⊃ ML ⊃ DL ⊃ GenAI.
- Tom Mitchell's definition: a program learns from experience **E** on task **T**, measured by performance **P**, if its performance on T (measured by P) improves with E.
- Many ML ideas (least squares, Bayes' rule) date from the 1800s; they just weren't called ML.

## Classic AI vs modern AI
### GOFAI / symbolism ("programmed")
- A human expert writes the knowledge: IF–THEN rules, decision logic, or an analytical physics model with only a few fitted parameters.
- Examples: expert systems (MYCIN), chess engines with hand-written search and evaluation, theorem provers.
- Strengths: transparent, explainable, precise. Weaknesses: **brittle**, doesn't scale, can't handle ambiguity or noisy perception (the knowledge-acquisition bottleneck).
### Connectionism / ML ("trained")
- The machine learns thresholds (classification) or the parameters of a generic model (regression) from data. Knowledge is stored as distributed numeric weights.
- Strengths: handles noisy, high-dimensional data (images, speech, text). Weaknesses: needs lots of data and compute, opaque ("black box"), can learn biases.
- Connectionism was held back by slow computers and lack of data, then enabled by PCs, the Internet (data) and GPUs (compute).

## Tasks and learning paradigms
- **Classification**: predict a **discrete class**, often with a probability per class. Examples: spam / not spam, pneumonia / healthy X-ray, benign / malignant tumour, digit 0–9, 1 of 1000 ImageNet objects.
- **Regression**: predict a **continuous value**. Examples: house price, krill density, heart rate, temperature, a patient's age from an X-ray.
- **Supervised**: learn from labelled pairs (x, y) a mapping f(x) ≈ y. Labelling is expensive and is a big business.
- **Unsupervised**: only x; discover structure (clusters, densities, low-dimensional maps). Examples: K-means customer segments, GMM brain-tissue segmentation, PCA / t-SNE visualisation, anomaly detection.
- Related: **self-supervised** (the data makes its own labels, e.g. predict the next word), **semi-supervised** (few labels + many unlabelled), **reinforcement learning** (learn from rewards).

## ML as "creating maps"
- An image is just numbers: 4000 × 3000 × 3 = 36 million values. Text, audio and tables can also be converted to numbers.
- Layers of additions and multiplications map those numbers to a **location on a map** (a vector). That location is the **embedding / latent variable**. Similar inputs land close together.
- **Embedding**: a vector representing an object (word, image, user) so that similar things are nearby ("king" near "queen"; two photos of the same dog near each other).
- **Latent variables**: hidden factors that explain the data (for a face: pose, lighting, age, expression). Moving through the latent space changes those factors.
- **Classification** = read which region of the map the location falls in. **Generative AI** = pick a location and decode it into new data. A trajectory on the map gives a movie. ChatGPT repeatedly guesses the next location (token).
- Maps can be aligned across data types (text ↔ image), which is how text-to-image models work.

## History: two paradigm shifts
- 1872: impressionism (Monet, Impression, Sunrise) moved painting from accurate models to data-driven perception.
- 2012: a CNN (AlexNet) won ImageNet (14M images, 20,000+ categories), moving AI from models to data. Triggered by **GPU compute plus big data**.
### Timeline
- 1943 McCulloch–Pitts neuron; 1950 Turing test; 1956 Dartmouth workshop coins "AI"; 1958 Rosenblatt's perceptron.
- 1969 Minsky & Papert show a single-layer perceptron can't solve XOR, leading to the **first AI winter**.
- 1980s expert systems boom; 1986 backpropagation popularised (Rumelhart, Hinton, Williams); then the **second AI winter** as expert systems proved brittle.
- 1990s–2000s statistical ML: SVMs, decision trees, random forests, boosting; LeNet for digits; Deep Blue beats Kasparov (1997).
- 2012 AlexNet; 2014 GANs; 2015 ResNet; 2016 AlphaGo; 2017 the **Transformer** ("Attention is all you need").
- 2018 onward: foundation models and LLMs (BERT, GPT, Claude), diffusion models, multimodal models, and the AGI debate.

## What could go wrong (repeated in weeks 5 and 12)
1. **Unknown data / lack of generalisation**: the input is unlike anything in training, so the location on the map is meaningless. Examples: a wolf-vs-husky classifier that really learned "snow = wolf"; a pneumonia model that learned hospital-specific markers and failed at a new hospital; a car trained in sunny California failing in snow.
2. **Data drift**: after deployment the data changes. **Covariate drift** p(x) changes (new scanner, new camera); **concept drift** p(y|x) changes (fraudsters adapt, COVID changes shopping); **prior drift** p(y) changes (disease prevalence rises). Fix: monitor inputs and performance, retrain.
3. **Bias**: the mapping is systematically wrong for a subgroup (e.g. black patients mapped near healthy white volunteers). Sources: **class imbalance, sampling, and human labelling**. Real cases: face recognition worse on darker skin; Amazon's hiring model penalising "women's"; a health algorithm using past spending as a proxy for need.

## Computers vs humans (Kahneman, Thinking, Fast and Slow)
- **Traditional computer**: exact, vast memory; fast; reliable; only solves problems coded by humans.
- **System 1**: fast, automatic, intuitive, unconscious; solves **known** problems; requires learning; relatively fast and can **multitask** (recognise a face, read emotion, 2 + 2). Prone to heuristics and biases.
- **System 2**: slow, deliberate, conscious reasoning; solves **unknown** problems; effortful, **one task** at a time (17 × 24, planning a route, checking an argument).
- Current ML ≈ System 1 skills (pattern recognition). Reasoning about never-seen problems is still open; chain-of-thought / reasoning models aim at System 2. Many argue AGI needs both.

## Intelligence, AGI, ASI
- **Intelligence**: no single agreed definition. Commonly the ability to learn, reason, adapt and achieve goals across many environments (Legg & Hutter), including generalising from little data, abstraction, planning and common sense.
- **Narrow AI (ANI)**: excels at one task (chess, image classification). All current systems are essentially this, though LLMs are broader.
- **AGI** = human-level ability across essentially any intellectual task, transferring to new domains without task-specific retraining. **ASI** = substantially beyond the best humans in virtually all domains.
- Open questions: how to test it (Turing test, ARC benchmark)? Can scaling alone reach AGI? Would ASI be aligned with us?
- Efficiency gap: the brain uses ~20 W and learns from very few examples; today's models need huge data and energy.

## Ethical issues of human-level AI
- **Safety and alignment**: AI goals must match human values; risk of losing control of very capable systems.
- **Jobs and economy**: automation of cognitive work, inequality, concentration of power in a few companies or countries.
- **Bias and fairness** amplified at scale; **misinformation** (deepfakes, persuasion, election interference); **privacy and surveillance**.
- **Responsibility / accountability**: who answers for decisions made by an opaque model? **Misuse**: cyberattacks, weapons (dual use).
- **Moral status** of an AGI; **governance** (EU AI Act, international coordination, open vs closed models).

> Exam traps: System 1 is the **multitask** one (System 2 does one task at a time), and data bias can come from **all three** sources: class imbalance, sampling and human labelling.
`},
{ w: 2, md: `
## Supervised ML recipe
1. Get data and inspect it (plots, correlations, outliers).
2. Choose the simplest possible model.
3. Is there an analytical (closed-form) solution? Use it.
4. Can the parameter space be inspected? Use a grid search.
5. Otherwise optimise iteratively (gradient descent).
6. Analyse the results on test data.
- **Objective**: learn f(x; θ) that predicts correct outputs for **new, unseen** inputs, by minimising a loss between ŷ and y on the training set.
- **Assumption**: a relationship y = f(x) + noise exists, and training and test data are **i.i.d.** samples from the **same distribution** p(x, y). If that fails (drift), good training results don't transfer.
- "All models are wrong, but some are useful" (George Box).

## Correlation
- **Pearson** r = cov(x, y) / (σₓσᵧ) ∈ [−1, 1] measures **linear** correlation. **Spearman** = Pearson on the ranks, so it captures **monotonic** non-linear relationships.
- Why check before regression: a line explains nothing if there's no linear association; to pick informative features; to detect **multicollinearity** (correlated inputs make XᵀX near-singular and coefficients unstable); to spot outliers and curves in scatter plots.
- Careful: r ≈ 0 does **not** prove a feature is useless. A strong non-linear relationship (e.g. y = x² on symmetric x) can have r ≈ 0.

## Linear regression and OLS
- Single variable: ŷ = mx + c (slope m, intercept c). Multiple variables: ŷ = w₀ + w₁x₁ + … + w_d x_d. Matrix form for N samples: y = Xw + ε with noise ε.
- Residual: eᵢ = yᵢ − ŷᵢ. Loss: sum (or mean) of squared residuals, E = Σ eᵢ².
### How to derive OLS
1. Stack the samples: A = [[x₁, 1], …, [x_N, 1]], W = [m, c]ᵀ, B = [y₁ … y_N]ᵀ, so AW ≈ B.
2. Cost E(W) = ‖AW − B‖² = (AW − B)ᵀ(AW − B) = BᵀB − 2WᵀAᵀB + WᵀAᵀAW.
3. Derivative: ∂E/∂W = 2Aᵀ(AW − B). Set it to 0.
4. Normal equations: AᵀAW = AᵀB, so **W = (AᵀA)⁻¹AᵀB** (A⁺ = (AᵀA)⁻¹Aᵀ is the pseudo-inverse).
5. E is a convex quadratic, so this is the **global** minimum. Solvers use QR/SVD in practice, which also copes with rank-deficient A.
- One-variable closed form: m = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)², c = ȳ − m·x̄.
### Multiple and polynomial regression
- **Multiple variables**: add a column per feature. Each wⱼ = expected change in y per unit of xⱼ, holding the others fixed (e.g. wine quality from alcohol, acidity, sulphates).
- **Polynomial**: ŷ = w₀ + w₁x + … + w_M x^M. Build the design (Vandermonde) matrix with rows [1, xᵢ, xᵢ², …, xᵢ^M]. The model is still **linear in W**, so the same OLS formula applies. In sklearn: PolynomialFeatures then LinearRegression.
- A 1st-order polynomial **is** linear regression. "Linear" means linear in the parameters, so polynomial regression is still linear regression.

## Assumptions and limitations
### Linear regression assumptions
- **Linearity** between inputs and output.
- **Independence** of observations / residuals.
- **Homoscedasticity**: constant residual variance (no funnel shape in the residual plot).
- **Normality** of residuals with mean 0 (needed for confidence intervals).
- Little **multicollinearity** between features.
### Limitations
- Can't capture complex or non-linear relationships (underfits).
- Sensitive to outliers (the error is squared).
- Extrapolation beyond the data range is unreliable.
- **Simpson's paradox**: a trend that appears inside groups reverses when the groups are pooled.
### Polynomial regression
- Assumes a curvilinear relationship a polynomial can approximate. The **order M is the hyperparameter**: too low underfits (high bias), too high overfits (high variance), chosen on validation data.
- High orders oscillate wildly between points (Runge's phenomenon), explode outside the data range, and make XᵀX ill-conditioned; scaling features helps.

## Three ways to find parameters
- **OLS**: closed form, exact, one step. Only for linear-in-parameter models with squared loss.
- **Grid search**: choose a range and step for each parameter, evaluate the loss on every combination (the Cartesian product), keep the best. Simple and needs no gradients, but costs **kᵖ** evaluations for k values of p parameters (curse of dimensionality) and is only as accurate as the grid.
- **Gradient descent**: start somewhere, repeat **w ← w − η ∂E/∂w**, **b ← b − η ∂E/∂b**. For MSE: ∂E/∂w = −(2/N) Σ xᵢ(yᵢ − ŷᵢ), ∂E/∂b = −(2/N) Σ (yᵢ − ŷᵢ).
- In PyTorch each step is: zero_grad() → forward pass (loss) → loss.backward() (autograd) → optimizer.step().
- On un-normalised data (one variable ~0.05, another ~30) the loss surface is a long narrow valley and GD **zig-zags**, so **normalise** first: z = (x − μ)/σ.
- **Parameters** (w, b) are learned from the training loss; **hyperparameters** (degree, K, learning rate) are chosen with validation data.

## Experimental design
- **Train** fits the parameters, **validation** picks hyperparameters, **test** is kept secret, used once, ideally from a different source (e.g. another hospital).
- **k-fold CV**: split into K folds, train on K − 1 and validate on the remaining one, rotate, and average. Use it when data is small or expensive.
- **Overfitting**: the model learns the noise. Signs: **low training error, high validation error**; on learning curves training loss keeps falling while validation loss starts **rising**.
- Remedies: simpler model, more data, regularisation (L1/L2), early stopping, pruning (trees), dropout and augmentation (networks).

## kNN (K nearest neighbours)
1. Normalise the features (otherwise large-range features dominate the distance).
2. Store the training data. There is no training phase: the model is the dataset (a **lazy**, non-parametric method).
3. Compute the distance (e.g. Euclidean) from the new point to every training point.
4. Take the K closest.
5. Majority vote (classification) or mean / distance-weighted mean (regression).
- **Assumption**: points close in feature space have similar labels, and the distance metric is meaningful.
- Pick K on validation data: small K is noisy (overfits), large K is over-smooth (underfits). Odd K avoids ties in binary problems.
- Pros: simple, interpretable, no training. Cons: slow and memory-heavy at prediction, sensitive to scale, irrelevant features, imbalance and **high dimension** (distances lose meaning).

## Decision trees and random forests
- A tree is a sequence of axis-aligned rules (income < $35k?). Internal nodes test a feature, branches are outcomes, leaves give the prediction.
- Built **greedily top-down**: at each node pick the feature and threshold that best separates the data, measured by **Gini impurity** or **entropy / information gain** (classification), or **variance reduction** (regression). Stop at max depth, min samples, or pure nodes.
- Assumes variables can be split to separate the classes.
- Pros: **interpretable**, handle numerical and categorical features, no scaling needed, capture interactions. Cons: **overfit** (deep trees memorise), blocky axis-aligned boundaries, **unstable** (small data change → different tree), greedy so not globally optimal.
### Pruning
- **Pre-pruning** (early stopping): max depth, min samples to split / per leaf, min impurity decrease, max leaves. Cheap but may stop before a useful later split.
- **Post-pruning**: grow the full tree, then remove branches that don't help validation. Reduced-error pruning replaces a subtree with a leaf if validation accuracy doesn't drop. Cost-complexity pruning minimises error + α·(number of leaves), with α chosen by CV.
### Random forest
- An **ensemble** of many trees, each trained on a **bootstrap sample** (bagging: sampling with replacement) and a **random subset of features** at each split, combined by **majority vote** (classification) or average (regression).
- The randomness decorrelates the trees, so averaging **lowers variance** and generalises better. Costs interpretability and compute; gives feature importances and out-of-bag error for free.

## Confusion matrix
### How to build one
1. Train the model on the training set and predict on the test set.
2. Make a C × C table: rows = **ground truth**, columns = **prediction** (state your convention).
3. For every test sample add 1 to cell [true class, predicted class].
- The diagonal is correct; off-diagonal cells show **which classes are confused** with which. Multi-class: focus on one class to get its TP / FP / FN / TN.
- Binary cells: **TP** (actual +, predicted +), **FN** (actual +, predicted −, Type II error), **FP** (actual −, predicted +, Type I error), **TN**.
- Metrics: accuracy = (TP + TN)/total; precision = TP/(TP + FP); recall = TP/(TP + FN); specificity = TN/(TN + FP); F1 = 2PR/(P + R). Per-class views reveal problems that accuracy hides under **class imbalance**.

> Exam traps: polynomial regression is still solved with OLS because it is linear in the parameters, and grid search cost grows exponentially (kᵖ) with the number of parameters.
`},
{ w: 3, md: `
## From linear classifier to neuron
- Class score **z = w·x + b** (for many classes: s = Wx + b, one row of W per class; predict the argmax).
- The **decision boundary** is z = 0: a line in 2D, a plane in 3D, a hyperplane in general. **w is perpendicular** to it (for ax + by + c = 0 the normal is (a, b)), so w sets the **orientation**; b **shifts** the boundary away from the origin (distance |b|/‖w‖).
- For multiple classes, the boundary between classes j and k is where sⱼ = sₖ.
- The score is a scaled signed distance (true geometric distance = z/‖w‖): the sign gives the class, the magnitude gives "confidence". It is not a calibrated probability; use sigmoid (binary) or softmax (multi-class) to get probabilities.
- **Neuron** = linear classifier + activation: a = f(w·x + b). One neuron gives one linear boundary. A neuron is a projection (dot product) of x onto w.
- A single sigmoid neuron = logistic regression (still a linear boundary). Non-linearity only pays off when neurons are **stacked in layers**.

## Layers as geometric transformations
- A layer computes **h = f(Wx + b)**: a linear map (rotate, scale, shear, project) followed by a non-linearity (bend / fold). Each layer maps one feature space to another so the classes become **linearly separable** for the next layer.
- The **last layer is a linear classifier** (s = Wh + b, then softmax) on **learned features h** instead of raw x. That's why a linear last layer can solve non-linear problems. All layers are trained jointly, end to end.
- **Without activations**, W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) = Wx + b, so any depth collapses to one linear layer.
- A 2-layer ANN (one hidden layer, enough units) is a **universal approximator**: it can approximate any continuous function.

## Loss functions
- A loss turns "how wrong" into one number to minimise. It defines the objective, gives the **gradients** that drive learning, and encodes preferences (punish big errors, punish confident wrong answers). Training finds θ̂ = argmin (1/N) Σ L(f(xᵢ; θ), yᵢ) (+ regularisation).
### Regression losses
- **MSE (L2)**: (1/N) Σ (y − ŷ)², punishes large errors, sensitive to outliers.
- **MAE (L1)**: (1/N) Σ |y − ŷ|, robust to outliers.
- **Huber / smooth L1**: quadratic for small errors, linear for large ones.
### Classification losses
- **Softmax**: pᵢ = exp(sᵢ)/Σⱼ exp(sⱼ), turning scores into probabilities that sum to 1.
- **Cross-entropy (CE)** = −log p_correct: maximise the softmax probability of the correct class. Binary form: −[y log p + (1 − y) log(1 − p)].
- **Hinge (multiclass SVM)**: Σ over j ≠ y of max(0, sⱼ − s_y + Δ). The correct score must beat every other score by margin Δ.
### Hinge vs cross-entropy
- Hinge works on raw scores and becomes **exactly 0** once the margin is met: it is "satisfied" and stops caring, so well-classified points give zero gradient.
- CE works on probabilities and **never reaches exactly 0**: it always pushes for more confidence, so every point gives some gradient (smaller when confident).
- CE gives probabilistic outputs; hinge gives uncalibrated scores. Other losses: focal loss (imbalance), KL divergence (distributions).
### How to compute a softmax / CE example
1. Scores s = [2, 1, 0]. exp gives [7.39, 2.72, 1.00], with sum 11.11.
2. p = [0.665, 0.245, 0.090]. If the correct class is 0: CE = −ln 0.665 = **0.41**.
3. If the correct class were 2: CE = −ln 0.090 = 2.41, so a confident wrong answer costs much more.
### How to compute a hinge example
1. Same scores [2, 1, 0], correct class 0, Δ = 1.
2. Class 1: max(0, 1 − 2 + 1) = 0. Class 2: max(0, 0 − 2 + 1) = 0. Loss = **0** (margin met).
3. If the correct class were 1: max(0, 2 − 1 + 1) + max(0, 0 − 1 + 1) = 2 + 0 = **2**.

## Gradients and the update rule
- The gradient ∇L is the vector of partial derivatives ∂L/∂wᵢ. It points towards **steepest increase**, so step the opposite way: **w ← w − η ∇L(w)**.
- Found analytically with the **chain rule**: ∂L/∂w = (∂L/∂ŷ)(∂ŷ/∂z)(∂z/∂w). Linear model + MSE: ∂L/∂w = −2(y − w·x)x. Softmax + CE: ∂L/∂sₖ = pₖ − yₖ (prediction minus one-hot target).
- Numerical check: [L(w + h) − L(w − h)] / 2h (finite differences, used for gradient checking).
- Initialise weights **randomly** (not all equal): identical weights get identical gradients, so the units never become different (symmetry).

## Gradient descent and variants
- **Batch GD**: gradient over **all** data per update. Accurate, stable, slow and memory-heavy.
- **SGD**: **1 sample** per update. Cheap but very noisy; the noise can escape shallow minima but jitters around the minimum.
- **Mini-batch**: **B samples** (32–256), the usual choice. Balances noise and speed and uses GPU parallelism (often just called "SGD").
- **Momentum**: v ← βv + ∇L, w ← w − ηv (β ≈ 0.9). A heavy ball: speeds up along consistent directions, damps oscillation in narrow valleys, rolls through small bumps and plateaus.
- **AdaGrad**: divides by the accumulated sum of g², so steps keep shrinking (can stop too early).
- **RMSProp**: divides by an **EMA** of g², so the step size adapts per parameter without dying out.
- **Adam** = momentum + RMSProp, the most used optimiser:
1. mₜ = β₁mₜ₋₁ + (1 − β₁)gₜ (first moment, **β₁ ≈ 0.9**).
2. vₜ = β₂vₜ₋₁ + (1 − β₂)gₜ² (second moment, **β₂ ≈ 0.999**).
3. Bias-correct: m̂ = mₜ/(1 − β₁ᵗ), v̂ = vₜ/(1 − β₂ᵗ).
4. Update: w ← w − η m̂/(√v̂ + ε), ε ≈ 1e-8.
- Adam converges faster with less tuning; SGD + momentum sometimes generalises better (still used for big vision models).

## EMA, learning rate, batch size, epochs
- **EMA**: yₜ = αyₜ₋₁ + (1 − α)xₜ. It stores only one value, whereas a centred moving average over ±N stores 2N + 1. α = 0.9 averages roughly the last 1/(1 − α) = 10 values. Used in momentum, Adam, smoothing loss curves, batch-norm running stats, teacher / EMA weights.
- **Learning rate η**: too small is slow or gets stuck; too large oscillates or diverges. Start large, then decay (schedules).
- **Batch size**: samples per gradient estimate. Larger = more accurate gradient, more memory, fewer updates; smaller = noisier (mild regularisation), more updates.
- **Epoch** = one full pass over the training set. **Iterations per epoch = N / batch size**; e.g. N = 10,000 and B = 100 gives 100 updates per epoch.

## Deep learning
- DL = ML with **more than one hidden layer**. Classic ML uses **hand-engineered features** (fine for small tabular data, more interpretable); DL **learns hierarchical features** from raw data (pixels → edges → textures → parts → objects), needs big data and GPUs, and keeps improving with scale.
- **Depth** = number of layers (more abstract features, harder to train). **Width** = units per layer (more capacity per layer).
- **Forward pass**: compute layer by layer up to the loss, storing activations. **Backward pass**: propagate ∂L back from the output to every weight. **Backpropagation** = doing that efficiently with the **chain rule**, reusing stored activations, then a gradient-descent step.
### Activation functions
- **Sigmoid** σ(z) = 1/(1 + e^(−z)): output (0, 1), not zero-centred, saturates; σ' = σ(1 − σ) ≤ **0.25**, so gradients **vanish** when multiplied through many layers.
- **Tanh**: output (−1, 1), zero-centred, still saturates.
- **ReLU** max(0, z): cheap, no saturation for z > 0, sparse; the default for hidden layers. Can produce "dead" units (fix: Leaky ReLU, GELU).
- **Softmax** at the output for multi-class probabilities.

## Overfitting in deep nets
- Networks have huge capacity (often more parameters than samples), so they can **memorise** the training set, even random labels, and exploit **shortcuts** (background, watermarks).
- Identify: training vs validation curves (validation loss rising while training loss falls), a large train–test gap, testing on external data.
- Mitigate: more data and **augmentation**, **early stopping**, **regularisation** (L2 weight decay, L1), **dropout** (randomly drop units during training), batch norm, a smaller model, **transfer learning**, ensembles, label smoothing.

## How to count MLP parameters
- Layer (n_in → n_out): **n_in·n_out + n_out** (weights + one bias per output unit).
- Example 2 → 10 → 1: (2·10 + 10) + (10·1 + 1) = 30 + 11 = **41**.
- Example 784 → 128 → 10: (784·128 + 128) + (128·10 + 10) = 100,480 + 1,290 = **101,770**.

> Exam traps: hinge loss can be exactly 0 but cross-entropy can't; stacking linear layers without activations is still one linear layer; sigmoid's derivative is at most 0.25.
`},
{ w: 4, md: `
## Computer vision tasks
- **Classification**: one label for the whole image (assumes one main object). Output: class probabilities. Example: "dog".
- **Object detection**: class + **bounding box** + confidence for each object (0 or more objects). Examples: YOLO, Faster R-CNN.
- **Semantic segmentation**: a class for **every pixel** (road, pedestrian). It has no notion of objects, only pixels.
- **Instance segmentation**: class + **mask per object**, so pedestrian 1 and pedestrian 2 are separated.
- Granularity increases: image level → box level → pixel level.

## Dataset assumptions
- The dataset distribution **matches the application** (lighting, viewpoint, camera, background, demographics) and train and test are i.i.d.
- The **ground truth is unambiguous** and correct (consistent annotation).
- The data is large and diverse enough, with classes balanced or the imbalance handled, and the images contain the information needed (no reliance on spurious cues).
- These are often violated, so an 87% benchmark score may not transfer to deployment.

## Why convolution
- An MLP ignores spatial order (shuffling the pixels in the same way gives the same result) and needs huge weight matrices: ~150,000 weights **per neuron** for a 224×224×3 image.
- Convolution = a sliding **match filter**: the dot product is high where the image patch matches the template (a "nose").
- **Local connectivity**: each output looks at a small neighbourhood, because nearby pixels are strongly related (edges, textures).
- **Weight sharing**: the same filter slides everywhere, so the parameter count is tiny and **independent of image size**, and a pattern is detected anywhere.
- It keeps the spatial structure, stacks into a **hierarchy** (edges → shapes → objects), and fewer parameters means less overfitting.
### Invariance
- Convolution is translation **equivariant**: shift the input and the feature map shifts by the same amount. With pooling this becomes approximate translation **invariance**.
- It is **not** rotation, scale or viewpoint invariant, so those are learned through **data augmentation**.

## Convolutional layer: parameters and hyperparameters
- **Learned parameters**: K filters, each **F × F × C_in** (the filter depth always equals the input channels), plus one bias per filter.
- **Output**: a feature map volume of size H_out × W_out × K, one 2D map per filter showing where its pattern occurs.
- **Hyperparameters**: number of filters K (output depth), kernel size F (3×3 is common), stride S (larger downsamples), padding P (zeros at the border; "same" keeps the size), plus dilation and groups.
### How to compute conv output size and parameters
- **Output = (N − F + 2P)/S + 1**. N = input size, F = filter, P = padding, S = stride.
- Examples: 7×7 input, F = 3, S = 1 gives 5×5; S = 2 gives 3×3; S = 3 gives 2.33, which doesn't fit (invalid).
- 32×32, F = 5, P = 0, S = 1 → 28×28. 32×32, F = 3, P = 1, S = 1 → 32×32.
- "Same" size at stride 1: F = 3, P = 1 (or F = 5, P = 2). In general P = (F − 1)/2.
- **Params = (F·F·C_in + 1)·C_out**.
- Examples: Conv2d(3, 5, k = 3) gives (3·3·3 + 1)·5 = **140**. 6 filters 5×5 on RGB gives (75 + 1)·6 = **456**. Conv2d(16, 32, k = 3) gives (144 + 1)·32 = **4,640**.
- Tensor layout in PyTorch: **[B, C, H, W]**, e.g. [100, 3, 224, 224] = 100 RGB images of 224×224.
### Worked chain (practice exam Q10)
1. [3, 64, 64] → Conv 16@3×3 P1 → [16, 64, 64] → ReLU (same shape) → MaxPool 2×2/2 → [16, 32, 32].
2. Conv 32@3×3 P1 → [32, 32, 32] → ReLU → MaxPool → **[32, 16, 16]**.
3. Flatten → 32·16·16 = 8,192 features into the linear head.

## Pooling and downsampling
- Purpose: **downsample** (less computation in later layers), **grow the receptive field**, add some **invariance to small shifts**, and help control overfitting.
- **Max pooling** keeps the strongest activation in each window (2×2, stride 2 halves H and W). **Average pooling** takes the mean. **Global average pooling** collapses each map to one number before the classifier (ResNet).
- Pooling has **no learnable parameters**; 3 hyperparameters (type, size, stride).
- A **strided conv** is a learnable alternative for downsampling. A **transpose conv** upsamples.

## What CNNs learn
- Early layers learn edges, colour blobs and contrast; deeper layers learn textures, parts and objects. The **receptive field grows with depth**.
- Generalisation is hard because of viewpoint, illumination, scale, deformation, occlusion, background clutter and intra-class variation, plus **domain shift** (new cameras, hospitals, weather), **shortcuts** (backgrounds, watermarks) and adversarial perturbations.

## Regularisation and normalisation
### Dropout vs BatchNorm
- **Dropout**: randomly zero a fraction p of activations during **training only**. Purpose: **regularisation**; stops neurons co-adapting, like training an ensemble. No learnable parameters. Turned off at test time.
- **BatchNorm**: normalise each feature over the mini-batch to mean 0, variance 1, then apply learnable scale **γ** and shift **β**. Purpose: **stabilise and speed up training**, allow higher learning rates, robust to bad initialisation (mild regularisation). At test time it uses the **running** mean and variance.
- **Input normalisation** per channel: (x − μ)/σ, using the training-set (or ImageNet) statistics.
### Data augmentation
- Label-preserving random transforms: flips, crops, rotations, scaling, colour jitter, brightness / contrast, noise, blur, cutout, mixup (torchvision transforms).
- Enlarges and diversifies the data, reduces overfitting, and teaches invariances CNNs lack (rotation, scale, lighting). It is also the core of contrastive self-supervised learning (week 6).
- The transform must keep the label: flipping a "6" can make a "9"; flipping a chest X-ray moves the heart to the wrong side.

## Segmentation: UNet
- Without pooling, each 3×3 conv adds 2 pixels of receptive field, so ~112 layers are needed to cover 224 pixels. That's too deep, and bigger kernels explode the parameter count.
- **Encoder** (contracting path): conv + pooling capture **context ("what")** while reducing resolution.
- **Decoder** (expanding path): up-convolutions restore the resolution for a per-pixel output.
- **Skip connections** concatenate encoder maps with decoder maps at the same resolution, passing **fine detail ("where")** so boundaries are precise. Multiscale.
- Fully convolutional, trains well with **little data** (originally biomedical), ubiquitous in medical imaging and the backbone of diffusion models.

## Very deep networks: ResNet
- The chain rule multiplies many small factors over many layers, so gradients **vanish**; plain deeper nets even got **higher training error** (the degradation problem).
- A **residual block** computes **y = F(x) + x**: the identity shortcut carries the gradient straight back, and a useless block can simply learn F(x) = 0. This allowed 50, 101, 152+ layers (ResNet won ImageNet 2015).
### Transfer learning
1. Load a **pre-trained** model (e.g. a ResNet trained on ImageNet from torchvision). Early layers hold general features (edges, textures).
2. **Replace the head**: the final linear layer gets the new number of classes (e.g. 1000 → 10).
3. Either **freeze** the backbone and train only the head (feature extraction), or **fine-tune** all or the last layers with a **small learning rate** so the old knowledge isn't destroyed.
- Benefits: needs less data, trains faster, usually more accurate.

> Exam traps: pooling has no learnable parameters; the filter depth always equals the input channels; convolution is translation equivariant but not rotation or scale invariant.
`},
{ w: 5, md: `
## Train / validation / test, again
- Hyperparameters are chosen on validation (or k-fold CV); the test set is used **once**, ideally from a different source.
- **model.train() vs model.eval()**: Dropout(0.1) zeroes 10% of activations in train mode and nothing in eval. BatchNorm uses batch statistics in train mode and running statistics in eval.
- Testing in train mode, especially with batch size 1, gives wrong (and random-looking) results. Also wrap evaluation in torch.no_grad().

## Two types of mistakes
- **False positive** = false alarm = **Type I** error ("overdo a task"). **False negative** = miss = **Type II** error ("not very good at a task").
- Costs are asymmetric: a missed cancer or a missed fall of an elderly person is far worse than a false alarm; a false positive pregnancy test has its own cost.
- **Accuracy mixes both** error types into one number, so it's rarely a good target, especially under class imbalance.

## How to compute the metrics
- From TP, FP, FN, TN:
- **Accuracy** = (TP + TN)/(all).
- **Precision (PPV)** = TP/(TP + FP): of the predicted positives, how many are real.
- **Recall = Sensitivity = TPR** = TP/(TP + FN): of the real positives, how many were found. Recall is the ML term, sensitivity the medical term: **same metric**.
- **Specificity (TNR)** = TN/(TN + FP): recall of the negative class. **FPR** = FP/(FP + TN) = 1 − specificity.
- **NPV** = TN/(TN + FN). **F1** = 2PR/(P + R), the harmonic mean, which is low if either P or R is low.
### Worked example
1. TP = 40, FN = 10, FP = 20, TN = 930 (1,000 cases, 50 positive).
2. Precision = 40/60 = 0.667; Recall = 40/50 = 0.80; Specificity = 930/950 = 0.979; Accuracy = 970/1000 = 0.97.
3. F1 = 2·0.667·0.8/(0.667 + 0.8) = 0.727.
4. The 97% accuracy hides that one in three alarms is false.
- Imbalance trap: 1% prevalence and "always healthy" gives **99% accuracy with 0% sensitivity**.
- **Multi-class**: focus on one class (e.g. "Cold") and group the rest as negative, then read TP / FP / FN / TN from the confusion matrix.

## ROC curve
1. The model outputs a score or probability for the positive class on a test set.
2. Sweep a threshold t from high to low (e.g. every unique score).
3. For each t, compute TPR (sensitivity) and FPR (1 − specificity).
4. Plot TPR (y) against FPR (x).
- The diagonal is random and the top-left corner (FPR 0, TPR 1) is perfect. A high threshold sits near (0, 0); a low threshold near (1, 1).
- **AUC**: 1.0 = perfect ranking, 0.5 = random. It equals the probability that a random positive gets a higher score than a random negative.
- Report the AUC or choose the operating point (e.g. closest to the corner, or the one the application needs).
- **Screening**: minimise FN (high sensitivity), because a miss is costly and a cheap follow-up exists. **Diagnosis**: minimise FP (high specificity), e.g. a PET scan (~$2,000) after cheap screening.
- **Low prevalence doesn't change the ROC** (TPR and FPR are computed within each class), but **precision drops**.
### How to compute PPV at low prevalence
1. 10,000 people, 1% prevalence: 100 sick, 9,900 healthy. Sensitivity = specificity = 98%.
2. TP = 98, FN = 2. FP = 2% of 9,900 = 198, TN = 9,702.
3. PPV = 98/(98 + 198) ≈ **33%**: two of every three positives are false alarms.
- For heavy imbalance a precision–recall curve is more informative than ROC.
- Small studies give noisy estimates of all these metrics; increase the sample size and report confidence intervals.

## Calibration
1. For each test prediction store the confidence (max softmax probability) and whether it was correct.
2. Bin by confidence (e.g. 10 bins: 0–0.1, …, 0.9–1.0).
3. In each bin compute the mean confidence and the accuracy (fraction correct).
4. Plot accuracy (y) against confidence (x) and compare with the diagonal.
- Calibrated means on the diagonal (the 50–60% bin is about 55% accurate). Below the diagonal = **over-confident**, which is typical for deep networks trained with cross-entropy.
- Summary number: **ECE** (expected calibration error) = weighted average gap between confidence and accuracy over the bins.
- Fixes: **temperature scaling** (divide logits by T > 1 before softmax, tuned on validation), label smoothing, ensembles.

## Robustness
### Feature shift vs class shift
- **Feature (covariate) shift**: p(x) changes but the classes are the same (new scanner, night, fog, another country's signs).
- **Class shift / open set**: deployment contains **novel classes** never seen in training (a cat shown to a dog-breed classifier). (Label / prior shift, where p(y) changes, is a different thing again.)
- Safety concern: the model **silently fails**. It can't say "I don't know", and often answers with **high confidence**.
### Why novel classes get high confidence
- **Softmax must pick a known class**: probabilities over the known classes always sum to 1, with no "none of the above".
- Cross-entropy pushes outputs towards 0/1, so networks are over-confident in general.
- ReLU logits grow roughly linearly away from the data, so **far-from-boundary points get extreme scores** and near-100% softmax.
- Discriminative models learn boundaries, not where the training data lives, and an unknown object may share textures or colours with a known class.
- Mitigation: out-of-distribution detection, calibration, uncertainty from ensembles, monitoring.
### Adversarial attack (test time)
- A tiny, imperceptible perturbation makes the model wrong with high confidence (panda → gibbon).
- Instead of changing weights to reduce the loss, change the **input** to **increase** it, using the gradient with respect to the input: **x_adv = x + ε·sign(∇ₓL)** (FGSM). This is the shortest step across the decision boundary.
- PGD repeats small FGSM steps; targeted attacks push towards a chosen class; black-box attacks rely on transfer from a surrogate model. Defences: adversarial training, input preprocessing, certified robustness.
### Backdoor attack (training time)
- The attacker **poisons the training data**: adds a trigger (a sticker, pixel pattern or phrase) to some samples and relabels them to a target class. The model learns "trigger → target" (stop sign + yellow sticker → speed limit).
- Invisible on clean tests: accuracy looks normal.
- A danger whenever you don't control the data or model: public / scraped datasets, downloaded **pre-trained models**, outsourced training, federated learning.
### Ensembles
- Combine several models by averaging probabilities or voting. **Bagging** (random forests) reduces variance; **boosting** (AdaBoost, XGBoost) reduces bias; **deep ensembles** use different random initialisations.
- Better accuracy, **calibration** and **robustness**; disagreement between members flags out-of-distribution inputs. Cost: more compute and memory.

## Explainability and ethics
- **Explainable AI**: humans can understand why a prediction was made, which inputs mattered and when to trust it. Needed for trust, debugging shortcuts, fairness audits and regulation.
- **Decision trees** are inherently explainable (a readable if–then path, if not too deep). **CNNs** are black boxes; post-hoc tools only approximate their reasoning.
- Tools: **gradient saliency** (∂score/∂image), **occlusion maps** (cover parts of the image and watch the score drop), **Grad-CAM** (class-specific heat map from the last conv layer), feature visualisation, LIME / SHAP.
- Case study: a CNN can predict **race from chest X-rays** even when humans can't, which is a hidden bias risk.
- Responsible AI along the **value chain** (data, developers, deployers, users). Australia's **AI Ethics Principles**; self-regulation vs government regulation (EU AI Act); AI standards.

> Exam traps: recall and sensitivity are the same metric; low prevalence leaves the ROC unchanged but lowers precision; adversarial attacks use the gradient with respect to the input, not the weights.
`},
{ w: 6, md: `
## Why go beyond supervised
- Supervised learning depends on the **quantity** of labels (ResNet18 has 11M parameters to feed), their **diversity** and their **quality** (noisy labels hurt).
- Labels are **expensive** (radiologist time) and some data is too **ambiguous or complex** to label.
- It learns only the **discriminative features** needed to separate the given classes (e.g. fur pattern), not a general understanding of the data. It also assumes train and test share a distribution.
- **Unsupervised**: only unlabelled data; find structure (clusters, densities, embeddings): K-means, GMM, PCA, t-SNE.
- **Self-supervised**: unlabelled data, but **create targets from the data itself** (a pretext task) and train like supervised learning: predict a masked word, match two augmented views.
- **Semi-supervised**: **few labels plus many unlabelled samples** for the same task: pseudo-labelling, consistency regularisation.

## k-means
1. Pick K random points as the initial centroids.
2. **Assign** each point to the nearest centroid (Euclidean distance).
3. **Update** each centroid to the mean of its assigned points.
4. Repeat 2–3 until the assignments stop changing.
- It minimises the within-cluster sum of squared distances.
- K is a hyperparameter (not kNN's k). Choose it with the elbow method or silhouette score.
- Pros: simple, efficient, scales to large data.
- Cons: choosing K; assumes **spherical**, similar-size clusters (fails on elongated or irregular shapes); **local minima** depending on initialisation (use k-means++ or restarts); sensitive to noise and outliers; **hard** assignments only; needs feature scaling.

## Maximum likelihood: how to derive it for a Gaussian
- **MLE**: choose the parameters that make the observed data most probable, θ̂ = argmax p(data | θ).
1. Independence gives a product: L(μ, σ) = Πᵢ (1/√(2πσ²)) exp(−(xᵢ − μ)²/(2σ²)).
2. Take the log (monotonic; product → sum) and negate: NLL = (n/2) log(2πσ²) + Σᵢ (xᵢ − μ)²/(2σ²).
3. ∂NLL/∂μ = −(1/σ²) Σᵢ (xᵢ − μ) = 0 gives **μ̂ = (1/n)Σxᵢ** (the sample mean).
4. ∂NLL/∂σ² = n/(2σ²) − Σᵢ (xᵢ − μ)²/(2σ⁴) = 0 gives **σ̂² = (1/n)Σ(xᵢ − μ̂)²**.
- σ̂² is **biased** (divides by n); for small n use 1/(n − 1). Negligible for large n.
- With σ fixed (e.g. 1), maximising the likelihood for μ **is** least squares. That's why NLL connects to MSE.

## GMM and EM
- **GMM**: p(x) = Σₖ πₖ N(x | μₖ, σₖ²), with mixing weights Σₖ πₖ = 1. Each component has its own mean, variance (covariance in higher dimensions) and weight.
- **Soft** clustering: every point gets a probability p(k | x) for each cluster. Own covariances let it fit **elliptical clusters of different sizes**, unlike k-means.
- History: Pearson (1894) fitted two Gaussians to Weldon's crab measurements.
### EM steps
1. **Initialise** μₖ, σₖ², πₖ (randomly, or from a template / prior).
2. **E-step**: responsibility γᵢₖ = πₖN(xᵢ|μₖ, σₖ²) / Σⱼ πⱼN(xᵢ|μⱼ, σⱼ²) (Bayes' rule).
3. **M-step**: Nₖ = Σᵢγᵢₖ; μₖ = Σᵢγᵢₖxᵢ/Nₖ (a **weighted** mean); σₖ² = Σᵢγᵢₖ(xᵢ − μₖ)²/Nₖ; πₖ = Nₖ/N.
4. Repeat until the log-likelihood converges. It **never decreases**, but can stop at a **local** maximum.
- k-means = EM with **hard** assignments (closest centre) and equal spherical variances.
- Application: **brain MRI tissue segmentation** (CSF, grey matter, white matter from intensities).
### Clustering applications
- Customer / market segmentation and **recommender systems**; **anomaly detection** (fraud, intrusion, defects: points far from every cluster); medical image segmentation; **exploring unlabelled data** too complex to label; colour quantisation; grouping gene-expression profiles or cell types.

## Visualisation: t-SNE and UMAP
- Non-parametric: they **move the data points** in 2D to form clusters, rather than learning a mapping you can reuse.
- t-SNE: Gaussian similarities P between points in high dimension, **Student-t** similarities Q in 2D (heavy tails stop crowding), then minimise **KL(P‖Q)** by moving the 2D points.
- UMAP is similar but faster and keeps more global structure. Distances between far-apart clusters in t-SNE plots don't mean much.
### KL divergence
- D_KL(p‖q) = ∫ p log(p/q) ≥ 0, = 0 iff p = q, and **not symmetric** (so it isn't a true distance).
- The lecture builds it in steps: naive differences p − q (can cancel), then ratios p/q, then log-ratios weighted by p.

## Transfer learning
- Use a model trained on Dataset A (e.g. 1000 ImageNet classes) and replace the head with 10 outputs for Dataset B.
- **Freezing**: train only the new head (feature extraction). **Fine-tuning**: update all layers with a small learning rate.
- It's a balance between new and old knowledge: too high a learning rate destroys the pre-trained features (catastrophic forgetting).

## Semi-supervised
- Suitable when labels are scarce or expensive but unlabelled data is plentiful **and from the same distribution**.
- Assumptions: **smoothness** (nearby points share labels), **cluster** (the boundary lies in low-density regions), **manifold** (data lies on a lower-dimensional surface). Unlabelled data from other classes or distributions can hurt.
### Pseudo-labelling (self-training)
1. Train on the small labelled set.
2. Predict the unlabelled data.
3. Keep **high-confidence** predictions (above a threshold) as pseudo-labels.
4. Add them to the training set, retrain, repeat.
- Risks: **confirmation bias** (wrong labels get reinforced), relies on calibrated confidence (networks are over-confident), threshold trades quantity against quality, can amplify class imbalance.
### Consistency regularisation
- The prediction for an unlabelled input should **stay the same under different augmentations** (or noise, dropout). Add a loss such as ‖f(aug₁(x)) − f(aug₂(x))‖² to the supervised loss.
- Examples: Π-model, Mean Teacher (EMA teacher), FixMatch (pseudo-label from a weak augmentation, train on a strong one).
- Key choices: **which augmentations** (they must preserve the label) and **the loss weight** (often ramped up); watch for collapse to constant predictions.

## Contrastive learning
1. **Positives** = two random augmentations of the same image (crop, colour jitter, flip, blur); **negatives** = other images.
2. An encoder (CNN backbone + projection head) maps each view to a low-dimensional embedding.
3. A contrastive loss (InfoNCE / NT-Xent) or margin loss **pulls positives together and pushes negatives apart**.
4. Use the trained backbone as pretraining for downstream tasks (linear probe, kNN, fine-tuning). Examples: SimCLR, MoCo.
- Limitations: choosing augmentations and their magnitude (they decide which invariances are learned); choosing negatives (too easy teaches nothing, too hard may be the same class); needs **many negatives** (large batches or memory banks); long, compute-heavy training.

## Predictive self-supervised learning
- **Masked prediction**: predict masked words (BERT) or image patches (MAE). The masking ratio must be tuned; image versions may focus on pixel detail.
- **Next-token prediction**: GPT and LLMs; huge data and compute, sequential generation.
- **Transformation prediction**: predict the **rotation** applied, solve a **jigsaw** of shuffled patches, predict relative patch positions. Risk of shortcuts (edge artefacts) and features tied to the pretext task.
- **Colourisation** (colour from greyscale, ambiguous) and **denoising** (reconstruct clean data).
- General rule: the pretext task must be relevant to the downstream task, or the features transfer poorly.

## Foundation models
- Very large models trained (mostly **self-supervised**) on huge, broad unlabelled data and **adaptable to many tasks** by fine-tuning, prompting or using their features. Examples: **DINOv2** for vision (open source, 142M images), CLIP, BERT, GPT, Claude.
- Advantages: general features, **few labels needed**, one model serves many tasks, emergent abilities such as in-context learning.
- Dangers: **inherited bias** from web data, and any flaw propagates to every downstream use; opacity and hallucination; huge **cost** and **concentration of power** in a few companies; backdoors and data poisoning; privacy, copyright and environmental costs.

> Exam traps: k-means is EM with hard assignments; the MLE variance divides by n (biased); KL divergence is not symmetric; t-SNE moves points rather than learning a reusable mapping.
`},
{ w: 7, md: `
## Sequence data
- **Time series**: observations **ordered in time**, x₁ … x_T (stock prices, ECG, sensors, speech audio). Text can be treated as a time series of tokens.
- Differences from images: **order matters** and samples are **not independent** (trend, seasonality, autocorrelation); sequences have **variable length**; structure runs along one time axis with long-range dependencies; models need **memory** of the past.
- Splits must respect time: no shuffling, no future data leaking into training.
- Applications: **forecasting** (stocks, energy demand, weather, sales, traffic), **prediction or classification** (activity from wearables, ECG / EEG anomalies, predictive maintenance, speech recognition, translation).
- NLP data is text: **sequential, discrete, variable-length, categorical**, with a huge vocabulary, and meaning depends on context and order. Applications: translation, chatbots, sentiment, spam / topic classification, named entities, QA, summarisation, search, code generation.

## Tokenisation
- Pipeline: clean text → **tokenise** → build a **vocabulary** (token → integer ID) → turn IDs into vectors (counts or embeddings) → pad / truncate to a fixed length for batching.
- Shakespeare example: 65 symbols (26 × 2 letters + specials) and 1,115,000 characters.
- **Characters**: small vocabulary, long sequences. **Words**: large vocabulary, short sequences, out-of-vocabulary problems. **Subwords** (BPE, SentencePiece) are the compromise. LLM vocabularies are ~50,000 tokens.

## Classic language models
- Letter frequency gives a naive model (random letters with the right frequencies).
- **Bigram**: P(next | current), counted from data or learned. Generates text that is "not random but still gibberish". N-grams explode as **V^N** possible contexts.
- **Bag of words**: a |V|-sized count (or TF-IDF) vector per document. Simple, fast, interpretable, fine for spam / topic classification. But it **loses order** ("dog bites man" = "man bites dog"), is high-dimensional and sparse, and has no notion of word similarity.
- **Embeddings** (lookup table): each token ID indexes a row of a learnable matrix E (|V| × d), giving a **dense vector** with semantic meaning. The sequence of vectors keeps the order. Word2Vec: king − man + woman ≈ queen.
- A static embedding gives one vector per word regardless of context ("bank" of a river vs a bank); later layers (attention) add the context.

## RNN and LSTM
- A feedforward net takes a fixed-size input in one pass with no memory. An **RNN** reads one element at a time and keeps a **hidden state**: **hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)**, output yₜ = W_y hₜ.
- The **same weights are shared over all time steps**, so it handles any length. Trained with **backpropagation through time** (unroll over the sequence).
- Shapes: **many-to-many** (translation, tagging), **many-to-one** (sentiment), **one-to-many** (generation, captioning).
### RNN limitations
- **Can't parallelise**: step t needs step t − 1, so training is slow on GPUs.
- **Vanishing or exploding gradients**: BPTT multiplies by the same W many times.
- **Short memory**: early information is forgotten; the fixed-size hidden state is a bottleneck.
### LSTM
- Adds a **cell state** cₜ (a memory conveyor belt) and three gates (sigmoids between 0 and 1):
1. **Forget gate** fₜ: what to erase from memory.
2. **Input gate** iₜ with candidate c̃ₜ = tanh(…): what new information to write.
3. Cell update **cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ**. The **additive** update lets gradients flow over many steps.
4. **Output gate** oₜ: hₜ = oₜ ⊙ tanh(cₜ).
- Longer memory and robust to vanishing gradients. GRU is a simpler 2-gate variant.
- Still **sequential and slow**, ~4× the parameters of an RNN, and still struggles with very long sequences.

## Attention
- Self-attention lets every token gather information from **all** other tokens, weighted by relevance, in **parallel**. Any two tokens are one step apart, so distant context isn't forgotten.
### How to compute it
1. Q = XW_q, K = XW_k, V = XW_v. For T tokens: Q, K, V are T × D.
2. Scores S = QKᵀ/√D give a **T × T** matrix (how well each query matches each key). Dividing by √D stops the softmax saturating.
3. Causal mask (decoder): set j > i to −∞ so a token can't see the future.
4. A = softmax(S) row-wise, so **each row sums to 1**.
5. Output = AV (T × D), added to the embeddings (residual): tokens are "nudged" by their context.
- Example: in "The animal didn't cross the street because it was tired", attention links "it" to "animal".
- **Multi-head**: split D into h heads of size D/h (768/12 = 64), compute in parallel, concatenate, project. Different heads capture different relations (syntax, coreference).
- **Positional encoding**: attention by itself ignores order (permuting tokens just permutes outputs), so "not happy because raining" looks like "happy because not raining". Add PE(pos, 2i) = sin(pos/10000^(2i/D)), PE(pos, 2i + 1) = cos(…), a unique pattern per position. Alternatives: learned positions, RoPE.

## Transformer block and LLMs
- Block: multi-head attention + add & norm + feed-forward + add & norm. Stack N blocks (GPT-3: **96**, DeepSeek-R1: **61**).
- Output: a linear layer over the vocabulary, then softmax, giving next-token probabilities. Generate **autoregressively**: sample a token, append it, repeat.
### Training in practice
1. Tokenise a huge corpus (BPE).
2. **Self-supervised pre-training**: GPT-style **next-token prediction** (targets = inputs shifted by 1, causal mask, cross-entropy over the vocabulary); BERT-style **masked tokens** (~15%).
3. Mini-batches, Adam / AdamW, learning-rate warm-up and decay, dropout, layer norm, many GPUs.
4. Adapt: supervised fine-tuning, instruction tuning, **RLHF** for chat assistants, LoRA, or just prompting / in-context learning.
- Strengths vs RNN: **parallel training**; **global context** via self-attention; scales with data and compute.
- Limitations: **data hungry**, **O(T²)** attention cost in memory and compute (limits context), no built-in sense of order, weak inductive bias on small data, hallucination and bias in LLMs.
- Long context: **KV cache** (reuse past K and V at generation time), linear or windowed attention. Context windows of 128k+ tokens.

## Baby LM tutorial
- Addition model: '23+12=' completes to '35'.
- Character tokenizer, padding + EOS token, transformer blocks, autoregressive generation one digit at a time.
- Accuracy = % of correct digits in the generated answers.

> Exam traps: the attention matrix is T × T and each row sums to 1; LSTMs fix vanishing gradients but still can't parallelise over time; transformers need positional encoding because attention ignores order.
`},
{ w: 8, md: `
## Discriminative vs generative
- **Discriminative** models learn **p(y|x)** (or a direct map x → y), i.e. the decision boundary. They classify but can't generate. Examples: logistic regression, CNN classifiers.
- **Generative** models learn **p(x)** or the joint **p(x, y)**, i.e. how the data is produced. They can **sample new data**, score likelihoods, and classify with Bayes' rule. Examples: GMM, Naive Bayes, VAE, GAN, diffusion.
- Product rule: **p(x, y) = p(y|x)p(x) = p(x|y)p(y)**, so p(y|x) = p(x|y)p(y)/p(x), with p(x) = Σ_y p(x|y)p(y).
- High p(y|x) but low p(x) means the input is unusual, so the decision is uncertain even if the classifier looks confident (links to class shift in week 5).

## Manifold hypothesis
- A 3×32×32 image with 256 levels has 256³⁰⁷² possibilities, almost all of them noise. A 256×256 RGB image has 196,608 dimensions.
- Real data lies **on or near a low-dimensional manifold** governed by a few factors (pose, lighting, shape). That justifies dimensionality reduction and latent-variable models.
- Linear manifold → **PCA**; non-linear → **autoencoder**; probabilistic latent → **VAE**.

## PCA: how to derive it
1. Centre the data: X ← X − mean. Covariance C = (1/N)XᵀX.
2. Find the unit vector u that maximises the variance of the projection: maximise **uᵀCu** subject to **uᵀu = 1**.
3. Lagrangian L = uᵀCu − λ(uᵀu − 1). ∂L/∂u = 2Cu − 2λu = 0 gives **Cu = λu**, so the directions are **eigenvectors** and the variance along each is uᵀCu = **λ** (the eigenvalue).
4. Sort by λ (descending). The first component has the largest eigenvalue; components are orthogonal; Σλ = total variance.
5. Keep the top k as U_k. Encode z = U_kᵀ(x − x̄); reconstruct x̂ = x̄ + U_k z (UUᵀx for centred data).
- Explained variance = Σ(top-k λ) / Σ(all λ). PCA also **minimises the squared reconstruction error** over all linear k-dimensional projections, which makes it a linear autoencoder and a denoiser (drop the small-variance directions).
- SVD of X is one way to compute it (optional). With N samples there are at most ~N meaningful components (50,000 images in 10⁶ dimensions gives ≤ 50,000).

## Autoencoders
- **Encoder** x → z (a low-dimensional **bottleneck**), **decoder** z → x̂, loss **‖x − x̂‖²**. Self-supervised: the input is its own target.
- Manifold view: the encoder learns a **coordinate chart** of the data manifold (z = coordinates on it), and the decoder maps the coordinates back into data space. The bottleneck forces it to keep the factors of variation and drop noise.
- A **linear** AE (no activations) with MSE learns the **same subspace as PCA** (though the basis needn't be orthogonal or ordered). With non-linear layers an AE learns **curved** manifolds PCA can't.
### Variants
- **Denoising**: input x + noise, target the clean x. It can't just copy, so it learns to **project noisy points back onto the manifold**, i.e. the direction towards high density (the score, the foundation of diffusion).
- **Masked**: hide patches (MAE hides ~75%) and reconstruct them; the high ratio forces semantic understanding. BERT's masked tokens are the text analogue.
- **Sparse**: a large latent (e.g. 784, even bigger than the input) with an **L1 or top-k** sparsity constraint so only a few units fire per input. Gives interpretable features, used for **LLM interpretability**.
- Encoder-decoders are everywhere: transformers, UNet, masked AEs, DALL-E, Stable Diffusion.
### Why a plain AE is a bad generator
- The latent distribution is uncontrolled: codes are scattered with gaps ("holes"), so a random z decodes to rubbish.

## VAE: how to derive the loss
- The encoder outputs a **distribution** q_φ(z|x) = N(μ(x), σ²(x)), not a point; the decoder models p_θ(x|z); the prior is p(z) = N(0, I).
1. We want max log p(x), but p(x) = ∫p(x|z)p(z)dz is intractable.
2. Introduce the encoder q_φ(z|x): log p(x) = ELBO + KL(q_φ(z|x) ‖ p(z|x)) ≥ ELBO, since KL ≥ 0.
3. **ELBO = E_q[log p_θ(x|z)] − KL(q_φ(z|x) ‖ N(0, I))**: a reconstruction term minus a regulariser.
4. With a Gaussian decoder: **Loss = MSE(x, x̂) + β·KL** (BCE instead of MSE for a Bernoulli decoder, e.g. binary MNIST).
- **KL closed form** (per latent dimension): KL = ½(μ² + σ² − ln σ² − 1). When μ = 0, σ = 1, KL = 0. Summed over dimensions: −½Σ(1 + ln σ² − μ² − σ²).
- **Reparameterisation**: z = μ + σ·ε, ε ~ N(0, 1). Sampling becomes differentiable, so gradients reach μ and σ. The encoder outputs μ and **ln σ²** (2 × latent dimension); σ = exp(0.5·logvar).
- Manifold view: each input maps to a small Gaussian cloud, and the KL term packs the clouds around the origin with no gaps, so nearby z decode to similar realistic data and interpolation is smooth.
- **Generation**: z ~ N(0, I), x = Dec(z). The encoder is discarded.
- Samples tend to be **blurry** (the reconstruction averages over the uncertainty).

## Balancing β and failure modes
- Reconstruction wants distinct, precise codes; KL wants overlapping, smooth codes. β sets the trade-off (**β-VAE**: larger β → more disentangled but blurrier).
- **β too high**: perfect prior fit but poor reconstruction, up to **posterior collapse** (μ = 0, σ = 1 for every x, so z carries no information).
- **β too low**: behaves like a plain AE: good reconstruction, poor prior fit, bad random samples.
- Collapse extremes: σ → 0 ("autoencoder mode") or a decoder that **ignores z**.
### Conditional VAE
- 2 classes give 2 manifolds and a ridge (gap) in z. Solution: the **conditional VAE**: give the condition c (one-hot label, attribute, text) to the **encoder and decoder**, q(z|x, c) and p(x|z, c).
- z no longer has to encode the class and captures the remaining **style** (thickness, slant).
- To generate a "7": fix c = 7 and sample z. To **classify**: try each c and take the one with the lowest reconstruction error.
- A conditional AE is the same without the sampling and KL. The condition can also enter through embeddings, FiLM layers or cross-attention.

## KL divergence (recap)
- KL(p‖q) = Σ p log(p/q) ≥ 0, = 0 only if p = q, **not symmetric**. It equals cross-entropy minus entropy, H(p, q) − H(p).
- Uses: the VAE regulariser, t-SNE, knowledge distillation, RLHF penalties. Minimising KL(p_data‖p_model) = maximum likelihood.

> Exam traps: a linear AE spans the same subspace as PCA; the VAE encoder outputs μ and log σ² (two vectors); KL = 0 exactly when μ = 0 and σ = 1; posterior collapse comes from β too high.
`},
{ w: 9, md: `
## VAE as a generator (recap)
1. Sample z ~ N(0, I) from the prior.
2. Decode: x = Dec(z). The encoder is not needed.
- Other uses: interpolate between two encoded images, latent arithmetic (add an attribute vector), choose a class with a conditional VAE. VAEs are also the compressor in **latent diffusion** (Stable Diffusion).

## High-dimensional geometry
- For z ~ N(0, I_D): **‖z‖ ≈ √D** (D = 10 gives 3.16; D = 400 gives 20). Samples live in a thin shell (the "soap bubble"); almost none are near the mean, and a norm < 1 is almost impossible when D > 100.
- Two random samples are **nearly orthogonal** (cos ≈ 0 ± 1/√D). They're **not** aligned with the axes.
- **Distance concentration**: pairwise distances between random points become nearly equal, so nearest and farthest neighbours look alike (kNN and clustering degrade).
- Most of a hypercube's volume sits in its corners; a ball's volume concentrates near its surface.
- **Interpolation**: the midpoint of two samples has norm ≈ 0.71√D (√(1/2) of the shell radius), inside the shell where the model never trained (atypical). Use **slerp** (spherical interpolation) instead.
### How to compute the typical norm
1. E‖z‖² = Σᵢ E[zᵢ²] = D·1 = D.
2. So ‖z‖ ≈ √D, with a spread that stays ≈ 0.7 while √D grows (a thin shell).
3. Example: D = 3072 (CIFAR 3×32×32, normalised) gives ≈ 55.
### Curse of dimensionality
- 3 bins per dimension gives **3^D** cells (10 per axis gives 10^D), so the samples needed to cover the space grow exponentially and data becomes **sparse**.
- Grid search and histograms become infeasible, distances lose meaning, overfitting risk grows.
- Real data has structure (the manifold hypothesis), which is why learning still works.

## Normalisation
- **Data normalisation**: centre and scale to std 1 per feature / channel using training statistics, or scale to [0, 1] / [−1, 1] (GAN and diffusion images use [−1, 1]).
- **BatchNorm**: per channel **across the batch** (and spatial positions): y = γ(x − μ_B)/√(σ_B² + ε) + β. Running statistics at inference; depends on batch size (poor with small batches); different in train and eval mode. Standard in CNNs.
- **LayerNorm**: per sample **across its features** (per token), independent of the batch, same in training and inference, fine with batch size 1 and variable lengths. Standard in **transformers** and RNNs.
- Both have learnable **γ, β**. Variants: instance norm (style transfer), group norm (small-batch vision, diffusion UNets), RMSNorm (LLMs).

## GAN
- **Generator** G(z): noise z ~ N(0, I) → fake x̃; tries to fool D. **Discriminator** D(x): P(real); tries to tell real from fake.
- Minimax objective: min_G max_D E[log D(x)] + E[log(1 − D(G(z)))].
### Training loop (alternate)
1. **Train D**: real batch labelled 1, fake batch G(z) labelled 0, **BCE loss**; update D only (detach G's output).
2. **Train G**: generate fakes, pass them through D, BCE **with label 1** (the non-saturating loss); update G only.
3. Repeat. After training only **G** is needed.
- DCGAN recipe: G uses ConvTranspose2d + BatchNorm + ReLU with a Tanh output; D uses Conv2d + LeakyReLU with a sigmoid output; Adam with lr 2e-4, β₁ = 0.5.
- Pros: **sharp, realistic** samples. Cons: **unstable** training (G and D must stay balanced), **mode collapse** (little variety), no encoder or likelihood, hard to evaluate. Fixes: WGAN loss, spectral normalisation, gradient penalty.
### Variants
- **cGAN**: G(z, c) and D(x, c) both get the condition (one-hot label, embedding, text, or an image in Pix2Pix); the condition modulates the features. D judges "real **and** matches c".
- **VAE-GAN**: the VAE decoder doubles as the GAN generator and a discriminator judges its outputs. Loss = KL + reconstruction (often in D's feature space) + adversarial. Gives the VAE's structured latent plus the GAN's **sharper outputs**.

## Score-based and diffusion
- **Score** s(x) = ∇ₓ log p(x): a vector field pointing towards **higher data density** (towards the manifold). It doesn't need the intractable normalising constant.
- **Langevin dynamics**: x ← x + (η/2)s(x) + √η·ε, a gradient walk uphill with noise so it samples rather than just finding the peak.
- Problem: in sparse regions (far from the data) there is no training data, so the score estimate is bad. Fix: **add noise at multiple scales**, starting with lots of noise (which fills the space) and annealing down.
- A denoiser learns the score: ∇ log p(xₜ) ≈ −ε/√(1 − ᾱₜ), so **predicting the noise is equivalent to predicting the score**.
### DDPM: how training and sampling work
1. Forward process (no training): add Gaussian noise over T steps (e.g. 1000). Closed form: **xₜ = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε**, with αₜ = 1 − βₜ and ᾱₜ = Π αₛ.
2. **Noise schedule** β₁ … β_T: how much noise each step adds. **Linear** (1e-4 to 0.02) or **cosine** (destroys information more gradually, better for small images).
3. Training: pick a random t, sample ε, form xₜ, and train ε_θ(xₜ, t) (a UNet or transformer) with loss = **‖ε − ε_θ(xₜ, t)‖²**.
4. Sampling: start from x_T ~ N(0, I) and iteratively remove the predicted noise (adding a little fresh noise) down to t = 0. Each step is an easy denoising problem.
### Guidance
- **Classifier guidance**: ∇log p(x|c) = ∇log p(x) + s·∇log p(c|x). Needs a separate classifier trained on **noisy** images; its gradients can be adversarial.
- **Classifier-free guidance**: train one model both conditionally and unconditionally (drop the condition 10–20% of the time). At sampling: **ε̂ = ε(x, ∅) + w[ε(x, c) − ε(x, ∅)]**. w > 1 boosts prompt adherence and quality but reduces diversity (over-saturation at high w). Used in Stable Diffusion, DALL·E, Imagen.

## Conditional flow matching
1. Pair noise x₀ ~ N(0, I) with data x₁ and sample t ~ U(0, 1).
2. Straight path xₜ = (1 − t)x₀ + t·x₁; the true velocity along it is constant: **x₁ − x₀**.
3. Loss = **MSE(u_θ(xₜ, t), x₁ − x₀)**.
4. Generate: start from noise and solve the **ODE** dx/dt = u_θ(x, t) with a solver (Euler: x ← x + u_θ(x, t)·Δt).
- Deterministic: no random walk, but still several steps because the learned field is non-linear (paths cross and curve). **Rectified flow** straightens the paths so fewer steps are needed. Stable Diffusion 3 uses flow-matching-style training. Conditioning and CFG work the same way.

## Summary of generative models
- **AE / VAE / GAN**: a **low-dimensional** latent, **one-shot** decoding, but many random z are bad (holes or collapse).
- **Diffusion / flow**: the latent has the **same dimension as x**, **multi-step** generation, and any random z is good.

## Measuring quality: FID
- A good generator needs **fidelity** (realistic), **diversity** (covers all modes) and, if conditional, consistency with the condition.
1. Real and generated images go through **Inception-v3** (299×299 input) to the pool3 features (2048-D).
2. Fit Gaussians (μ, Σ) to each set.
3. **FID = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_rΣ_g)^½)**. **Lower is better**. Captures quality and diversity; needs many samples; biased towards Inception's features.
- Others: **IS** (Inception Score: confident and varied classes; higher is better; doesn't compare with real data), **KID**, **SWD**, precision / recall for generative models (fidelity vs coverage), CLIP score (text–image match), likelihood / bits per dimension, PSNR / SSIM / LPIPS for reconstructions, train-on-synthetic test-on-real, and human evaluation.

> Exam traps: ‖z‖ ≈ √D, not 1; the linear midpoint has norm ≈ 0.71√D; the DDPM network predicts the noise ε; flow matching regresses the velocity x₁ − x₀; lower FID is better.
`},
{ w: 10, md: `
## Overfitting as a feature
- Normally overfitting is bad: we want the ANN to **generalise** patterns from the data to new samples.
- In **implicit neural representations** we deliberately **overfit one signal** (a shape, an image, a 3D scene). The network's weights become the representation of that signal.
- The usual recipe changes: there is no analytical solution (the activations make it non-linear) and the parameter space is too big to inspect, so we optimise a cost function with gradient descent.
- The input is a **coordinate** (x, y), (x, y, z) or (x, y, z, θ, φ); the output is the signal value at that coordinate (inside / outside, intensity, colour and density).

## Universal approximation, constructively
- One hidden layer with enough units can approximate any function (a 2-layer ANN is a universal approximator). In theory you can get arbitrarily close; in practice **more units are harder to optimise**.
### Shaping a sigmoid
- y = sigmoid(w·x + b) for one input x.
- **w = steepness**: a large w (e.g. 50) makes the sigmoid almost a step.
- **b = shift**: the step happens at x = −b/w.
### How to build a box from two sigmoids
1. Unit 1: a steep sigmoid that steps **up** at the left edge (w = 50, b chosen so −b/w = left edge).
2. Unit 2: a steep sigmoid that steps up at the right edge.
3. Output = unit 1 − unit 2: 1 between the edges and 0 elsewhere, a "box".
- Many boxes of different heights side by side approximate any 1D function (like a histogram). 10 sigmoid units give a rough fit; ReLU units work similarly (piecewise-linear pieces), and 100 ReLUs fit closely.
### Wider vs deeper
- To get closer, stacking layers works better than one huge layer: two or three hidden layers of 100 units fit better than a single wide one.

## 2D and 3D signals
### Binary image
- An MLP (x, y) → 0/1 learns a star shape. The edges come out smooth (**anti-aliased**) because sigmoids give a smooth decision boundary. The network doesn't "know" it's an image; it only learned which (x, y) pairs give 1.
- Three ways to store the star: an **equation** (~200 characters ≈ 200 bytes), **pixels** (100×100 = 10,000 bytes, or a 1,900-byte jpg), or **MLP weights** (~105 floats ≈ 450 bytes).
### Signed distance function (SDF)
- Each location stores the **closest distance to the shape**: > 0 outside, < 0 inside, = 0 on the boundary. An MLP with two hidden layers can learn it.
- Uses: **threshold at 0** gives the binary mask; any location gives the distance to the object; the **gradient** (computed with PyTorch **autograd**) points towards / away from the object.
- **Infinite resolution**: once trained, query the model on any grid (e.g. 1000×1000) with the **same number of parameters**, even though the image is 100× larger.
### Image intensity
- The intensity I(x, y) is also a function of x and y, so an MLP can learn it.
- 3 layers of 32 sigmoids is blurry; 5 layers of 256 ReLUs still **misses detail**, because deep networks struggle to backpropagate the gradient. Adding **batch norm** between layers normalises the features and the gradient; Fourier features (below) recover high frequencies.
### Geometric transforms
- Because the model is a function of coordinates, you can **transform the coordinates** before querying it: rotation, translation, scaling, perspective. X' = (T + P + S)·R·X, then query I(X'). Example: angle = π/6, perspective = −0.1, scale = 1.5.
- The same idea extends to 3D: an MLP (x, y, z) → intensity works well for **medical scans** (MRI), where the signal is known everywhere in space.
- Natural scenes are different: we only have **2D projections** (photos), and the scene is **sparse** (mostly air). That's the problem NeRF solves.

## Graphics refresher
- **Rendering**: going from a 3D description of a scene to a 2D image.
- Coordinate systems: the extrinsic matrix **[R|T]** (rotation, translation) takes world coordinates to camera coordinates; the intrinsic matrix **K** takes camera coordinates to the screen.
- **Ray tracing**: shoot a ray from the camera centre through every pixel: **X = P + t·d**, t > 0 (only in front of the camera). If it hits an object, compute that colour for the pixel.
- **Rendering equation**: light leaving point p towards direction v = light **emitted** by p + **∫ BRDF · incoming light · (n·s)** over incoming directions s. The BRDF (bidirectional reflectance distribution function) is the fraction of light from s reflected towards v.
- **Lambert's law** gives the n·s factor: a surface tilted away from the light spreads it over a larger area, so the intensity drops.
- Materials: **diffuse** (reflects equally in all directions), **specular** (prefers one direction), **mirror** (exactly one direction).
- **Opacity**: a volume (cloud) between the point and the camera absorbs some light and emits its own.
- **Alpha compositing** of K colours front to back: **C = Σₖ cₖαₖ Π_(j<k)(1 − αⱼ)**. α = 1 is opaque (full colour), α = 0 is transparent.
- Old-school graphics model scenes with primitives (triangles, voxels); ML models them with networks (SRN, NeRF).

## NeRF (neural radiance field)
- MLP **F(x, y, z, θ, φ) → (RGB, σ)**: a 5D input (location + viewing direction) gives a colour and a **volume density**. Colour depends on direction (reflections); density only on position.
- Learn 3D from 2D photos: render each training pixel by **volume rendering** along its ray and use **MSE on the pixel colours** as the loss. Rendering is differentiable, so gradients reach the MLP.
### Volume rendering: how to compute it
1. Ray r(t) = o + t·d (o = camera position, d = direction, t = distance along the ray). The probability of hitting a particle in [t, t + dt] is **σ(t)dt** (denser medium → more likely to hit).
2. Transmittance **T(t) = exp(−∫₀ᵗ σ(s) ds)** is the probability of reaching t without being blocked.
3. Continuous colour: C = ∫ T(t)σ(t)c(t) dt.
4. Discrete: split the ray into segments of length δᵢ with constant density. **αᵢ = 1 − exp(−σᵢδᵢ)**, **Tᵢ = Π_(j<i)(1 − αⱼ)**, **C = Σ Tᵢαᵢcᵢ**. This is exactly alpha compositing.
5. Two passes: a **coarse** pass with evenly spaced samples, then a **fine** pass with more samples where the weights Tᵢαᵢ are high (near surfaces). The loss sums both.
### Spectral bias and Fourier features
- Coordinate MLPs are biased towards **low frequencies** (Neural Tangent Kernel theory): fed raw (x, y, z) they act like a low-pass filter and the result is **blobby**.
- Fix: map each coordinate p ∈ [−1, 1] through sinusoids first, **γ(p) = [sin(2ᵏπp), cos(2ᵏπp)] for k = 0 … L − 1**, a vector of length 2L.
### How to compute the encoding dimension
- dim = components × 2L.
- Position (x, y, z), L = 10: 3 × 2 × 10 = **60**.
- Direction as a unit vector, L = 4: 3 × 2 × 4 = **24**; as 2 angles (θ, φ): 2 × 8 = **16**.
### Camera poses
- NeRF needs the camera pose of every photo: either **given** with the data, **estimated** beforehand (most commonly with **COLMAP**, structure from motion), or **optimised jointly** with the rendering (**BARF**, bundle-adjusting NeRF).
### SRN (Scene Representation Network)
- An earlier approach with two networks: learn local 3D features plus a learned renderer. Works reasonably well but is a complex optimisation. NeRF is simpler: it differentiates the rendering physics directly instead of learning a renderer.

## Gaussian splatting
- NeRF is **dense**: it parameterises the radiance field at every point and queries it everywhere, but natural scenes are mostly air.
- Splatting is **sparse**: represent the scene as many **3D Gaussians** (position, shape / covariance, colour, opacity) only where the density is non-zero.
### How rendering works
1. **Cull** Gaussians outside the view frustum (less than 99% confidence of being in view), keeping a list of the Gaussians that contribute to each pixel.
2. **Project** each 3D Gaussian to the image: a projected 3D Gaussian is a **2D Gaussian**.
3. **Sort** by depth and **alpha-composite** their contributions directly, with no ray marching. Fast enough for **real time**.
### How training works
- Initialise the Gaussians from a **structure-from-motion (SfM) point cloud**, then optimise them against the photos with iterations of **cloning and splitting** (add detail where needed) and pruning (remove useless ones).

> Exam traps: NeRF outputs colour and density (not just colour); position encoding is 60-D and direction 24-D (16 with 2 angles); splatting is sparse while NeRF is dense; SDF is negative inside.
`},
{ w: 11, md: `
## Exam logistics (from the slides)
- Final exam (mid November): **~50 MCQ + 2 questions with a few short answers**, answered on a bubble sheet.
- **2 h + 10 min perusal**. Bring **one page, double-sided** (one piece of paper) of notes. A calculator may be needed for a few simple sums (e.g. 16 × 2).
- Plan: ~1 min per MCQ (1 h), then **30 min for each** short-answer question.
- Covers material from the **lectures, tutorials and assessment**. The Canvas review page is a planning summary; the answers are in the slides. Week 12 goes through practice questions.
### Assessment 2 interview
- An individual, recorded **12-minute** oral exam during the week 12 and 13 tutorials, with a panel of at least two staff.
- Questions cover **all eight** project assignments: the code, the report, the decisions made, and the related tutorial concepts, including parts written by other group members.
- Arrive 5 minutes early and bring printed reports. Be concise; if you don't know, say so and move on.

## LLM and attention refresher
- Pipeline: words → tokens (e.g. "university" → "uni", "ver", "sity"; 9 tokens out of a ~50,000 vocabulary) → embeddings + positional encoding (9 × 16) → N transformer blocks → ~50,000 next-token probabilities (e.g. 0.8 for "world").
### Attention shapes
1. X is T × dim_emb. Q = XW_q, K = XW_k, V = XW_v with each W of size dim_emb × D, so Q, K, V are T × D (**often D = dim_emb**).
2. QKᵀ is a **T × T** attention matrix; apply the **causal mask** (future positions → −∞).
3. A = softmax(QKᵀ/√D), applied row-wise, so **every row sums to 1** (e.g. [0.1, 0.9], [0.2, 0.5, 0.3]).
4. Output AV is T × D and is added back to the token embeddings.
- One attention block, stacked N times: **GPT-3 has 96 layers, DeepSeek-R1 has 61** transformer layers.

## Vision Transformer (ViT)
- The state-of-the-art vision model uses the same architecture as an LLM. The image equivalent of a sequence of words is a **sequence of patches**.
1. Cut the image into fixed-size patches (e.g. 16×16).
2. Flatten each patch and **project it to an embedding vector** (a linear layer).
3. Add a **2D positional encoding**, since attention ignores order.
4. Run the transformer blocks; a classification token or pooled output feeds the head.
- Transformer blocks inside **UNets** are now common (medical imaging, the diffusion denoisers below).

## DINO (self-distillation, no labels)
- DINO uses a ViT (DINOv3 is the latest version from Meta).
1. Each image x is turned into **different views** x₁, x₂ by data augmentation (crops, colour jitter, …).
2. The **student** sees one view, the **teacher** another. The teacher is not trained by gradient: it is the **EMA** of the student's weights (a filtered, slowly moving copy).
3. **Stop-gradient** (.detach()) on the teacher output.
4. **Centering** the teacher's features avoids **mode collapse** (the teacher collapsing to a constant output); **softmax sharpening** (low temperature) makes its predictions peaked.
5. Loss = **cross-entropy** between the teacher and student distributions, so the student learns to match the teacher across views.
- The best classification results come from **kNN on the learned features**, with no classifier training at all.
- Compare: contrastive learning needs negatives; DINO needs none (centering + sharpening do the anti-collapse work).

## Stable diffusion / latent diffusion
### The generative family
- **AE**: reduce the dimension. **VAE**: reduce the dimension + make the latent match a prior. **GAN**: a discriminator-driven transformation.
- **Diffusion** (Langevin dynamics, e.g. DDPM, learning score(x)) and **flow matching** (an ODE) generate in **several steps**.
- AE / VAE / GAN: low-dimensional latent, one-shot generation (pick z, x = Decode(z)), but **most random z give bad samples**. Diffusion / flow: latent with the same dimension as x, multi-step generation, **any random z gives a good sample**.
### Why VAEs alone aren't enough
- VAE training: Enc gives μ, σ; z = μ + σ·ε; loss = MSE(x, x̂) + β·KL(q(z|x) ‖ N(0, 1)). Generation: z ~ N(0, 1) → Dec.
- Latent-size trade-off: a **small latent (e.g. 64)** gives blurry reconstructions and equally blurry random samples; a **large latent (e.g. 2048)** gives sharp reconstructions, but random samples are rubbish.
- **Most latent locations are invalid**: the encoder only fills small regions of the latent space, and decoding anywhere else gives rubbish, so the VAE can't be used directly as a generator.
- Fix: use **diffusion or flow matching to map low-dimensional noise to valid VAE latents**, taking many small denoising steps (with a little random motion each step).
### Latent diffusion pipeline
1. A **VAE encoder** compresses images to a low-dimensional latent (cheaper than working on pixels).
2. **Diffusion runs between Gaussian noise and the latent**, not on the image directly. The denoiser is a **UNet that includes transformers**; reverse diffusion is solved as an SDE or ODE.
3. **Conditions** from different sources (text, images, maps) are encoded and **concatenated with, or attended by (cross-attention), the UNet features**. Text is the usual condition.
4. The **VAE decoder** turns the final latent into the image.
- Most modern generative AI generates the latent of a VAE this way.

## Encoder-only and predictive models
- VAE baseline problems: quite unstable, and the **μ and z manifolds are not identical**, so inference can fall outside the learned z manifold.
- Does intelligence need to **generate** data? Humans don't produce perfect images, and many narrow-AI tasks (classification) only need **features**: x → Enc → z → classifier.
- **Yann LeCun** (and others) argue intelligence **doesn't need a decoder**; it's based on **prediction** ("if one can predict the world, one can survive"), done in representation space.
### Contrastive learning (recap)
- Train an encoder so the **same object gives the same features** (positive pair) and **different objects give different features** (negative pair).
- Hard in practice: good positive and negative pairs are scarce, and training is long.
### JEPA (joint-embedding predictive architecture)
- Encode a context x into s_x and a target y into s_y, then **predict s_y from s_x directly in representation space**.
- **Strictly non-contrastive**: no negative samples. The prediction can be conditioned, and the target may lie in the **future** (temporal models, **world models**). x and y can be e.g. two parts of an image or two moments of a video.

> Exam traps: the DINO teacher is an EMA of the student (no gradient); centering prevents collapse; latent diffusion runs in the VAE latent space, not on pixels; JEPA predicts embeddings, not pixels, and uses no negatives.
`},
{ w: 12, md: `
## How to use this page
- Week 12 is the revision lecture. This chapter is the whole unit on one page plus a formula sheet you can copy to your one double-sided page of notes.
- Exam: ~50 MCQ (~1 min each) + 2 short-answer questions (~30 min each), 2 h + 10 min perusal. Each topic below points back to the week with the full notes.

## The rise of machine learning (week 1)
- **ML vs GOFAI**: the machine learns the **thresholds** that split data into classes (classification) or the **parameters** of a generic model (regression), versus a human expert writing **rules and logic** or an **analytical physics model** with a few fitted parameters.
- **Symbolism** (GOFAI) vs **connectionism** (modern AI). Connectionism was limited by slow computers and lack of data, then unlocked by PCs, the Internet (= data) and GPUs.
- ML is "creating maps": inputs become locations (embeddings); classification reads the region, generation decodes a location.
### Failure modes
1. **Unknown data**: a new patient doesn't match anything learned (failing to generalise, not enough training data).
2. **Data drift**: after deployment the data changes (new camera, new circumstances), so locations move.
3. **Bias**: locations are wrong for a subgroup (black patients placed near healthy white volunteers and not detected).
### Computer vs brain
- Traditional computer: exact, vast memory, fast, reliable, only solves problems coded by humans (**exact computing**).
- System 1: automated, unconscious, solves known problems, requires learning, relatively fast, **multitask** (**skills**: with enough data and compute a machine can learn any skill).
- System 2: conscious thinking, solves **unknown** problems by reasoning, relatively slow, **one task** (**reasoning**: we don't know how a machine can self-learn to reason about never-seen problems). Open questions: intelligence, understanding, AGI, ASI.

## Regression and experimental design (week 2)
- Pick the parameters that minimise the error between the prediction ŷ and the ground truth y (residual e = y − ŷ). 4 main assumptions: linearity, independence, homoscedasticity, normality of residuals.
- **Linear least squares**: set the derivative of the cost to 0, get the normal equations, so **W = (AᵀA)⁻¹AᵀB** (pseudo-inverse). Polynomial regression (3rd, 4th, N-th order) uses the same formula with columns 1, x, x², …
- Otherwise optimise with gradient descent; momentum smooths the path.
- ML recipe: get and inspect data → simplest model → analytical solution? → inspect the parameter space (grid search)? → optimise → analyse on the test set.
- **Experimental design**: **training** fits the model, **validation** selects hyperparameters (e.g. polynomial order; 90/10 split or k-fold), **testing** is secret data used once to benchmark, ideally from **another source**.

## Classic methods: how each defines classes
- **kNN** (fully supervised): normalise, store the training data, compute distances, take the K closest, majority vote (K = 5: 4/5 approved → approved). Classes defined by **neighbours**.
- **k-means** (unsupervised): each cluster is **one centre point**; the closest centre wins, so space is divided by straight lines.
- **GMM**: clusters are **Gaussians** (fuzzy C-means uses circles), fitted by EM.
- **Decision tree**: **conditions on each axis** (X > 1, Y > 0), giving boxy regions. **Random forest**: an **ensemble of trees**.
- **Artificial neuron**: one line **z = ax + by + c** (class 1 if z > 0, class 2 if z < 0) that minimises classification errors.
- **SVM**: a line with a **buffer zone** (margin), chosen to minimise errors inside the buffer.

## Performance (week 5)
- **FN** = miss = **Type II** ("not very good at a task"). **FP** = false alarm = **Type I** ("overdo a task"). Costs differ, so choose the target carefully; accuracy mixes both.
- **ROC**: sensitivity 98% = only 2% of the sick are missed; specificity 98% = 2% of the healthy are called sick. The point closest to the corner is often reported. **Screening** minimises FN; **diagnosis** minimises FP.
- **Calibration**: compare confidence with accuracy; under-confident (above the diagonal) vs over-confident (below).

## ANN and CNN key concepts (weeks 3–5)
- Geometric view: a **layer maps one feature space to another** where a new linear separation can be learned by the next layer; the output y is a **1D problem**, so everything from regression and classification applies.
- Training: backpropagation, gradient descent and optimisers (momentum, Adam), activation functions, overfitting and dropout.
- CNNs: convolution (e.g. a 5×5×3 filter on a 32×32×3 RGB image), pooling and stride, batch norm and layer norm, classic networks (convnet, VGG, ResNet, UNet), transfer learning, distillation, self-supervision and semi-supervision.
- Safety: adversarial attack, backdoor attack, saliency map, responsible AI.

## Sequences and transformers (weeks 7 and 11)
- Text processing timeline: **N-gram → Word2Vec → RNN → LSTM → attention**.
- LLM: words → tokens (~50,000 vocabulary) → embeddings + positional encoding → N attention blocks (GPT-3: 96, DeepSeek-R1: 61) → next-token probabilities.
- Attention: Q, K, V = XW_q, XW_k, XW_v; A = softmax(QKᵀ) with a causal mask, T × T, rows sum to 1; output AV (often D = dim_emb).
- **ViT**: the same model with **image patches** as tokens, projected to embeddings plus a 2D positional encoding.

## Generative AI: learning p(x) (weeks 6, 8, 9, 11)
- **p(x, y) = p(y|x)p(x)**: a high p(y|x) with a low p(x) means an uncertain decision.
- Progression: **GMM** (maximum likelihood via EM) → **VAE** (latent + q(z|x), ELBO) → **diffusion** (score, SDE / Langevin) → **flow matching** (velocity, ODE).
- AE / VAE / GAN: low-dimensional latent, one-shot, many random z are bad. Diffusion / flow: same dimension as x, multi-step, any z is good.
- **Latent diffusion**: diffusion or flow maps low-dimensional noise to **valid VAE latents**, conditioned on text / images, then the VAE decoder makes the image.
- Self-supervised representation learning: contrastive learning (positives and negatives), DINO (EMA teacher, centering), JEPA (predict embeddings, no negatives, no decoder).

## Implicit neural models (week 10)
- An MLP can approximate any function (sigmoid boxes), model a 2D image or SDF, or a 3D scan; NeRF models **density and colour** from 2D projections via volume rendering; Gaussian splatting uses a **sum of 3D Gaussians** (sparse, real time).

## Formula sheet (copy to your one page)
### Regression and optimisation
- OLS **W = (AᵀA)⁻¹AᵀB**. Pearson r = cov(x, y)/(σₓσᵧ). Grid search cost kᵖ.
- GD **w ← w − η∇L**. Momentum v ← βv + ∇L, w ← w − ηv. Adam β₁ ≈ 0.9, β₂ ≈ 0.999. EMA **y ← αy + (1 − α)x** (~1/(1 − α) values). Iterations per epoch = N / batch size.
### Networks
- Linear params **n_in·n_out + n_out**. Conv output **(N − F + 2P)/S + 1**. Conv params **(F²·C_in + 1)·C_out**. Pooling has **0** params. Same padding at stride 1: P = (F − 1)/2.
- Softmax **exp(sᵢ)/Σexp(sⱼ)**. CE **−log p_y**. Hinge **Σ_(j≠y) max(0, sⱼ − s_y + Δ)**. Sigmoid' ≤ 0.25. ResNet y = F(x) + x.
### Performance
- Accuracy (TP + TN)/all. Precision **TP/(TP + FP)**. Recall = Sensitivity **TP/(TP + FN)**. Specificity **TN/(TN + FP)**. FPR = 1 − specificity. F1 = 2PR/(P + R).
- PPV at low prevalence: count TP and FP on a population (1% prevalence, 98%/98% → PPV ≈ 33%).
- FGSM **x + ε·sign(∇ₓL)**.
### Probability and unsupervised
- Gaussian MLE **μ̂ = mean, σ̂² = (1/n)Σ(x − μ̂)²** (biased). EM responsibility γᵢₖ = πₖN(xᵢ|k)/Σⱼ πⱼN(xᵢ|j).
- **KL = ∫p log(p/q)**, ≥ 0, not symmetric.
### Sequences
- Attention **softmax(QKᵀ/√D)V**, a **T × T** matrix; per-head dim = **D/h** (768/12 = 64). PE sin / cos(pos/10000^(2i/D)).
### Generative
- PCA **Cu = λu**. VAE loss **MSE + β·KL**, **KL = ½Σ(μ² + σ² − ln σ² − 1)**, **z = μ + σε**.
- High dimension: **‖z‖ ≈ √D**, random vectors ⟂, midpoint norm ≈ 0.71√D.
- DDPM **xₜ = √ᾱₜx₀ + √(1 − ᾱₜ)ε**, loss **‖ε − ε_θ‖²**. CFG ε̂ = ε(∅) + w[ε(c) − ε(∅)]. Flow target velocity **x₁ − x₀**.
- FID ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_rΣ_g)^½), lower is better.
### Implicit models
- NeRF PE dimension = components × 2L (**60** for position, **24** for direction, 16 with 2 angles). αᵢ = 1 − exp(−σᵢδᵢ), Tᵢ = Π_(j<i)(1 − αⱼ), **C = Σ Tᵢαᵢcᵢ**.

> Exam strategy: about 1 minute per MCQ. For "combination" questions mark each statement T/F first; for "NOT true" questions find the one false statement. For "compute" questions write the formula first, then plug in. For short answers give the definition, how it works (steps), one example and one limitation.
`}
];
