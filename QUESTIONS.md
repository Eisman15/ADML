# ADML Exam Practice: Question Bank

243 multiple choice questions generated from the ADML lectures (weeks 1–12), using the weekly review questions and the practice exam / Week 12 mock quiz as a guide. Every answer has an in-depth explanation.

Interactive version: https://eisman15.github.io/ADML/ · Study notes: [NOTES.md](NOTES.md)

## Contents

- [Week 12 Mock Quiz](#week-12-mock-quiz)
- [Week 1: Introducing AI and Machine Learning](#week-1)
- [Week 2: Classical Machine Learning](#week-2)
- [Week 3: ANN and Deep Learning](#week-3)
- [Week 4: CNNs and Computer Vision](#week-4)
- [Week 5: Ethics and Performance](#week-5)
- [Week 6: Unsupervised Learning](#week-6)
- [Week 7: Language Models](#week-7)
- [Week 8: Autoencoder and VAE](#week-8)
- [Week 9: GAN, Diffusion and Flow](#week-9)
- [Week 10: Implicit Models (NeRF, Splatting)](#week-10)
- [Week 11: Advanced Methods (ViT, DINO, Latent Diffusion)](#week-11)
- [Week 12: Revision (cross-topic)](#week-12)

## Week 12 Mock Quiz

Official practice exam / Week 12 mock quiz questions first, then mock-style questions in the same format.

**Q1. [Week 1: Bias]** · _Practice exam_

Where is data bias coming from?

- A. From the class label imbalance in the dataset
- B. From the way the data is sampled from a distribution
- C. From the way humans label data during supervised learning
- D. The 3 other sources are possible

<details><summary>Answer</summary>

**D.** Bias can come from imbalance, from sampling (e.g. only healthy white volunteers), and from human labelling. All three are possible.

Why each option:

- A ✗ Partially true but not the only source.
- B ✗ Partially true but not the only source.
- C ✗ Partially true but not the only source.
- D ✓ Correct: imbalance, sampling and human labelling can all introduce bias.

In depth: Bias means the model is systematically wrong for some group or situation. It can enter at every stage of the data pipeline. (1) Class imbalance: rare classes get little weight, so the model favours the majority. (2) Sampling: if data is collected only from one hospital, population or camera, the training distribution differs from reality (e.g. only healthy white volunteers). (3) Labelling: annotators bring their own subjectivity, errors and cultural assumptions. Because all three are real, the 'all of the above' option is right.

</details>

**Q2. [Week 1: System 1 / System 2]** · _Practice exam_

'Thinking, Fast and Slow' (2011) describes how the brain works using two cognitive systems. Which statement is true?

- A. Brain System 1 is slower than System 2
- B. Brain System 1 is more logical than System 2
- C. Brain System 1 is more emotional than System 2
- D. Brain System 1 is more multitask than System 2

<details><summary>Answer</summary>

**D.** System 1 is automated, unconscious, relatively fast and multitask, and it solves known problems. System 2 is conscious reasoning: slow, one task at a time.

Why each option:

- A ✗ Wrong: System 1 is the fast one.
- B ✗ Wrong: System 2 is the logical, reasoning one.
- C ✗ Distractor. 'Emotional' is not the property the lecture uses; the key properties are fast, automatic and multitask.
- D ✓ Correct: System 1 runs automatically and can multitask.

In depth: Kahneman's System 1 is fast, automatic, unconscious and able to run several processes at once (driving while talking). It handles known problems through learned skills. System 2 is slow, effortful, conscious reasoning that handles one thing at a time (solving 17 × 24). The lecture compares current ML to System 1: with enough data it learns skills, but we don't know how to make machines do System 2 style reasoning on genuinely new problems.

</details>

**Q3. [Week 3: ML vs DL]** · _Practice exam_

What is the difference between machine learning and deep learning?

- A. Deep learning is a subset of machine learning that involves neural networks with more than one hidden layer
- B. Deep learning is a type of machine learning that uses only supervised learning during training
- C. Deep learning is a type of machine learning that has neural networks with very wide layers
- D. There is no difference between machine learning and deep learning

<details><summary>Answer</summary>

**A.** 'Deep' refers to depth: multiple hidden layers stacked on top of each other.

Why each option:

- A ✓ Correct: a subset of ML with multiple hidden layers.
- B ✗ DL is used with all learning paradigms, including self-supervised LLMs.
- C ✗ 'Deep' refers to depth, not width.
- D ✗ DL is a subset of ML, so there is a difference.

In depth: Deep learning is ML with deep neural networks, meaning several hidden layers stacked to learn hierarchical features automatically (edges → parts → objects). Classic ML (linear models, kNN, trees, SVMs) often relies on hand-engineered features. DL can be supervised, unsupervised or self-supervised, and depth, not width, is the defining property.

</details>

**Q4. [Week 3: ANN]** · _Week 12 mock quiz (More practice)_

Which of the following statements about Artificial Neural Networks (ANNs) is correct?

- A. ANNs are only capable of linear transformations and cannot model non-linear relationships without external preprocessing
- B. The activation function in a neuron is applied to the input features before computing the weighted sum
- C. During backpropagation, the weights are updated using the chain rule to compute gradients, and optimisers like SGD are used to minimise the loss
- D. A neural network with more hidden layers will always perform better than a shallower network, regardless of the task or dataset size

<details><summary>Answer</summary>

**C.** A is wrong: non-linear activations let ANNs model non-linear relationships. B is wrong: z = W·x + b comes first, then a = σ(z). D is wrong: deeper networks can overfit, suffer vanishing or exploding gradients, and cost more.

Why each option:

- A ✗ False: activations make ANNs non-linear.
- B ✗ False: the activation is applied after the weighted sum.
- C ✓ Correct: the chain rule plus an optimiser such as SGD.
- D ✗ False: depth must match the data and task.

In depth: Go through each option. A: non-linear activations (ReLU, sigmoid, tanh) let ANNs model non-linear relationships, so false. B: a neuron first computes z = W·x + b, then a = σ(z), so the activation comes after the weighted sum, so false. C: backpropagation uses the chain rule to get gradients, and optimisers (SGD, Adam, RMSProp) update the weights to minimise the loss, so true. D: deeper isn't always better; it can overfit, suffer vanishing or exploding gradients, and cost more, so false.

</details>

**Q5. [Week 4: Conv parameters]** · _Week 12 mock quiz Q7 (Practice exam)_

Consider a batch of 100 feature tensors of size [3,32,32]. We apply nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1, bias=True).  
A/ the output will be of size [100,5,32,32]  
B/ the output will have 3 channels  
C/ the number of parameters is 20  
D/ the number of parameters is 140  
What combination of statements is true?

- A. A and C
- B. A and D
- C. B and C
- D. B and D

<details><summary>Answer</summary>

**B.** Padding 1 with a 3×3 kernel keeps 32×32, and there are 5 output channels, so A is true. Parameters = (3·3·3 + 1)·5 = 28·5 = 140, so D is true.

Why each option:

- A ✗ A is true but C is false (20 is wrong).
- B ✓ Correct: A and D.
- C ✗ B is false (5 channels), and C is false.
- D ✗ B is false (the output has 5 channels).

In depth: Output size: (N − F + 2P)/S + 1 = (32 − 3 + 2)/1 + 1 = 32, so spatial size is kept. The batch stays 100 and there are 5 filters, so 5 output channels: [100, 5, 32, 32], so A is true and B is false. Parameters: each filter spans all 3 input channels, 3·3·3 = 27 weights + 1 bias = 28, and 5 filters give 140. So C (20) is false and D is true. The batch size never affects parameter count.

</details>

**Q6. [Week 4: Conv output size]** · _Week 12 mock quiz Q10 (Practice exam)_

You are designing a CNN for image classification. Input images are 64×64 with 3 colour channels. Layers: (1) Conv 16 filters 3×3, stride 1, padding 1; (2) ReLU; (3) MaxPool 2×2 stride 2; (4) Conv 32 filters 3×3, stride 1, padding 1; (5) ReLU; (6) MaxPool 2×2 stride 2. What is the shape of the output after the second max-pooling layer?

- A. [16, 16, 16]
- B. [32, 16, 16]
- C. [32, 32, 32]
- D. [32, 8, 8]

<details><summary>Answer</summary>

**B.** [3,64,64] → conv → [16,64,64] → pool → [16,32,32] → conv → [32,32,32] → pool → [32,16,16].

Why each option:

- A ✗ Wrong channel count (16 is from the first conv).
- B ✓ Correct: [32, 16, 16].
- C ✗ Size before the second pooling.
- D ✗ One pooling too many.

In depth: Track [C, H, W] step by step. Input [3, 64, 64]. Conv 16@3×3, P=1, S=1: (64 − 3 + 2)/1 + 1 = 64, so [16, 64, 64]. ReLU doesn't change the shape. MaxPool 2×2 S2 gives [16, 32, 32]. Conv 32@3×3, P=1: [32, 32, 32]. ReLU keeps it. MaxPool gives [32, 16, 16]. Rule of thumb: '3×3 with padding 1, stride 1' keeps size, each 2×2/2 pool halves it, and channels always equal the number of filters in the last conv.

</details>

**Q7. [Week 5: Metrics]** · _Practice exam_

In a medical diagnosis task with high class imbalance (rare disease), which metric would be the most appropriate to evaluate model performance?

- A. Accuracy
- B. Sensitivity and specificity
- C. Mean squared error
- D. Training loss

<details><summary>Answer</summary>

**B.** With 1% prevalence, predicting 'healthy' for everyone gives 99% accuracy. Sensitivity and specificity measure each type of error separately.

Why each option:

- A ✗ Misleading under imbalance (the 99% trap above).
- B ✓ Correct: per-class rates that are robust to imbalance.
- C ✗ MSE is a regression metric.
- D ✗ Training loss measures fit, not test performance.

In depth: Worked example: 1,000 patients, 10 sick (1% prevalence). A useless model that says 'healthy' to everyone gets TN = 990, FN = 10, so accuracy = 990/1000 = 99%, yet sensitivity = TP/(TP+FN) = 0/10 = 0%. Accuracy mixes the two error types and is dominated by the majority class. Sensitivity (how many sick are caught) and specificity (how many healthy are correctly cleared) each look at one class, so neither can be inflated by imbalance. Precision, recall and the ROC curve are also appropriate. MSE is for regression, and training loss says nothing about deployment.

</details>

**Q8. [Week 6: Semi-supervised]** · _Practice exam_

Which of the following statements about semi-supervised learning is correct?

- A. Semi-supervised learning leverages both labelled and unlabelled data to improve model performance
- B. Semi-supervised learning uses only labelled data to train models, ignoring unlabelled data entirely
- C. Semi-supervised learning assumes that unlabelled data carries no useful information and is primarily used to augment small labelled datasets with synthetic data
- D. Semi-supervised learning is only applicable to classification tasks and cannot be used for regression

<details><summary>Answer</summary>

**A.** It's suitable when labels are scarce or expensive but unlabelled data is plentiful.

Why each option:

- A ✓ Correct.
- B ✗ That's fully supervised learning.
- C ✗ Unlabelled data is useful, which is the whole point.
- D ✗ It also works for regression and other tasks.

In depth: Semi-supervised learning assumes unlabelled data does carry information about the structure of p(x): the cluster assumption (points in the same cluster share a label) and the smoothness or manifold assumption. Methods propagate the few labels through that structure. It's suitable when labelling is expensive but raw data is plentiful (medical scans, web images), and it works for classification and regression.

</details>

**Q9. [Week 7: LSTM]** · _Practice exam_

What problem do LSTM networks primarily solve compared to vanilla RNNs?

- A. Faster training speed
- B. Lower memory requirements
- C. Elimination of the need for activation functions
- D. Vanishing gradient problem for long sequences

<details><summary>Answer</summary>

**D.** The LSTM cell state, regulated by gates, keeps memory over longer sequences and is more robust to vanishing gradients.

Why each option:

- A ✗ LSTMs are slower than plain RNNs.
- B ✗ They use more memory (4 gate matrices).
- C ✗ They depend on sigmoid and tanh activations.
- D ✓ Correct: they solve vanishing gradients over long sequences.

In depth: In a vanilla RNN the gradient through time is a product of many Jacobians, so it vanishes (or explodes) over long sequences. The LSTM adds a cell state c_t updated additively: c_t = f_t ⊙ c_(t−1) + i_t ⊙ c̃_t. When the forget gate f_t ≈ 1, information and gradient flow through many steps almost unchanged (like a residual connection through time). LSTMs aren't faster or lighter; they're slower and use more memory, and they rely on sigmoid and tanh activations.

</details>

**Q10. [Week 7: Attention concepts]** · _Week 12 mock quiz Q12 (Practice exam)_

Consider 4 concepts: 1. Multi-Head Attention, 2. Value (V), 3. Self-attention, 4. Causal Masking. And 4 statements: A. Each token in a sequence attends to all other tokens in the same sequence. B. Ensures each token can only attend to previous tokens in autoregressive models. C. Determines how much each value contributes to the final output based on attention scores. D. Splits the input into multiple smaller attention heads to capture different relationships in parallel. Which combination is correct?

- A. 1-D, 2-C, 3-A, 4-B
- B. 1-A, 2-B, 3-C, 4-D
- C. 1-D, 2-B, 3-A, 4-C
- D. 1-B, 2-D, 3-C, 4-A

<details><summary>Answer</summary>

**A.** Multi-head attention splits into heads (D). Values are what gets weighted and mixed (C). Self-attention is all-to-all within the sequence (A). Causal masking blocks future tokens (B).

Why each option:

- A ✓ Correct: 1-D, 2-C, 3-A, 4-B.
- B ✗ Takes the statements in order, which is wrong.
- C ✗ Swaps the Value and Causal Masking statements.
- D ✗ Mismatches several.

In depth: Match one at a time. Multi-head attention runs several smaller attentions in parallel on split dimensions, so 1-D. Value V holds the content that the attention weights combine: the scores determine how much each value contributes, so 2-C. Self-attention means each token attends to all tokens of the same sequence, so 3-A. Causal masking sets future positions to −∞ before softmax so token t only sees ≤ t, needed for autoregressive generation, so 4-B. Answer: 1-D, 2-C, 3-A, 4-B.

</details>

**Q11. [Week 8: PCA]** · _Practice exam_

Considering the principal component analysis method, which statement is NOT true?

- A. Each component direction is perpendicular to every other one
- B. The sum of the variance of all the components should be equal to the total variance of the data
- C. Each component direction is an eigenvector of the covariance matrix
- D. One has to use singular value decomposition (SVD) to compute a PCA

<details><summary>Answer</summary>

**D.** SVD is one way to compute PCA, but you can also do an eigendecomposition of the covariance matrix. It is not required.

Why each option:

- A ✗ True: the components are orthogonal.
- B ✗ True: the variances sum to the total variance.
- C ✗ True: they're eigenvectors of the covariance.
- D ✓ NOT true, so this is the answer: SVD is optional.

In depth: PCA: centre the data, compute the covariance C = (1/N)XᵀX, find its eigenvectors uᵢ (directions) and eigenvalues λᵢ (variance along each). The eigenvectors of a symmetric matrix are orthogonal (A true), λᵢ is the variance captured and Σλᵢ = trace(C) = total variance (B true), and the directions are eigenvectors of C (C true). D is false: SVD of X is one convenient, numerically stable way (the right singular vectors are the uᵢ, and λᵢ = σᵢ²/N), but eigendecomposition of C or iterative methods work too.

</details>

**Q12. [Week 9: High dimension]** · _Practice exam_

Considering a D-dimensional multivariate normal distribution (D > 100), which statement is NOT true?

- A. The norm is expected to be on average sqrt(D)
- B. Two random samples would be orthogonal
- C. The vectors are most likely to be aligned with the orthonormal axes
- D. It is almost impossible to sample a vector with a norm lower than 1

<details><summary>Answer</summary>

**C.** In high dimension, samples lie in a thin shell of radius ≈ √D and random pairs are almost perpendicular. They are not aligned with the axes.

Why each option:

- A ✗ True: E‖z‖ ≈ √D.
- B ✗ True: random vectors are nearly orthogonal.
- C ✓ NOT true, so this is the answer: isotropic samples have no axis preference.
- D ✗ True: they concentrate around √D, far above 1.

In depth: For z ~ N(0, I_D): (1) ‖z‖² = Σzᵢ² is chi-squared with mean D, so ‖z‖ ≈ √D and its spread stays ~1/√2, so samples live in a thin shell (A true). (2) For two independent samples, cos θ = z₁·z₂/(‖z₁‖‖z₂‖) has mean 0 and std ≈ 1/√D, so they're almost perpendicular (B true). (3) A norm below 1 when the typical norm is √100 = 10 is astronomically unlikely (D true). (4) An isotropic Gaussian has no preferred direction, and being aligned with an axis would mean one large coordinate and all others ≈ 0, which almost never happens. C is NOT true.

</details>

**Q13. [Week 9: Diffusion]** · _Practice exam_

Which of the following statements about diffusion models is correct?

- A. Diffusion models generate data by iteratively adding noise to a random sample until it matches the target distribution
- B. The forward process gradually adds Gaussian noise to the data over a series of timesteps, while the reverse process learns to denoise the data
- C. Diffusion models are trained using a single-step process where the model directly predicts the final denoised sample from pure noise
- D. Diffusion models are only applicable to discrete data (e.g. text) and cannot be used for continuous data like images or audio

<details><summary>Answer</summary>

**B.** The forward process (an encoder with no training) adds noise. The reverse process is learned and denoises iteratively.

Why each option:

- A ✗ Backwards: generation removes noise iteratively.
- B ✓ Correct.
- C ✗ Generation is iterative, not single-step.
- D ✗ It's mainly used for continuous data like images and audio.

In depth: Forward process: x_t = √(ᾱ_t)·x₀ + √(1 − ᾱ_t)·ε with ε ~ N(0, I). Over T steps the image becomes pure noise; it's fixed and has no training (like an encoder without parameters). Reverse process: a network (UNet) learns to undo one small step at a time, from noise back to data. A is backwards (generation removes noise). C is wrong: sampling takes many steps. D is wrong: diffusion is the state of the art for images and audio.

</details>

**Q14. [Week 1: System 1 / System 2]** · _Mock-style_

Consider the statements:  
A/ System 1 is automated and unconscious  
B/ System 2 is relatively fast and multitask  
C/ System 2 is used to solve unknown problems through reasoning  
D/ System 1 cannot be learned  
Which combination is true?

- A. A and C
- B. B and C
- C. A and D
- D. B and D

<details><summary>Answer</summary>

**A.** System 1 is fast, multitask and requires learning. System 2 is slow, one task at a time, and reasons.

Why each option:

- A ✓ Correct: A and C are the true statements.
- B ✗ B is false (it describes System 1).
- C ✗ D is false (System 1 skills are learned).
- D ✗ Both B and D are false.

In depth: Check each statement. A: System 1 is automated and unconscious, so true. B: 'relatively fast and multitask' describes System 1, not System 2, so false. C: System 2 is conscious reasoning that handles unknown problems, so true. D: System 1 does require learning (skills become automatic through practice), so false. That leaves A and C. In combination questions, mark each statement T/F first, then match the option.

</details>

**Q15. [Week 2: Linear regression]** · _Mock-style_

Consider the statements:  
A/ A 1st-order polynomial regression is the same as linear regression  
B/ Polynomial regression cannot be solved with OLS  
C/ The polynomial order is a hyperparameter selected on validation data  
D/ The coefficients are selected on the test set  
Which combination is true?

- A. A and C
- B. A and B
- C. B and D
- D. C and D

<details><summary>Answer</summary>

**A.** Polynomial regression is linear in its parameters, so OLS works. The test set is never used for fitting or selection.

Why each option:

- A ✓ Correct: A and C are true.
- B ✗ B is false (OLS works for polynomials).
- C ✗ B and D are both false.
- D ✗ D is false (never fit on the test set).

In depth: A: a 1st-order polynomial is y = w₀ + w₁x, the same as linear regression, so true. B: polynomial regression is linear in its parameters, so OLS applies, so false. C: the order is a hyperparameter chosen on validation data, so true. D: coefficients are fitted on training data, and the test set is only for final evaluation, so false. Answer: A and C.

</details>

**Q16. [Week 3: Parameters]** · _Mock-style_

How many learnable parameters does an MLP with 2 inputs, one hidden layer of 10 units and 1 output have (with biases)?

- A. 41
- B. 30
- C. 20
- D. 31

<details><summary>Answer</summary>

**A.** Hidden layer: 2·10 + 10 = 30. Output layer: 10·1 + 1 = 11. Total 41.

Why each option:

- A ✓ Correct: 30 + 11 = 41.
- B ✗ Only the hidden layer (forgets the output layer).
- C ✗ Hidden weights only (forgets biases and the output layer).
- D ✗ Forgets the output layer's weights or biases.

In depth: A fully connected layer with n_in inputs and n_out outputs has n_in·n_out weights plus n_out biases. Hidden layer: 2·10 + 10 = 30. Output layer: 10·1 + 1 = 11. Total = 41. Exam tip: write each layer as (n_in + 1)·n_out, here (2+1)·10 + (10+1)·1 = 30 + 11.

</details>

**Q17. [Week 4: Conv parameters]** · _Mock-style_

A batch of 8 tensors of size [16,64,64] goes through nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=True).  
A/ the output size is [8,32,64,64]  
B/ the number of parameters is 12,832  
C/ the output size is [8,32,60,60]  
D/ the number of parameters is 12,800  
Which combination is true?

- A. A and B
- B. A and D
- C. C and B
- D. C and D

<details><summary>Answer</summary>

**A.** Padding 2 with a 5×5 kernel keeps 64×64: (64 + 4 − 5)/1 + 1 = 64. Parameters = (5·5·16 + 1)·32 = 401·32 = 12,832. D forgets the 32 biases.

Why each option:

- A ✓ Correct: A and B.
- B ✗ D forgets the biases.
- C ✗ C would be the no-padding size.
- D ✗ Both C and D are wrong.

In depth: Size: (64 − 5 + 2·2)/1 + 1 = 64, so the output is [8, 32, 64, 64] and A is true, C false (60 would be without padding). Parameters: (5·5·16 + 1)·32 = (400 + 1)·32 = 12,832, so B is true. D (12,800) forgets the 32 biases. Pattern: padding P = (F − 1)/2 keeps the size for odd F at stride 1.

</details>

**Q18. [Week 4: Conv output size]** · _Mock-style_

Input [3,128,128]. Apply: (1) Conv 8 filters 3×3, stride 2, padding 1; (2) ReLU; (3) MaxPool 2×2 stride 2. What is the output shape?

- A. [8, 32, 32]
- B. [8, 64, 64]
- C. [3, 32, 32]
- D. [8, 16, 16]

<details><summary>Answer</summary>

**A.** Conv: floor((128 + 2 − 3)/2) + 1 = 64, giving [8,64,64]. ReLU doesn't change the shape. Pool halves it to [8,32,32].

Why each option:

- A ✓ Correct: [8, 32, 32].
- B ✗ Forgets the pooling step.
- C ✗ Channels become 8, not 3.
- D ✗ One halving too many.

In depth: Conv: (128 − 3 + 2·1)/2 + 1 = 127/2 + 1 = 64.5, and PyTorch floors it to 64, giving [8, 64, 64] (a 3×3, stride-2, padding-1 conv halves the size). ReLU doesn't change the shape. MaxPool 2×2 S2 gives [8, 32, 32]. Channels = 8 filters. So two halvings take 128 → 64 → 32.

</details>

**Q19. [Week 4: Parameters]** · _Mock-style_

The feature map [32,16,16] is flattened and fed to nn.Linear(…, 10). How many parameters does this linear layer have?

- A. 81,930
- B. 81,920
- C. 8,202
- D. 5,130

<details><summary>Answer</summary>

**A.** 32·16·16 = 8,192 inputs. 8,192 × 10 weights + 10 biases = 81,930.

Why each option:

- A ✓ Correct: 81,920 + 10.
- B ✗ Forgets the 10 biases.
- C ✗ Wrong flatten size.
- D ✗ Uses 512 features instead of 8,192.

In depth: Flattening gives 32 × 16 × 16 = 8,192 features. nn.Linear(8192, 10) has a weight matrix of 10 × 8,192 = 81,920 entries plus 10 biases, so 81,930. Notice this one layer has more parameters than many conv layers combined, which is why modern CNNs use global average pooling before the classifier.

</details>

**Q20. [Week 4: Pooling]** · _Mock-style_

Consider the 4 statements about a MaxPool2d(2, stride=2) layer applied to [64,32,32]:  
A/ it has 0 learnable parameters  
B/ the output is [32,32,32]  
C/ the output is [64,16,16]  
D/ it has 64 learnable parameters  
Which combination is true?

- A. A and C
- B. A and B
- C. D and C
- D. B and D

<details><summary>Answer</summary>

**A.** Pooling has no parameters, keeps the number of channels, and halves the spatial size.

Why each option:

- A ✓ Correct: A and C.
- B ✗ B is false (pooling keeps channels).
- C ✗ D is false (no parameters).
- D ✗ Both B and D are false.

In depth: Max pooling has no learnable parameters: it just takes the maximum in each window, so A is true and D false. It operates on each channel independently, so channels stay 64 and spatial size halves: (32 − 2)/2 + 1 = 16, so the output is [64, 16, 16]. C is true and B false (it doesn't change the channel count).

</details>

**Q21. [Week 5: Metrics]** · _Mock-style_

A classifier on 100 healthy patients gives TN = 90 and FP = 10, and on 20 sick patients gives TP = 15 and FN = 5. What are sensitivity and specificity?

- A. Sensitivity = 0.75, Specificity = 0.90
- B. Sensitivity = 0.90, Specificity = 0.75
- C. Sensitivity = 0.60, Specificity = 0.90
- D. Sensitivity = 0.875, Specificity = 0.75

<details><summary>Answer</summary>

**A.** Sensitivity = TP/(TP+FN) = 15/20 = 0.75. Specificity = TN/(TN+FP) = 90/100 = 0.90.

Why each option:

- A ✓ Correct: 0.75 and 0.90.
- B ✗ Swapped.
- C ✗ 0.60 is precision, not sensitivity.
- D ✗ Wrong denominators.

In depth: Arrange the counts. Sick (positives): TP = 15, FN = 5, so 20 total. Healthy (negatives): TN = 90, FP = 10, so 100 total. Sensitivity = TP/(TP+FN) = 15/20 = 0.75. Specificity = TN/(TN+FP) = 90/100 = 0.90. Precision would be 15/(15+10) = 0.60 (one of the distractors). Always check which denominator belongs to which metric.

</details>

**Q22. [Week 6: Clustering]** · _Mock-style_

Which statement about k-means is NOT true?

- A. K is a hyperparameter chosen by the user
- B. It can converge to a local minimum depending on initialisation
- C. It requires labelled data to compute the centroids
- D. Each point is assigned to its closest centroid

<details><summary>Answer</summary>

**C.** k-means is unsupervised. kNN is the method that needs labels.

Why each option:

- A ✗ True: K is a hyperparameter.
- B ✗ True: there are local minima.
- C ✓ NOT true, so this is the answer: k-means is unsupervised.
- D ✗ True: that's the assignment step.

In depth: Check each statement. K is set by the user (true); results depend on the initial centroids (true); each point goes to its nearest centroid (true). 'Requires labelled data' is false: k-means only uses the feature vectors and finds groups by distance. The names are confusingly similar: kNN (supervised) needs labels to vote; k-means (unsupervised) doesn't.

</details>

**Q23. [Week 6: EM vs k-means]** · _Mock-style_

Consider the statements:  
A/ GMM-EM gives soft (probabilistic) assignments  
B/ k-means gives soft assignments  
C/ EM alternates an Expectation step and a Maximisation step  
D/ EM requires the class labels of all samples  
Which combination is true?

- A. A and C
- B. B and C
- C. A and D
- D. B and D

<details><summary>Answer</summary>

**A.** k-means uses hard assignments, and EM for GMM is unsupervised.

Why each option:

- A ✓ Correct: A and C.
- B ✗ B is false.
- C ✗ D is false.
- D ✗ B and D are both false.

In depth: A: the E-step computes probabilistic responsibilities, so assignments are soft. True. B: k-means assigns each point to exactly one cluster (hard), so false. C: EM alternates the E-step (responsibilities) and M-step (parameter updates), so true. D: EM fits GMMs without labels (unsupervised), so false. Answer: A and C.

</details>

**Q24. [Week 7: Sequence models]** · _Mock-style_

Consider the 4 statements:  
A/ Transformers process all tokens of a sequence in parallel  
B/ RNNs can easily be parallelised over the sequence  
C/ Positional encoding is needed because attention alone ignores token order  
D/ LSTMs eliminate the need for activation functions  
Which combination is true?

- A. A and C
- B. A and B
- C. B and D
- D. C and D

<details><summary>Answer</summary>

**A.** RNNs are sequential, so B is false. LSTMs use sigmoid and tanh gates, so D is false.

Why each option:

- A ✓ Correct.
- B ✗ B is false.
- C ✗ B and D are false.
- D ✗ D is false.

In depth: A: transformers feed all tokens simultaneously, true. B: RNNs are sequential (h_t depends on h_(t−1)), so they can't be parallelised over time, false. C: attention is permutation-equivariant, so positional encoding is needed, true. D: LSTMs rely on sigmoid gates and tanh activations, false. Answer: A and C.

</details>

**Q25. [Week 7: Attention]** · _Mock-style_

A transformer processes a sequence of T = 9 tokens with embedding dimension 16. What is the size of the attention matrix A = softmax(QKᵀ) for one head?

- A. 9 × 9
- B. 16 × 16
- C. 9 × 16
- D. 16 × 9

<details><summary>Answer</summary>

**A.** Q and K are T × D, so QKᵀ is T × T: one weight per pair of tokens. Its cost grows quadratically with T.

Why each option:

- A ✓ Correct: T × T.
- B ✗ That's D × D, the weight matrix size.
- C ✗ That's the shape of Q or the output.
- D ✗ Transposed shape of Q.

In depth: Shapes: X is T × d = 9 × 16. Q = XW_q is 9 × D, K is 9 × D. QKᵀ = (9 × D)(D × 9) = 9 × 9. One score for each (query token, key token) pair. Softmax keeps the shape; A·V = (9 × 9)(9 × D) gives 9 × D outputs. The 9 × 9 size (T²) is why attention is quadratic in sequence length.

</details>

**Q26. [Week 7: Multi-head]** · _Mock-style_

An embedding dimension of 768 is split into 12 attention heads. What is the dimension per head?

- A. 64
- B. 12
- C. 768
- D. 9,216

<details><summary>Answer</summary>

**A.** 768 / 12 = 64. The heads' outputs are concatenated back to 768.

Why each option:

- A ✓ Correct: 768/12 = 64.
- B ✗ 12 is the number of heads.
- C ✗ That's the full model dimension.
- D ✗ Multiplied instead of divided.

In depth: Per-head dimension = d_model / h = 768 / 12 = 64. Each head projects to 64-dim Q, K, V; attention runs per head; the 12 outputs of size 64 are concatenated: 12 × 64 = 768. This is GPT-2 small / BERT-base sizing.

</details>

**Q27. [Week 8: VAE]** · _Mock-style_

Consider the 4 statements about VAEs:  
A/ The KL divergence is symmetric  
B/ The encoder outputs μ and ln(σ²)  
C/ The reparameterisation trick samples z = μ + σ·ε with ε ~ N(0,1)  
D/ The latent of a standard (non-variational) autoencoder is always Gaussian  
Which combination is true?

- A. B and C
- B. A and B
- C. A and D
- D. C and D

<details><summary>Answer</summary>

**A.** KL is not symmetric, and a plain AE's latent distribution is uncontrolled. That's why the VAE adds the KL term.

Why each option:

- A ✓ Correct: B and C.
- B ✗ A is false.
- C ✗ A and D are false.
- D ✗ D is false.

In depth: A: KL(p‖q) ≠ KL(q‖p) in general, so false. B: the encoder outputs μ and ln σ² (2 × latent dim), so true. C: reparameterisation z = μ + σ·ε, ε ~ N(0,1), so true. D: a plain AE puts no constraint on its latent distribution, so false (that's why it's a poor generator). Answer: B and C.

</details>

**Q28. [Week 9: GAN]** · _Mock-style_

Which statement about GANs is NOT true?

- A. The discriminator is trained to output 1 for real images and 0 for fake images
- B. The generator maps a random latent vector to a sample
- C. The discriminator is required at generation (inference) time
- D. A conditional GAN feeds a condition such as the class to the generator

<details><summary>Answer</summary>

**C.** After training, only the generator is needed to create data. The discriminator only drives training.

Why each option:

- A ✗ True.
- B ✗ True.
- C ✓ NOT true, so this is the answer: only G is needed at inference.
- D ✗ True.

In depth: Check each statement. The discriminator's labels: real = 1, fake = 0 (true). The generator maps z to a sample (true). A cGAN conditions G on a class (true). Is D required at generation? No. D only provides the training signal; after training you sample z and run G alone. So C is the NOT-true statement.

</details>

**Q29. [Week 9: Flow matching]** · _Mock-style_

Which statement about flow matching is NOT true?

- A. It learns a velocity field between noise and data
- B. Sampling solves an ODE with an ODE solver
- C. Sampling requires a random walk with Langevin dynamics
- D. Several steps are needed because the learnt velocity field is non-linear

<details><summary>Answer</summary>

**C.** Langevin random walks are used by score-based diffusion. Flow matching follows a deterministic ODE.

Why each option:

- A ✗ True.
- B ✗ True.
- C ✓ NOT true, so this is the answer: no random walk.
- D ✗ True.

In depth: Flow matching learns a velocity field u_θ(x, t) (true) and generates by integrating an ordinary differential equation with a solver (true). Several steps are needed because the learned field isn't a straight line everywhere (true). Langevin dynamics, with its random noise at each step, belongs to score-based diffusion (SDE). Flow matching's ODE path is deterministic. So C is NOT true.

</details>

**Q30. [Week 9: High dimension]** · _Mock-style_

Samples are drawn from an isotropic normal distribution N(0, I) in D = 400 dimensions. What is the typical norm of a sample?

- A. About 20
- B. About 1
- C. About 400
- D. About 0

<details><summary>Answer</summary>

**A.** The norm concentrates around √D = √400 = 20, and random pairs are nearly orthogonal.

Why each option:

- A ✓ Correct: √400 = 20.
- B ✗ A norm that small almost never happens.
- C ✗ That's the squared norm.
- D ✗ The mass isn't at the centre.

In depth: E‖z‖² = Σ E[zᵢ²] = 400 × Var(zᵢ) = 400 × 1 = 400, so ‖z‖ ≈ √400 = 20. The spread around 20 is small (std ≈ 0.7), so virtually all samples are on the shell of radius 20. Two such samples have cosine ≈ 0 ± 1/√400 = ±0.05, so they're nearly orthogonal.

</details>

**Q31. [Week 10: NeRF encoding]** · _Mock-style_

In NeRF, the viewing direction is a 3D unit vector encoded with L = 4 frequencies (sin and cos). What is the encoded dimension?

- A. 24
- B. 8
- C. 12
- D. 60

<details><summary>Answer</summary>

**A.** 3 components × 2L = 3 × 8 = 24. For position, L = 10 gives 60.

Why each option:

- A ✓ Correct.
- B ✗ That's 2L for one component.
- C ✗ Forgets the sin/cos pair.
- D ✗ That's the position encoding.

In depth: Dimension = components × 2L = 3 × (2 × 4) = 24. With the direction as 2 angles (θ, φ): 2 × 8 = 16. A lower L is used for direction than for position (L = 10, giving 60) because view-dependent effects are smoother than geometry.

</details>

**Q32. [Week 10: Gaussian splatting]** · _Mock-style_

Which statement about Gaussian splatting is NOT true?

- A. It represents the scene densely with an MLP queried at every point in space
- B. Projected 3D Gaussians become 2D Gaussians on the image
- C. It is typically initialised from a Structure-from-Motion point cloud
- D. Gaussians are cloned and split during optimisation

<details><summary>Answer</summary>

**A.** Dense MLP queries are NeRF. Splatting is sparse: Gaussians only where density is non-zero.

Why each option:

- A ✓ NOT true, so this is the answer: that's NeRF.
- B ✗ True.
- C ✗ True.
- D ✗ True.

In depth: Splatting: 3D Gaussians project to 2D Gaussians on the image (true), are initialised from an SfM point cloud (true), and are cloned or split to add detail (true). A dense MLP queried at every point in space is NeRF's approach, the very thing splatting avoids. So A is NOT true.

</details>

**Q33. [Week 11: Advanced methods]** · _Mock-style_

Match the concepts 1. ViT, 2. DINO, 3. Latent diffusion, 4. JEPA with the statements:  
A. Predicts target embeddings from context in representation space, without negatives  
B. Splits an image into patches used as tokens  
C. Self-distillation with an EMA teacher  
D. Runs diffusion in the latent space of a VAE  
Which combination is correct?

- A. 1-B, 2-C, 3-D, 4-A
- B. 1-A, 2-B, 3-C, 4-D
- C. 1-C, 2-B, 3-A, 4-D
- D. 1-B, 2-D, 3-C, 4-A

<details><summary>Answer</summary>

**A.** ViT = patches, DINO = self-distillation, Stable Diffusion = latent diffusion, JEPA = non-contrastive embedding prediction.

Why each option:

- A ✓ Correct.
- B ✗ Matches in order, which is wrong.
- C ✗ Swaps ViT and DINO.
- D ✗ Swaps DINO and latent diffusion.

In depth: ViT turns patches into tokens (1-B). DINO uses self-distillation with an EMA teacher (2-C). Latent diffusion (Stable Diffusion) runs diffusion inside a VAE latent space (3-D). JEPA predicts target embeddings from context without negatives (4-A). Answer: 1-B, 2-C, 3-D, 4-A.

</details>

<a id="week-1"></a>

## Week 1: Introducing AI and Machine Learning

**Q34. [Week 1: AI vs ML]**

How is machine learning (ML) related to artificial intelligence (AI)?

- A. ML is a subset of AI in which machines learn patterns from data instead of following hand-written rules
- B. AI is a subset of ML that only applies to robotics
- C. ML and AI are unrelated fields
- D. ML is the hardware that AI software runs on

<details><summary>Answer</summary>

**A.** AI is intelligence demonstrated by machines. ML is the part of AI where the machine learns the model (thresholds, parameters) from data.

Why each option:

- A ✓ Correct: ML is the data-driven subset of AI.
- B ✗ Reversed. ML sits inside AI, and robotics is only one application.
- C ✗ They are tightly related: ML is today the dominant way of building AI.
- D ✗ ML is a family of algorithms, not hardware. GPUs are the hardware that made it practical.

In depth: AI is the broad goal: machines showing intelligent behaviour. It includes hand-written expert systems (GOFAI), search, planning, and learning. ML is the sub-field where the behaviour is not programmed rule by rule but fitted from examples. The model has free parameters (weights, thresholds) and an optimiser adjusts them to minimise an error on data. Deep learning is in turn a subset of ML that uses multi-layer neural networks. Remember the nesting: AI ⊃ ML ⊃ DL.

</details>

**Q35. [Week 1: Classification]**

Which of the following is a classification task?

- A. Predicting whether a chest X-ray shows pneumonia or not
- B. Predicting next year's krill density from water temperature
- C. Predicting the price of a house in dollars
- D. Predicting tomorrow's temperature in degrees

<details><summary>Answer</summary>

**A.** Classification predicts a discrete class label. The other three predict a continuous value, which makes them regression.

Why each option:

- A ✓ Correct: two discrete outcomes, so binary classification.
- B ✗ Krill density is a continuous number, so regression.
- C ✗ A dollar price is continuous, so regression.
- D ✗ Temperature in degrees is continuous, so regression.

In depth: The question to ask is: is the output a category or a quantity? Classification maps an input to one of a finite set of labels (pneumonia / no pneumonia). The model usually outputs a score or probability per class, and we pick the highest. Regression maps to a continuous value on a number line. The same input (an X-ray) could be used for either, depending on what we predict: 'pneumonia yes/no' is classification, 'lung opacity percentage' is regression.

</details>

**Q36. [Week 1: Regression]**

Which of the following is a regression task?

- A. Estimating a patient's heart rate (beats per minute) from a video
- B. Deciding if a loan should be approved or denied
- C. Identifying which of 1000 object classes is in an image
- D. Detecting if an email is spam

<details><summary>Answer</summary>

**A.** Regression outputs a continuous number. Loan approval, object recognition and spam detection all output a class.

Why each option:

- A ✓ Correct: beats per minute is a continuous quantity.
- B ✗ Approve/deny is a two-class decision, so classification.
- C ✗ Choosing 1 of 1000 classes is multi-class classification.
- D ✗ Spam/not spam is binary classification.

In depth: Regression predicts a real-valued quantity, and we measure error by how far the prediction is (e.g. squared error), not just right/wrong. Heart rate in bpm can be 71.4 or 72.0, and being off by 1 bpm is better than being off by 20. That notion of distance is why regression uses losses like MSE. Classification tasks use losses like cross-entropy over classes.

</details>

**Q37. [Week 1: GOFAI vs connectionism]**

What is the main difference between classic AI (GOFAI, symbolism) and modern AI (machine learning, connectionism)?

- A. In GOFAI a human expert writes the rules or analytical model; in ML the machine learns parameters of a generic model from data
- B. GOFAI uses neural networks while ML uses decision rules
- C. GOFAI requires large datasets while ML requires none
- D. There is no difference; they are two names for the same approach

<details><summary>Answer</summary>

**A.** Symbolic AI encodes human-written logic and physics models. Connectionism learns from data, which is what made it take off once computers became fast and data became plentiful.

Why each option:

- A ✓ Correct: humans write the rules in GOFAI, while the machine learns the parameters in ML.
- B ✗ Reversed. Neural networks are the connectionist (modern) approach.
- C ✗ Reversed. Data-hungry is the ML side; GOFAI needs expertise rather than data.
- D ✗ They are historically and conceptually opposite approaches.

In depth: Symbolism (GOFAI) assumes intelligence is manipulation of explicit symbols and rules, which a human expert must write down: IF income < X THEN deny, or a physics equation with a few fitted constants. Connectionism (neural networks) assumes knowledge is distributed in many numeric weights learned from data. GOFAI was limited by the expert's ability to formalise knowledge. Connectionism was held back for decades by slow computers and lack of data, until PCs, the Internet (data) and GPUs arrived.

</details>

**Q38. [Week 1: Supervised vs unsupervised]**

Which statement correctly contrasts supervised and unsupervised learning?

- A. Supervised learning uses labelled input-output pairs; unsupervised learning finds structure (e.g. clusters) in unlabelled data
- B. Supervised learning needs no data; unsupervised learning needs labelled data
- C. Unsupervised learning always gives higher accuracy than supervised learning
- D. Supervised learning can only do regression, unsupervised can only do classification

<details><summary>Answer</summary>

**A.** Labels are the key difference. Labelling is expensive, and it's a big business, which is why unsupervised and self-supervised methods matter.

Why each option:

- A ✓ Correct: labels are the key difference.
- B ✗ Both need data, and supervised is the one that needs labels.
- C ✗ Not true. They solve different problems, and accuracy depends on the task.
- D ✗ Supervised covers both classification and regression; unsupervised covers clustering, density estimation and more.

In depth: Supervised learning learns a mapping x → y from pairs where the 'teacher' signal y is given (labels, measured values). Unsupervised learning has only x and must discover structure: clusters (k-means, GMM), low-dimensional manifolds (PCA, autoencoders), or densities. Supervised learning is usually more accurate for a fixed task but depends on costly, sometimes noisy labels; labelling is a whole industry. Unsupervised and self-supervised approaches exploit the huge amount of unlabelled data.

</details>

**Q39. [Week 1: System 1 / System 2]**

According to the lecture, which describes Brain System 2?

- A. Conscious thinking based on reasoning that can solve unknown problems, but is relatively slow and does one task at a time
- B. Automated unconscious actions learned through repetition
- C. Exact and vast memory that always gives the same answer
- D. A fast multitask system that only solves known problems

<details><summary>Answer</summary>

**A.** System 2 is conscious reasoning. Current ML is closer to System 1: with enough data it learns skills, but we don't know how to make a machine reason about problems it has never seen.

Why each option:

- A ✓ Correct: this is the System 2 column of the slide.
- B ✗ This describes System 1.
- C ✗ This describes a traditional computer.
- D ✗ This describes System 1.

In depth: The lecture's comparison table has three columns. A traditional computer has exact, vast memory, is fast and reliable, but only solves problems coded by humans. System 1 handles automated unconscious actions and known problems, needs learning, and is relatively fast and multitask. System 2 is conscious thinking that solves unknown problems by reasoning, but is relatively slow and does one task at a time. Being able to place each property in the right column is a typical exam check.

</details>

**Q40. [Week 1: Data drift]**

An AI system works well at deployment, but months later the hospital replaces its camera and accuracy drops. This is an example of:

- A. Data drift
- B. Bias
- C. Overfitting to the test set
- D. Underfitting

<details><summary>Answer</summary>

**A.** Data drift: over time the data changes (new camera, new circumstances), so inputs map to different locations than the ones learned during training.

Why each option:

- A ✓ Correct: the input distribution changed over time.
- B ✗ Bias is systematic error for a subgroup present from the start, not a change over time.
- C ✗ Overfitting to the test set comes from repeatedly tuning on test data, not hardware change.
- D ✗ Underfitting would have shown poor performance from the start.

In depth: Data drift (or dataset shift) means the statistical distribution of inputs changes after deployment. Here the new camera changes colours, sharpness and noise, so images map to different locations in the learned feature space, and the decision boundaries no longer fit. The model itself hasn't changed; the world has. Mitigations include monitoring input statistics, recalibrating or retraining on new data, and data augmentation that covers equipment variation.

</details>

**Q41. [Week 1: Generalisation]**

A new patient's scan does not resemble anything seen during training, and the model gives a wrong answer. Which failure mode is this?

- A. Lack of generalisation (unknown data / not enough training data)
- B. Data drift caused by changing equipment
- C. Label noise
- D. Vanishing gradients

<details><summary>Answer</summary>

**A.** 'Unknown data': the model fails to generalise because the training data did not cover this case.

Why each option:

- A ✓ Correct: the case was not represented in the training data.
- B ✗ Drift needs a change over time (e.g. new equipment); here the case was simply never covered.
- C ✗ Label noise means wrong training labels, which isn't described here.
- D ✗ Vanishing gradients are a training optimisation problem, not a deployment failure.

In depth: Supervised models interpolate within the region of input space covered by training data. A case unlike anything in training lands in an unexplored region, where the decision boundary is essentially arbitrary, so the prediction is unreliable. The model often doesn't signal uncertainty, which makes this dangerous. The lecture calls it 'unknown data' or lack of generalisation. Remedies: more diverse training data, out-of-distribution detection, and calibrated uncertainty.

</details>

**Q42. [Week 1: Bias]**

A cancer model was trained with healthy white volunteers and white cancer patients. Black patients with cancer get mapped close to healthy white volunteers and are missed. What is this an example of?

- A. Bias
- B. Data drift
- C. Adversarial attack
- D. Backdoor attack

<details><summary>Answer</summary>

**A.** Bias: the learned mapping is wrong for a subgroup that was under-represented in training.

Why each option:

- A ✓ Correct: systematic error for an under-represented subgroup.
- B ✗ Nothing changed over time, so it's not drift.
- C ✗ No one deliberately perturbed the inputs.
- D ✗ No one poisoned the training data with a trigger.

In depth: This is the lecture's bias example. The training set lacked black patients, so the model never learned where their scans should map. Their cancer scans end up near the healthy cluster and are missed: higher false negatives for one subgroup. Overall accuracy can still look high, which is why fairness evaluation must report performance per subgroup. It's a sampling bias problem, fixed by representative data collection and subgroup analysis.

</details>

**Q43. [Week 1: ML as maps]**

In the lecture's 'machine learning = creating maps' view, what is an embedding / latent variable?

- A. The location (vector of numbers) that the model maps an input to, where similar inputs end up close together
- B. The raw pixel values of the input image
- C. The label provided by a human annotator
- D. The learning rate used during training

<details><summary>Answer</summary>

**A.** Layers of additions and multiplications map, for example, 36 million pixel numbers to a location on a map. That location is the embedding, or latent variable.

Why each option:

- A ✓ Correct: an embedding is the learned location (vector) on the map.
- B ✗ Pixels are the raw input before mapping.
- C ✗ Labels are the supervision targets, not the internal representation.
- D ✗ The learning rate is an optimiser hyperparameter.

In depth: A phone image is 4000 × 3000 × 3 = 36 million numbers. A network's layers of multiplications and additions transform those numbers into a short vector: a location on a learned 'map' (latent space). Similar images land near each other, so a new image can be classified by where it lands. Generative AI goes the other way: pick a location and decode it into data. That location vector is the embedding. 'Latent' means it's a hidden variable not directly observed.

</details>

**Q44. [Week 1: Generative AI]**

In the 'map' view of ML, how does a chatbot such as ChatGPT generate text?

- A. It guesses the next location (token) on the map given the previous ones
- B. It retrieves complete answers from a database of human-written responses
- C. It runs hand-written grammar rules
- D. It clusters the question with k-means

<details><summary>Answer</summary>

**A.** LLMs are autoregressive: they predict the next token's location on the map, append it, and repeat.

Why each option:

- A ✓ Correct: next-token prediction, repeated.
- B ✗ LLMs do not retrieve whole stored answers (unless augmented with retrieval tools).
- C ✗ Hand-written grammar rules are GOFAI, not LLMs.
- D ✗ k-means is a clustering algorithm unrelated to text generation.

In depth: An LLM maps the text so far (as tokens) to a probability distribution over the next token. It samples one, appends it, and repeats: autoregressive generation. In map terms it's moving step by step along a trajectory in embedding space. It doesn't look up stored answers. Everything is generated from learned weights, which is also why it can hallucinate plausible but false text.

</details>

**Q45. [Week 1: History]**

What triggered the 2012 paradigm shift in AI?

- A. Convolutional neural networks trained on ImageNet using GPUs: data-driven models with no hand-crafted model
- B. The invention of decision trees
- C. The invention of symbolic logic programming
- D. The first chess program beating a human

<details><summary>Answer</summary>

**A.** In 2012 CNNs won the ImageNet challenge (14M images, 20,000+ categories). Compute (GPUs) and plenty of data made data-driven models win.

Why each option:

- A ✓ Correct: deep CNNs plus ImageNet plus GPUs.
- B ✗ Decision trees were decades older and didn't cause this shift.
- C ✗ Symbolic logic programming is the GOFAI era (1970s-80s).
- D ✗ Chess (Deep Blue, 1997) used search, not data-driven deep learning.

In depth: In 2012 AlexNet, a deep CNN trained on GPUs, won the ImageNet challenge by a large margin over hand-engineered feature pipelines. The lecture links this to the 1872 shift in painting (impressionism): move from building explicit models to letting data speak. The enablers were (1) generic models such as neural networks, (2) huge labelled datasets (ImageNet: 14M images, 20k+ categories), and (3) cheap parallel compute (GPUs).

</details>

**Q46. [Week 1: AGI / ASI]**

Which statement best distinguishes AGI from ASI?

- A. AGI is human-level general intelligence across many tasks; ASI would surpass human intelligence
- B. AGI is narrow AI for one task; ASI is AI for two tasks
- C. AGI and ASI are two image classification benchmarks
- D. ASI is a weaker version of AGI that runs on phones

<details><summary>Answer</summary>

**A.** Artificial General Intelligence means human-level ability across tasks. Artificial Super Intelligence means beyond human. Open questions include whether scaling alone can get there, and the ethics.

Why each option:

- A ✓ Correct: general human-level vs beyond-human.
- B ✗ Nonsense definitions, since 'general' means broad, not one task.
- C ✗ They are concepts, not benchmarks.
- D ✗ Reversed: ASI is beyond AGI, not weaker.

In depth: Today's systems are mostly narrow AI: very good at specific tasks. AGI would match humans across the full range of cognitive tasks, including learning new skills on its own. ASI would exceed the best humans in essentially every domain. The lecture's open questions: can AGI be reached by scaling current models? Is today's AI 'nowhere near super intelligence'? And what ethical issues arise if ASI does not share human goals?

</details>

**Q47. [Week 1: Intelligence]**

Which point did the lecture raise when questioning whether current AI is truly 'intelligent'?

- A. The human brain uses about 20 W and can learn new skills from few examples, whereas AI needs huge data and energy
- B. AI cannot perform any useful task
- C. Computers have too little memory to store a neural network
- D. AI systems always explain their reasoning

<details><summary>Answer</summary>

**A.** Intelligence questions include learning new skills on its own, solving never-seen problems, and efficiency: the brain runs on ~20 W.

Why each option:

- A ✓ Correct: the energy and data-efficiency argument.
- B ✗ Wrong: the lecture stresses current AI can do a lot.
- C ✗ Memory is not the limitation discussed.
- D ✗ Wrong: deep models are notoriously hard to explain.

In depth: The lecture questions whether scaling data and compute is the same as intelligence. A human brain runs on ~20 W and can learn a new skill from a handful of examples; large models need enormous datasets and megawatts. True intelligence would also mean learning new skills autonomously and solving never-seen problems. Current AI is still very useful, and it will change how we work, even if it isn't intelligent in that full sense.

</details>

<a id="week-2"></a>

## Week 2: Classical Machine Learning

**Q48. [Week 2: Supervised ML]**

What is the core assumption made when performing supervised machine learning?

- A. There is a relationship (mapping) between the input (independent) variables and the output (dependent) variable that can be modelled
- B. The data has no structure
- C. All features are independent of the target
- D. The test set is used to fit the parameters

<details><summary>Answer</summary>

**A.** All supervised ML models a relationship between inputs and outputs. The model is a hypothesis about that mapping.

Why each option:

- A ✓ Correct: we assume a learnable input-output relationship.
- B ✗ If the data had no structure, there would be nothing to learn.
- C ✗ The opposite: features must carry information about the target.
- D ✗ The test set must never be used for fitting.

In depth: Supervised learning assumes y = f(x) + noise for some unknown function f, and that the training pairs are samples of that same relationship that will hold for future data. The model is our hypothesis for f, and training tunes its parameters so predictions match the examples. If no relationship exists, or it changes at deployment, no amount of fitting helps. 'All models are wrong, but some are useful' (George Box).

</details>

**Q49. [Week 2: Correlation]**

Why do we check data correlation (e.g. Pearson's coefficient) before performing linear regression?

- A. To check there is a linear relationship between input and output worth modelling
- B. To compute the learning rate
- C. To remove the need for a test set
- D. Because regression only works on uncorrelated data

<details><summary>Answer</summary>

**A.** Pearson's correlation measures linear correlation. Spearman's measures monotonic, possibly non-linear, correlation.

Why each option:

- A ✓ Correct: to confirm a linear relationship is worth modelling.
- B ✗ The learning rate is unrelated to correlation.
- C ✗ You always need a held-out test set.
- D ✗ Regression needs correlation between x and y; it doesn't need uncorrelated data.

In depth: Linear regression assumes y changes linearly with x. Pearson's r measures exactly that: r = cov(x, y)/(σₓσᵧ) ranges from −1 to 1, and r ≈ 0 means no linear relationship. Checking first tells you whether a line is a sensible model. If the plot is curved but monotonic, Spearman's correlation may still be high and you might transform the data or use polynomial regression. Correlation is not causation, but no correlation means a linear model will explain little.

</details>

**Q50. [Week 2: Correlation]**

Which coefficient should you use to detect a non-linear but monotonic relationship?

- A. Spearman's rank correlation
- B. Pearson's correlation
- C. Mean squared error
- D. Cross-entropy

<details><summary>Answer</summary>

**A.** Spearman's correlation works on ranks, so it captures any monotonic relationship, not only linear ones.

Why each option:

- A ✓ Correct: rank-based, so it captures monotonic trends.
- B ✗ Pearson only measures linear association.
- C ✗ MSE is a loss for regression, not a correlation measure.
- D ✗ Cross-entropy is a classification loss.

In depth: Spearman's ρ is Pearson's correlation computed on the ranks of the data. Any strictly increasing relationship (y = eˣ, y = log x) gives ρ = 1 even though it's not linear. Use Pearson for linear relationships and Spearman for monotonic non-linear ones. Neither detects non-monotonic relationships such as a U-shape, so always plot the data too.

</details>

**Q51. [Week 2: OLS]**

What is the ordinary least squares (OLS) solution for the parameters W in the linear system AW = B?

- A. W = (AᵀA)⁻¹AᵀB
- B. W = A⁻¹Bᵀ
- C. W = AᵀB(AAᵀ)
- D. W = (A + B)⁻¹

<details><summary>Answer</summary>

**A.** Setting the derivative of the squared error to 0 gives the normal equations AᵀAW = AᵀB, so W = (AᵀA)⁻¹AᵀB (the pseudo-inverse).

Why each option:

- A ✓ Correct: the normal-equation solution.
- B ✗ A is usually not square, so it has no inverse; the transpose placement is also wrong.
- C ✗ Wrong matrix order: AAᵀ is N×N, not the parameter-sized matrix.
- D ✗ Adding matrices of different shapes makes no sense here.

In depth: Write the model for all N samples as AW = B, where each row of A is [xᵢ, 1] and W = [m, c]ᵀ. The squared error is E = ‖AW − B‖². Setting ∂E/∂W = 0 gives the normal equations AᵀAW = AᵀB, hence W = (AᵀA)⁻¹AᵀB. (AᵀA)⁻¹Aᵀ is the Moore-Penrose pseudo-inverse. The same formula works for multiple variables (more columns in A) and for polynomial features.

</details>

**Q52. [Week 2: Linear regression]**

For the model y = mx + c, what are the parameters and how are they found?

- A. m and c, found by minimising the error between predictions and ground truth (e.g. least squares)
- B. x and y, found by grid search on the test set
- C. The learning rate and number of epochs, found analytically
- D. The polynomial order, found by OLS

<details><summary>Answer</summary>

**A.** Parameters (m, c) are chosen to minimise the residual error. This can be done analytically (OLS), by grid search, or by gradient descent.

Why each option:

- A ✓ Correct: m and c, fitted by minimising the error.
- B ✗ x and y are data, and the test set is never used for fitting.
- C ✗ Learning rate and epochs are optimiser hyperparameters.
- D ✗ Polynomial order is a hyperparameter of polynomial regression.

In depth: Parameters are the numbers the training procedure adjusts: slope m and intercept c. x is the input and y the output; they're data, not parameters. We choose m and c to minimise the residuals (ground truth minus prediction), usually the sum of squared residuals. The lecture shows three ways: the closed-form OLS solution, grid search over (m, c), and gradient descent (from scratch and with PyTorch).

</details>

**Q53. [Week 2: Assumptions]**

Which is NOT one of the four main assumptions of linear regression?

- A. The input features must follow a uniform distribution
- B. Linearity between input and output
- C. Independence of the residuals
- D. Homoscedasticity (constant variance of residuals)

<details><summary>Answer</summary>

**A.** The four assumptions are linearity, independence, homoscedasticity, and normality of residuals (centred at 0). Nothing requires uniform inputs.

Why each option:

- A ✓ Correct (the false assumption): no distribution is required for the inputs.
- B ✗ Linearity is assumption 1.
- C ✗ Independence of residuals is assumption 2.
- D ✗ Homoscedasticity is assumption 3.

In depth: The four classic assumptions are about the relationship and the residuals: (1) Linearity: E[y|x] is linear in x. (2) Independence: residuals are independent of each other (no autocorrelation, as can happen with time series). (3) Homoscedasticity: residual variance is constant across x. (4) Normality: residuals are normally distributed around 0. Nothing is assumed about the distribution of x itself, so 'uniform inputs' is the odd one out.

</details>

**Q54. [Week 2: Assumptions]**

What does homoscedasticity mean in linear regression?

- A. The residuals have constant variance at every level of the input
- B. The residuals follow a uniform distribution
- C. The inputs are all on the same scale
- D. The output is always positive

<details><summary>Answer</summary>

**A.** Homoscedasticity is constant residual variance. If the spread of errors grows with x, the assumption is violated.

Why each option:

- A ✓ Correct: constant residual variance.
- B ✗ The residuals should be normal (that's assumption 4), not uniform.
- C ✗ Feature scaling is a separate preprocessing step.
- D ✗ Nothing requires a positive output.

In depth: Homo = same, scedastic = scatter. In a residual plot (residual vs x or vs prediction), the cloud should be a band of constant width. If it fans out (e.g. errors grow with house price), the data is heteroscedastic: standard errors and confidence intervals are wrong and the fit is dominated by the high-variance region. Fixes include transforming y (e.g. log) or weighted least squares.

</details>

**Q55. [Week 2: Polynomial regression]**

How can polynomial regression (e.g. y = w₀ + w₁x + w₂x²) be solved as an ordinary least squares problem?

- A. Build a matrix A whose columns are 1, x, x², …; the model is still linear in the parameters, so W = (AᵀA)⁻¹AᵀB applies
- B. It cannot. Polynomial regression always needs gradient descent
- C. Take the log of the output and use a decision tree
- D. Use k-means to find the coefficients

<details><summary>Answer</summary>

**A.** The model is non-linear in x but linear in the weights, so the same OLS solution works with polynomial feature columns.

Why each option:

- A ✓ Correct: polynomial features plus the same OLS formula.
- B ✗ Gradient descent is possible but not required; OLS works.
- C ✗ Log-transforming and using trees is a different, unrelated method.
- D ✗ k-means does clustering, not coefficient fitting.

In depth: 'Linear' in linear least squares refers to the parameters, not the input. y = w₀ + w₁x + w₂x² is a weighted sum of fixed basis functions (1, x, x²). Build A with one column per basis function: row i = [1, xᵢ, xᵢ²]. Then AW = B and W = (AᵀA)⁻¹AᵀB exactly as before. This is the general trick behind basis-function regression.

</details>

**Q56. [Week 2: Polynomial regression]**

What is the difference between a linear regression model and a 1st order polynomial regression model?

- A. There is no difference; both are y = w₀ + w₁x
- B. The 1st order polynomial includes an x² term
- C. Linear regression has no intercept
- D. The polynomial model must be solved with gradient descent

<details><summary>Answer</summary>

**A.** A 1st order polynomial is a straight line, which is exactly linear regression.

Why each option:

- A ✓ Correct: identical models.
- B ✗ The x² term only appears at order 2.
- C ✗ Linear regression does have an intercept c.
- D ✗ Both can be solved with OLS.

In depth: A polynomial of order N is y = Σₖ wₖxᵏ for k = 0…N. For N = 1 that's y = w₀ + w₁x, which is the straight line y = c + mx. Linear regression is the 1st-order special case of polynomial regression. Exams sometimes test this to check you're not confused by the different names.

</details>

**Q57. [Week 2: Hyperparameters]**

What is the hyperparameter of a polynomial regression model?

- A. The polynomial order (degree)
- B. The coefficients w₀…w_N
- C. The residuals
- D. The ground-truth values

<details><summary>Answer</summary>

**A.** The order is chosen with validation data. The coefficients are parameters fitted on training data.

Why each option:

- A ✓ Correct: the polynomial order.
- B ✗ The coefficients are parameters, learned by OLS.
- C ✗ Residuals are errors computed after fitting.
- D ✗ Ground-truth values are data.

In depth: Hyperparameters are chosen before training and control model complexity; parameters are learned during training. For polynomial regression the order N decides how flexible the curve is. Too low underfits; too high overfits, wiggling through every training point. Select N by comparing error on validation data (or with k-fold cross-validation), then report the final performance on the test set.

</details>

**Q58. [Week 2: Overfitting]**

How can overfitting be identified?

- A. Error is low on the training data but much higher on validation (unseen) data
- B. Error is high on both training and validation data
- C. Training error keeps increasing
- D. Validation error is lower than training error

<details><summary>Answer</summary>

**A.** An overfitted model (e.g. a high-order polynomial) memorises training points and fails to generalise. Compare performance on data not used for fitting.

Why each option:

- A ✓ Correct: low training error, high validation error.
- B ✗ That's underfitting: the model is too simple.
- C ✗ Training error normally decreases during training.
- D ✗ Validation lower than training is unusual (e.g. dropout effects); it isn't the overfitting signature.

In depth: An overfitted model has learned the noise in the training data rather than the underlying pattern. Its signature is a gap: training error is small (it may even be zero for a high-order polynomial) while error on held-out data is large. Plotting both errors as model complexity grows gives a U-shaped validation curve; the bottom of the U is the best complexity. High error on both sets is underfitting instead.

</details>

**Q59. [Week 2: Experimental design]**

In best-practice experimental design, what is the validation set used for?

- A. Selecting hyperparameters (e.g. polynomial order, K in kNN)
- B. Reporting the final performance of the model
- C. Fitting the model parameters
- D. Nothing; it is optional decoration

<details><summary>Answer</summary>

**A.** Training data fits parameters, validation data selects hyperparameters, and testing data (kept secret, ideally from another source) estimates final performance.

Why each option:

- A ✓ Correct: validation data selects hyperparameters.
- B ✗ Reporting final performance is the test set's job.
- C ✗ Fitting parameters is the training set's job.
- D ✗ Validation is essential to avoid tuning on the test set.

In depth: Three roles for data. Training data fits the parameters. Validation data selects hyperparameters (polynomial order, K, learning rate, architecture) and is used for early stopping. Test data is kept secret until the end to estimate real-world performance once. If you tune on test data, the reported number becomes optimistic, because you have implicitly fitted to it. Ideally the test set even comes from a different source (another hospital or scanner) to measure generalisation.

</details>

**Q60. [Week 2: Gradient descent]**

Gradient descent on the krill data (one variable ~0.05, another ~30) did not follow the steepest slope. What was the easy fix?

- A. Normalise the data so variables have a similar scale
- B. Increase the polynomial order
- C. Use the test set for training
- D. Remove the bias term

<details><summary>Answer</summary>

**A.** With different scales, the gradient is dominated by one variable. After normalisation, gradient descent converges quickly.

Why each option:

- A ✓ Correct: normalisation equalises the scales.
- B ✗ Model order doesn't fix the conditioning problem.
- C ✗ Never train on the test set.
- D ✗ Removing the bias changes the model and doesn't fix the scale issue.

In depth: When one feature is ~0.05 and the other ~30, the loss surface is a very elongated valley. The gradient points mostly along the large-scale direction, so updates zig-zag and crawl along the small-scale one. A learning rate small enough to be stable for one direction is far too small for the other. Normalising (subtract the mean, divide by the standard deviation) makes the valley round, so the gradient points at the minimum and convergence is fast.

</details>

**Q61. [Week 2: Grid search]**

How does grid search find the parameters of a model?

- A. It evaluates the loss for every combination of parameter values on a predefined grid and keeps the best
- B. It follows the negative gradient of the loss
- C. It solves the normal equations analytically
- D. It randomly removes neurons

<details><summary>Answer</summary>

**A.** Grid search is exhaustive and simple, but its cost grows exponentially with the number of parameters.

Why each option:

- A ✓ Correct: exhaustive evaluation on a grid.
- B ✗ That's gradient descent.
- C ✗ That's the analytical OLS solution.
- D ✗ That's dropout.

In depth: Grid search discretises each parameter into a list of values, evaluates the loss for every combination, and keeps the best. It's simple, embarrassingly parallel, and lets you visualise the loss surface, but the cost grows exponentially: k values for each of p parameters means kᵖ evaluations. It suits two or three parameters (or hyperparameters) but not millions of neural network weights, which is why we need gradient descent.

</details>

**Q62. [Week 2: kNN]**

Which sequence describes the K Nearest Neighbours classifier?

- A. Normalise data → store training data → compute distance to training points → take the K closest → majority vote
- B. Initialise K centroids → assign points → update centroids → repeat
- C. Fit a line → compute residuals → apply softmax
- D. Split on the feature that best separates classes → recurse

<details><summary>Answer</summary>

**A.** kNN stores the training data and classifies by majority among the K nearest points. The second option describes k-means, which is unsupervised.

Why each option:

- A ✓ Correct: the kNN procedure.
- B ✗ That's k-means clustering.
- C ✗ That's a linear model with softmax.
- D ✗ That's decision tree induction.

In depth: kNN is a 'lazy' learner: training just stores the (normalised) data. To classify a new point, compute its distance (usually Euclidean) to every training point, take the K closest, and use the majority class (or average, for regression). Normalisation is step 1 because otherwise the feature with the largest numeric range dominates the distance. Don't confuse kNN (supervised, K = number of neighbours) with k-means (unsupervised, K = number of clusters).

</details>

**Q63. [Week 2: kNN]**

Which is a limitation of KNN?

- A. Prediction is slow and memory-heavy for large datasets because all training data must be stored and compared
- B. It cannot be interpreted at all
- C. It requires gradient descent to train
- D. It can only be used for regression

<details><summary>Answer</summary>

**A.** kNN is easy to interpret, but it is slow at inference, sensitive to feature scale, and suffers in high dimension. It works for classification and regression.

Why each option:

- A ✓ Correct: slow and memory-heavy at prediction time.
- B ✗ kNN is actually easy to interpret: you can show the neighbours.
- C ✗ kNN has no gradient-based training.
- D ✗ kNN works for both classification and regression.

In depth: Because kNN keeps the whole training set and compares every query against it, prediction costs O(N·d) time and O(N·d) memory. In high dimension, distances become less informative (curse of dimensionality). It's also sensitive to feature scaling and irrelevant features. Its advantages: it's easy to interpret (you can show the neighbours), has no training phase, and works for both classification and regression.

</details>

**Q64. [Week 2: kNN]**

How should the value of K in KNN be chosen?

- A. Check performance for a range of K values on the validation dataset
- B. Always use K = 1
- C. Choose K that maximises accuracy on the test set
- D. K must equal the number of features

<details><summary>Answer</summary>

**A.** K is a hyperparameter, so it's selected on validation data, never on the test set.

Why each option:

- A ✓ Correct: choose K by validation performance.
- B ✗ K = 1 is very sensitive to noise.
- C ✗ Choosing on the test set inflates the reported performance.
- D ✗ There's no reason K should equal the number of features.

In depth: K controls the bias-variance trade-off. K = 1 follows every noisy point (high variance, overfitting), while a very large K smooths everything towards the majority class (high bias). Pick K by evaluating a range of values on validation data, or with k-fold cross-validation, and choose the best. An odd K avoids ties in binary problems. Choosing K on the test set would leak test information.

</details>

**Q65. [Week 2: kNN regression]**

How can KNN be used for a regression task?

- A. Predict the average (or weighted average) of the target values of the K nearest neighbours
- B. It cannot be used for regression
- C. Use the majority class of the neighbours
- D. Fit a decision tree on the neighbours

<details><summary>Answer</summary>

**A.** For regression, replace the majority vote with an average of the neighbours' values.

Why each option:

- A ✓ Correct: average the neighbours' targets.
- B ✗ kNN works for regression.
- C ✗ Majority voting is for classification.
- D ✗ A decision tree is a different model.

In depth: For regression, the K neighbours' target values are averaged, optionally weighted by 1/distance so closer neighbours count more. The lecture's example estimates a loan amount from similar applicants. The prediction is a piecewise-constant function of the input. It's simple and non-parametric, but it can't extrapolate beyond the range of the training targets.

</details>

**Q66. [Week 2: Decision trees]**

Which is an advantage of decision trees?

- A. They are interpretable: the decision-making follows explicit rules (e.g. income < $35,000)
- B. They never overfit
- C. They always produce smooth, curved boundaries
- D. They require no training data

<details><summary>Answer</summary>

**A.** Trees are interpretable and handle mixed feature types, but they overfit easily and produce axis-aligned boundaries.

Why each option:

- A ✓ Correct: interpretable rule-based decisions.
- B ✗ Deep trees overfit easily.
- C ✗ Boundaries are axis-aligned and step-shaped, not smooth.
- D ✗ Trees are learned from training data.

In depth: A decision tree is a sequence of axis-aligned questions (income < $35,000? LVR > 80%?), each splitting the data to make the groups purer. You can read the path for any prediction, which makes trees one of the few truly explainable models and valuable in regulated domains like lending. The downsides: they overfit when grown deep, they produce blocky axis-aligned boundaries, and small data changes can change the tree a lot.

</details>

**Q67. [Week 2: Pruning]**

How can overfitting in decision trees be prevented?

- A. Pruning: limit tree growth (max depth, min samples per leaf) or remove branches that do not improve validation performance
- B. Grow the tree until every leaf has one sample
- C. Increase the learning rate
- D. Use more features at every split

<details><summary>Answer</summary>

**A.** Pre-pruning stops growth early. Post-pruning grows the full tree and then removes branches that don't help on validation data.

Why each option:

- A ✓ Correct: pre-pruning or post-pruning.
- B ✗ One sample per leaf is maximal overfitting.
- C ✗ Trees have no learning rate.
- D ✗ More features per split increases overfitting risk.

In depth: Pre-pruning stops growth early using limits: maximum depth, minimum samples per split or leaf, or minimum impurity decrease. Post-pruning grows a full tree and then removes subtrees whose removal doesn't hurt validation accuracy (e.g. cost-complexity pruning). Both trade a little training accuracy for better generalisation. Ensembles such as random forests are another remedy.

</details>

**Q68. [Week 2: Random forest]**

What is the difference between a decision tree and a random forest?

- A. A random forest is an ensemble of many trees trained on random subsets of data/features whose predictions are combined by voting
- B. A random forest is a single, very deep tree
- C. A random forest uses neural networks at each node
- D. There is no difference

<details><summary>Answer</summary>

**A.** Several are better than one: averaging many diverse trees reduces overfitting and variance.

Why each option:

- A ✓ Correct: an ensemble of randomised trees with voting.
- B ✗ A forest is many trees, not one deep tree.
- C ✗ There are no neural networks inside.
- D ✗ Ensembling is the key difference.

In depth: A random forest trains many trees, each on a bootstrap sample of the data and considering a random subset of features at each split, so the trees make different errors. Their votes (or averages) are combined. Averaging de-correlated, high-variance models reduces variance a lot while keeping low bias. You lose some interpretability compared with one tree but gain accuracy and robustness. It's the 'several are better than one' slide.

</details>

**Q69. [Week 2: Confusion matrix]**

What does a confusion matrix show?

- A. Counts of predictions for each (ground-truth class, predicted class) pair; the diagonal holds correct predictions
- B. The loss value at each epoch
- C. The correlation between features
- D. The weights of a linear model

<details><summary>Answer</summary>

**A.** To build it, run the model on a dataset and count how often each true class is predicted as each class. Off-diagonal cells are the different types of errors.

Why each option:

- A ✓ Correct: counts per (true, predicted) pair.
- B ✗ That's a training curve.
- C ✗ That's a correlation matrix.
- D ✗ Weights are model parameters.

In depth: To build one, run the model on labelled data and, for each sample, add 1 to the cell (true class, predicted class). Rows are usually ground truth and columns predictions. The diagonal is correct predictions; each off-diagonal cell is a specific kind of mistake (e.g. 'warm' predicted as 'cold'). From the matrix you derive accuracy, and, by focusing on one class, TP/FP/FN/TN, precision, recall and specificity.

</details>

**Q70. [Week 2: Simpson's paradox]**

Simpson's paradox is listed as a limitation for linear regression. What is it?

- A. A trend that appears in several groups of data disappears or reverses when the groups are combined
- B. Gradient descent diverging when the learning rate is too high
- C. The residuals not being normally distributed
- D. Having more features than samples

<details><summary>Answer</summary>

**A.** Pooling sub-groups can hide or reverse a relationship, so a single linear fit over everything can mislead.

Why each option:

- A ✓ Correct: trends can reverse when groups are combined.
- B ✗ That's an optimisation issue.
- C ✗ That's a violation of the normality assumption.
- D ✗ That's an under-determined system (p > n).

In depth: Example: within each of two hospitals a treatment improves outcomes, but pooled together it looks harmful, because one hospital treats more severe cases and uses the treatment more. A hidden (confounding) variable flips the trend. For linear regression this means a single fit on aggregated data can show the opposite relationship to every subgroup. Check subgroups and include confounders as variables.

</details>

<a id="week-3"></a>

## Week 3: ANN and Deep Learning

**Q71. [Week 3: Decision boundary]**

For a linear classifier with score z = w·x + b, what is the decision boundary?

- A. The set of points where z = 0; the vector w is perpendicular to this boundary
- B. The set of points where z = 1; w is parallel to the boundary
- C. The point where the loss is minimum
- D. The line through the origin regardless of b

<details><summary>Answer</summary>

**A.** The vector (a, b) is perpendicular to the line ax + by + c = 0. w sets the orientation and b shifts the boundary.

Why each option:

- A ✓ Correct: the boundary is z = 0 with w normal to it.
- B ✗ The boundary is z = 0, and w is perpendicular to it, not parallel.
- C ✗ The loss minimum is about training, not the boundary definition.
- D ✗ b shifts the line away from the origin.

In depth: z = w·x + b defines a hyperplane z = 0 (a line in 2D). Any vector lying in the boundary satisfies w·v = 0, so w is the normal vector, perpendicular to the boundary. Changing w rotates (and rescales) the boundary; changing b shifts it parallel to itself. Points with z > 0 are one class and z < 0 the other. In 2D, for ax + by + c = 0, (a, b) is perpendicular to the line.

</details>

**Q72. [Week 3: Class score]**

How is the class score of a linear classifier related to the decision boundary?

- A. The score is the (scaled) signed distance of the input from the boundary: positive on one side, negative on the other
- B. The score is the number of misclassified samples
- C. The score is always between 0 and 1
- D. The score is unrelated to the parameters

<details><summary>Answer</summary>

**A.** A neuron projects the input onto w (dot product). The sign gives the class and the magnitude grows with distance from the boundary.

Why each option:

- A ✓ Correct: a scaled signed distance to the boundary.
- B ✗ Misclassification counts are a metric, not the score.
- C ✗ Raw scores are unbounded; only after a sigmoid or softmax are they in [0, 1].
- D ✗ The score is computed directly from w and b.

In depth: The signed distance from x to the hyperplane is (w·x + b)/‖w‖, so the score z is that distance scaled by ‖w‖. Its sign tells the side of the boundary (the class) and its magnitude tells how far, often read as confidence. Geometrically a neuron is a projection of x onto w. That's also why points far from the boundary get extreme scores, even if they're unlike any training data (see Week 5, novel classes).

</details>

**Q73. [Week 3: Loss functions]**

What is the purpose of a loss function?

- A. To give a single number measuring how good the current parameters are, which we minimise to train the model
- B. To initialise the weights
- C. To choose the number of layers
- D. To normalise the input data

<details><summary>Answer</summary>

**A.** The loss (cost, objective) quantifies prediction error. Optimisation then finds the parameters that minimise it.

Why each option:

- A ✓ Correct: a scalar measure of fit that we minimise.
- B ✗ Initialisation is a separate step.
- C ✗ Architecture choice is a design decision.
- D ✗ Normalisation is preprocessing.

In depth: Training needs a single scalar that says how bad the current parameters are, so an optimiser can improve them. The loss compares predictions with targets over the training data (e.g. average cross-entropy). It must be differentiable (or nearly) for gradient descent. Choosing the loss defines what 'good' means: MSE penalises large errors heavily, hinge loss cares only about the margin, cross-entropy about probabilities.

</details>

**Q74. [Week 3: Loss functions]**

Which loss functions are most commonly used for classification and regression respectively?

- A. Cross-entropy for classification; mean squared error for regression
- B. Mean squared error for classification; hinge loss for regression
- C. Accuracy for classification; R² for regression
- D. KL divergence for both

<details><summary>Answer</summary>

**A.** Cross-entropy (softmax loss) and hinge loss are used for classification. MSE is used for regression.

Why each option:

- A ✓ Correct: CE for classification, MSE for regression.
- B ✗ Swapped and incorrect pairing.
- C ✗ Accuracy and R² are evaluation metrics, not differentiable training losses.
- D ✗ KL divergence is used for distributions (e.g. VAE), not as the default for both.

In depth: Regression outputs a number, so we penalise squared distance: MSE = mean((y − ŷ)²). That corresponds to assuming Gaussian noise and doing maximum likelihood. Classification outputs class probabilities via softmax, so we use cross-entropy = −log p(correct class), which is maximum likelihood under a categorical model. Hinge (multiclass SVM) loss is the margin-based alternative for classification. Accuracy isn't differentiable, so it can't be the training loss.

</details>

**Q75. [Week 3: Hinge vs cross-entropy]**

What is a key difference between hinge loss and cross-entropy loss?

- A. Hinge loss becomes zero once the correct class score exceeds others by a margin; cross-entropy keeps pushing the correct class probability towards 1
- B. Cross-entropy is only for regression
- C. Hinge loss requires a softmax
- D. They are mathematically identical

<details><summary>Answer</summary>

**A.** Hinge (multiclass SVM) loss sums over incorrect classes with a margin. Cross-entropy is the negative log-likelihood of the softmax probability of the correct class.

Why each option:

- A ✓ Correct: hinge saturates at the margin; CE never fully saturates.
- B ✗ CE is a classification loss.
- C ✗ Hinge works directly on raw scores, while CE needs softmax.
- D ✗ They behave differently, as explained.

In depth: Hinge loss: L = Σ over j ≠ correct of max(0, sⱼ − s_correct + Δ). Once the correct score beats every other score by margin Δ, the loss is exactly 0 and that sample stops influencing training. Cross-entropy: L = −log(softmax(s)_correct). It's never exactly 0, so it keeps pushing the correct probability towards 1 and keeps rewarding larger margins. Softmax plus CE also gives probabilistic outputs; hinge gives only scores.

</details>

**Q76. [Week 3: Softmax]**

What does the softmax function do?

- A. Normalises a vector of class scores into probabilities that are positive and sum to 1
- B. Sets all negative values to zero
- C. Selects the largest weight in the network
- D. Computes the gradient of the loss

<details><summary>Answer</summary>

**A.** softmax(sᵢ) = exp(sᵢ)/Σexp(sⱼ). It approximates a max function and turns scores into a probability distribution.

Why each option:

- A ✓ Correct: scores become a probability distribution.
- B ✗ That's ReLU.
- C ✗ Softmax doesn't select weights.
- D ✗ Gradients come from backpropagation.

In depth: softmax(s)ᵢ = exp(sᵢ)/Σⱼ exp(sⱼ). Exponentiation makes every value positive and amplifies differences; dividing by the sum makes the outputs a distribution summing to 1. It's a smooth, differentiable approximation of argmax: the largest score gets most of the mass. Combined with cross-entropy it gives the standard classification head. Note that it always distributes all the probability among known classes.

</details>

**Q77. [Week 3: Gradient]**

How is the gradient of the loss used to update model parameters in gradient descent?

- A. Parameters move a small step (learning rate) in the direction opposite to the gradient: w ← w − η∇L
- B. Parameters move in the direction of the gradient to increase the loss
- C. The gradient is added to the input data
- D. The gradient replaces the parameters directly

<details><summary>Answer</summary>

**A.** The gradient points uphill, so we step downhill, scaled by the learning rate.

Why each option:

- A ✓ Correct: step against the gradient.
- B ✗ Moving along the gradient increases the loss.
- C ✗ Inputs are fixed; parameters are updated.
- D ✗ The gradient sets the update direction; it doesn't replace the parameters.

In depth: The gradient ∇L(w) points in the direction of steepest increase of the loss. Moving a small amount against it decreases the loss: w ← w − η∇L(w), where η is the learning rate (step size). Repeat until convergence. With a convex loss (linear regression) this reaches the global minimum; neural network losses are non-convex, but in practice gradient descent finds good minima.

</details>

**Q78. [Week 3: GD variants]**

What distinguishes gradient descent, stochastic gradient descent and mini-batch gradient descent?

- A. GD uses the whole dataset per update, SGD uses one sample, mini-batch uses a small batch of samples
- B. GD uses one sample, SGD uses the whole dataset, mini-batch uses none
- C. They differ only in the activation function
- D. SGD has no learning rate

<details><summary>Answer</summary>

**A.** When data doesn't fit in memory, updates are computed on samples or batches. Mini-batch is the usual compromise.

Why each option:

- A ✓ Correct: full data vs one sample vs mini-batch.
- B ✗ Definitions swapped.
- C ✗ They differ in data per update, not activation.
- D ✗ All variants have a learning rate.

In depth: Batch GD computes the exact gradient over all N samples per update: accurate but expensive, and the data may not fit in memory. SGD uses one random sample per update: cheap and noisy, and the noise can help escape poor minima, but it fluctuates. Mini-batch GD averages over B samples (e.g. 32-256): a good gradient estimate that uses GPU parallelism efficiently. It's what 'SGD' usually means in practice.

</details>

**Q79. [Week 3: Momentum]**

What does adding momentum to gradient descent achieve?

- A. Like a heavy ball rolling in a canyon, it smooths updates, reducing oscillations and accelerating along consistent directions
- B. It guarantees the global minimum is found
- C. It removes the need for a loss function
- D. It sets the learning rate to zero at the end of training

<details><summary>Answer</summary>

**A.** Momentum keeps an exponential moving average of past gradients, which damps oscillation across a narrow valley.

Why each option:

- A ✓ Correct: smoothing and acceleration.
- B ✗ Nothing guarantees the global minimum on non-convex losses.
- C ✗ Momentum modifies how the loss gradient is used; it still needs a loss.
- D ✗ That's a learning-rate schedule.

In depth: Momentum keeps a velocity v ← βv + ∇L and updates w ← w − ηv. It's an exponential moving average of gradients. In a narrow canyon the gradient flips sign across the walls, so those components cancel, while the consistent component along the valley floor accumulates, so it speeds up. Like a heavy ball rolling, it can also roll over small bumps (shallow local minima). β ≈ 0.9 is typical.

</details>

**Q80. [Week 3: Adam]**

What is the main difference between plain SGD and Adam?

- A. Adam combines momentum (moving average of gradients) with an adaptive per-parameter learning rate (moving average of squared gradients, as in RMSProp)
- B. Adam does not use gradients
- C. Adam uses a fixed learning rate for every parameter with no momentum
- D. Adam only works for linear regression

<details><summary>Answer</summary>

**A.** Adam has two momentum parameters (β₁ for the first moment, β₂ for the second moment, typically 0.9 and 0.999). It is the most used optimiser.

Why each option:

- A ✓ Correct: momentum plus adaptive per-parameter scaling.
- B ✗ Adam is gradient-based.
- C ✗ That's plain SGD.
- D ✗ Adam is the default optimiser for deep networks.

In depth: Adam keeps two EMAs per parameter: m (first moment, the mean of gradients: momentum, β₁ ≈ 0.9) and v (second moment, the mean of squared gradients, β₂ ≈ 0.999). The update is w ← w − η·m̂/(√v̂ + ε), with bias-corrected m̂ and v̂. Dividing by √v gives each parameter its own effective step size: large where gradients are small and steady, small where they're large or noisy. Plain SGD uses one global learning rate. Adam inherits RMSProp's fix for AdaGrad's shrinking steps.

</details>

**Q81. [Week 3: Hyperparameters]**

Which definition is correct?

- A. An epoch is one full pass over the training dataset
- B. Batch size is the number of layers in the network
- C. The learning rate is the number of epochs
- D. An epoch is one parameter update

<details><summary>Answer</summary>

**A.** Batch size is the number of samples per update. The learning rate is the step size. An epoch is one pass through all training samples.

Why each option:

- A ✓ Correct: one full pass over the training data.
- B ✗ Batch size is samples per update, not layers.
- C ✗ The learning rate is the step size, not a count.
- D ✗ One update is an iteration, not an epoch.

In depth: Epoch: one complete pass over the training set. Batch size: the number of samples used to compute one gradient update. Iteration (step): one update. With N = 10,000 samples and batch size 100, one epoch is 100 iterations. The learning rate is the step size multiplier η. Training typically runs many epochs, and validation loss per epoch is monitored for early stopping.

</details>

**Q82. [Week 3: Learning rate]**

What happens when the learning rate is too small or too large?

- A. Too small: training takes many steps; too large: updates overshoot and the loss may oscillate or diverge
- B. Too small: the model overfits immediately; too large: training is perfectly stable
- C. The learning rate has no effect on training
- D. Too large always finds the global minimum faster

<details><summary>Answer</summary>

**A.** A common strategy is a large learning rate at the start to get close quickly, then decrease it to settle in the minimum.

Why each option:

- A ✓ Correct: slow vs overshooting or divergence.
- B ✗ A small learning rate makes training slow, not overfitting.
- C ✗ The learning rate is one of the most important hyperparameters.
- D ✗ Too large often diverges.

In depth: η scales each step. Too small: progress is tiny, training needs very many epochs, and it can get stuck on plateaus. Too large: steps jump over the minimum, the loss oscillates or explodes to NaN. A common recipe: start relatively large to approach the solution quickly, then decay it (schedules, warm restarts) to settle precisely. Adaptive optimisers such as Adam reduce the sensitivity but don't remove it.

</details>

**Q83. [Week 3: Neuron]**

What is the difference between a linear classifier and an artificial neuron?

- A. A neuron is a linear classifier (w·x + b) followed by a non-linear activation function
- B. A neuron has no weights
- C. A linear classifier uses backpropagation while a neuron does not
- D. There is no relationship between them

<details><summary>Answer</summary>

**A.** One neuron gives one linear boundary. The activation gives a smoother transition and lets stacked layers represent non-linear functions.

Why each option:

- A ✓ Correct: linear score plus activation.
- B ✗ Neurons are defined by their weights.
- C ✗ Both are trained with gradients.
- D ✗ A neuron contains a linear classifier.

In depth: A neuron computes a = f(w·x + b). The w·x + b part is exactly a linear classifier (a hyperplane), and the activation f (sigmoid, tanh, ReLU) adds a non-linearity that gives a smooth transition across the boundary. One neuron still gives one boundary; the power comes from stacking many neurons in layers, which requires the non-linearity so the layers don't collapse into one linear map.

</details>

**Q84. [Week 3: Activation functions]**

What happens to a multi-layer neural network with no activation functions?

- A. It collapses to a single linear transformation, no more expressive than one linear layer
- B. It becomes a universal approximator
- C. It can only output zeros
- D. It trains faster and represents more complex functions

<details><summary>Answer</summary>

**A.** A product of matrices is a matrix. Non-linear activations make each layer a non-linear transformation.

Why each option:

- A ✓ Correct: it collapses to a single linear map.
- B ✗ Universal approximation needs non-linear activations.
- C ✗ It outputs a linear function, not zeros.
- D ✗ It can represent only linear functions.

In depth: Without activations, layer outputs are W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) = Wx + b. Any depth of purely linear layers equals a single linear layer, so boundaries stay hyperplanes and XOR-like problems can't be solved. Activations make each layer a non-linear geometric transformation (bend, fold, squash) that can make classes linearly separable for later layers.

</details>

**Q85. [Week 3: Activation functions]**

Which three are common activation functions for neural networks?

- A. ReLU, sigmoid, tanh
- B. Softmax, MSE, cross-entropy
- C. Adam, SGD, RMSProp
- D. Dropout, batch norm, pooling

<details><summary>Answer</summary>

**A.** ReLU, sigmoid and tanh are classic activations. The other options are losses, optimisers, and layers.

Why each option:

- A ✓ Correct: ReLU, sigmoid and tanh.
- B ✗ Softmax is an output normaliser; MSE and CE are losses.
- C ✗ These are optimisers.
- D ✗ These are layer types for regularisation, normalisation and downsampling.

In depth: Sigmoid σ(x) = 1/(1+e⁻ˣ) squashes to (0, 1) and saturates at both ends. Tanh squashes to (−1, 1), is zero-centred, and also saturates. ReLU, max(0, x), is cheap, doesn't saturate for x > 0 (it helps with vanishing gradients), and is the default in deep nets. Variants include Leaky ReLU, GELU and SiLU. Softmax is technically an output activation, but MSE and cross-entropy are losses.

</details>

**Q86. [Week 3: Vanishing gradient]**

Why can sigmoid activations cause vanishing gradients in deep networks?

- A. The sigmoid saturates for large |x|, giving near-zero derivatives that shrink further when multiplied through many layers
- B. The sigmoid has a derivative of exactly 1 everywhere
- C. Sigmoid outputs negative values only
- D. Sigmoid cannot be differentiated

<details><summary>Answer</summary>

**A.** Backpropagation multiplies derivatives through the chain rule. Many small factors make the gradient vanish for early layers.

Why each option:

- A ✓ Correct: saturation gives tiny derivatives that multiply through layers.
- B ✗ Its derivative is at most 0.25.
- C ✗ Sigmoid outputs are in (0, 1), always positive.
- D ✗ Sigmoid is smooth and differentiable everywhere.

In depth: σ'(x) = σ(x)(1 − σ(x)) has a maximum of 0.25 at x = 0 and approaches 0 for large |x| (saturation). Backpropagation multiplies one such derivative per layer, so with 10 layers the factor can be ≤ 0.25¹⁰ ≈ 10⁻⁶. Early layers receive almost no gradient and stop learning. Remedies: ReLU-type activations, careful initialisation, normalisation layers (batch norm), and residual connections (ResNet).

</details>

**Q87. [Week 3: Backpropagation]**

What is backpropagation?

- A. An application of the chain rule to compute the gradient of the loss with respect to every weight, layer by layer from output to input
- B. The forward computation of the network's prediction
- C. A method to initialise weights
- D. A data augmentation technique

<details><summary>Answer</summary>

**A.** The forward pass computes outputs and the loss. The backward pass propagates gradients back through the layers with the chain rule.

Why each option:

- A ✓ Correct: the chain rule applied layer by layer, backwards.
- B ✗ That's the forward pass.
- C ✗ Initialisation happens before training.
- D ✗ Augmentation modifies data, not gradients.

In depth: The loss depends on early weights through every later layer. The chain rule gives ∂L/∂W₁ = ∂L/∂a_L · ∂a_L/∂a_(L−1) · … · ∂a₁/∂W₁. Backpropagation computes these products efficiently: the forward pass stores the activations, then the backward pass reuses the gradients from later layers to compute earlier ones, at roughly the cost of one forward pass. PyTorch autograd does this automatically with loss.backward().

</details>

**Q88. [Week 3: Depth and width]**

In neural network terminology, what are network depth and layer width?

- A. Depth = number of layers; width = number of neurons (units) in a layer
- B. Depth = number of neurons; width = number of layers
- C. Depth = batch size; width = learning rate
- D. Depth = number of epochs; width = number of classes

<details><summary>Answer</summary>

**A.** Increasing either increases capacity: more parameters and more complex functions, but also more risk of overfitting.

Why each option:

- A ✓ Correct: depth = layers, width = units per layer.
- B ✗ Swapped.
- C ✗ Batch size and learning rate are training hyperparameters.
- D ✗ Epochs and classes are unrelated to architecture shape.

In depth: Depth is how many layers of transformation are stacked; width is how many units each layer has. Both increase capacity. One wide hidden layer is already a universal approximator, but deeper networks can represent many functions with far fewer parameters by composing features hierarchically. Deeper networks are harder to optimise (vanishing gradients), which is why ResNets and normalisation matter.

</details>

**Q89. [Week 3: Overfitting]**

Which set of techniques helps mitigate or identify overfitting in neural networks?

- A. Regularisation, dropout, and early stopping based on the validation loss
- B. Increasing the number of parameters and training longer
- C. Removing the validation set
- D. Using the test set to choose the number of epochs

<details><summary>Answer</summary>

**A.** More parameters bring more capacity to overfit. Watch the validation loss (early stopping) and regularise (weight penalty, dropout).

Why each option:

- A ✓ Correct: regularisation, dropout and early stopping.
- B ✗ That increases overfitting.
- C ✗ Without validation you can't detect overfitting.
- D ✗ Tuning on the test set biases the final estimate.

In depth: Identify overfitting by plotting training and validation loss per epoch: when validation loss starts rising while training loss keeps falling, the model is memorising. Early stopping keeps the weights from the best validation epoch. Regularisation (an L2 weight-decay penalty) discourages large weights. Dropout randomly zeroes activations so the network can't rely on any single unit. Data augmentation and more data also help.

</details>

**Q90. [Week 3: EMA]**

Why is an exponential moving average (EMA) attractive compared with a standard moving average?

- A. It needs to memorise only the previous averaged value (yₜ = αyₜ₋₁ + (1−α)xₜ) instead of a window of 2N+1 values
- B. It has no hyperparameter
- C. It gives equal weight to all past values
- D. It can only be used on images

<details><summary>Answer</summary>

**A.** EMA is a recursive filter with very low memory. It is the core of momentum and Adam, and of EMA teachers (e.g. DINO).

Why each option:

- A ✓ Correct: one stored value, recursive update.
- B ✗ α is a hyperparameter.
- C ✗ Weights decay exponentially, not equally.
- D ✗ It works on any signal, including gradients.

In depth: A centred moving average over a window of 2N+1 samples must store all those values. EMA uses a recursion: yₜ = αyₜ₋₁ + (1 − α)xₜ. Only the previous output is stored, yet it effectively averages the whole past with exponentially decaying weights (αᵏ for a lag of k). α close to 1 means long memory and strong smoothing. This filter is the core of momentum, Adam's moment estimates, batch norm running statistics, and EMA teacher networks.

</details>

**Q91. [Week 3: Last layer]**

What is the relationship between the last layer of a neural network and a linear classifier?

- A. The last layer is a linear classifier operating on the features produced by earlier layers, which transform the space to make classes linearly separable
- B. The last layer is always a convolution
- C. The last layer performs k-means clustering
- D. They are unrelated

<details><summary>Answer</summary>

**A.** Each layer maps one feature space into another. Ideally the final space is linearly separable, so a linear classifier suffices.

Why each option:

- A ✓ Correct: a linear classifier on learned features.
- B ✗ The last layer of a classifier is usually fully connected.
- C ✗ No clustering is involved.
- D ✗ They're the same mechanism.

In depth: Think of a deep network as a feature extractor followed by a classifier. All hidden layers transform the input geometrically so that, in the final feature space, classes are (nearly) linearly separable. The last layer, a linear map plus softmax, is then just a linear classifier on those features. That's why transfer learning can replace only the last layer for a new task while reusing the features.

</details>

**Q92. [Week 3: Initialisation]**

How are the weights W and b of a neural network typically initialised?

- A. Randomly drawn from a uniform or normal distribution (small values)
- B. All set to exactly zero
- C. All set to one
- D. Copied from the input data

<details><summary>Answer</summary>

**A.** Random initialisation breaks symmetry, so neurons learn different features. With all-equal weights every neuron would get the same gradient.

Why each option:

- A ✓ Correct: small random values.
- B ✗ All zeros causes the symmetry problem.
- C ✗ All ones also causes the symmetry problem and saturation.
- D ✗ Weights aren't copied from data.

In depth: If all weights started equal, every neuron in a layer would compute the same output and get the same gradient, so they'd stay identical forever (the symmetry problem). Random initialisation from a uniform or normal distribution breaks the symmetry. The scale matters: too large saturates activations or explodes; too small vanishes. Schemes like Xavier/Glorot and He/Kaiming set the variance based on the number of inputs. Biases are often initialised to 0.

</details>

<a id="week-4"></a>

## Week 4: CNNs and Computer Vision

**Q93. [Week 4: CV tasks]**

What is the difference between image classification, semantic segmentation and object detection?

- A. Classification: one label per image; semantic segmentation: a class for every pixel; detection: classify and coarsely localise (bounding box) each object
- B. Classification: a class for each pixel; segmentation: one label per image; detection: image denoising
- C. All three produce one label per image
- D. Detection outputs a class per pixel, segmentation outputs bounding boxes

<details><summary>Answer</summary>

**A.** Instance segmentation also classifies and precisely localises (pixel mask) each object instance.

Why each option:

- A ✓ Correct: one label vs per-pixel vs boxes.
- B ✗ Classification and segmentation definitions swapped.
- C ✗ Only classification gives one label per image.
- D ✗ Detection gives boxes; segmentation gives per-pixel classes.

In depth: Classification assumes one dominant object per image and outputs one label (99% cat). Semantic segmentation assumes no notion of objects, only pixels: every pixel gets a class, but two adjacent cats merge into one 'cat' region. Object detection assumes 0 or more objects and outputs a class plus a bounding box for each. Instance segmentation combines both: per-object, per-pixel masks that separate the two cats.

</details>

**Q94. [Week 4: CV assumptions]**

Which is a major assumption made when using datasets to develop computer vision models?

- A. The dataset distribution matches the distribution of the data encountered in the end application
- B. Images in the test set are identical to training images
- C. Every image contains every class
- D. Labels are allowed to be random

<details><summary>Answer</summary>

**A.** Other assumptions include no ambiguity in the ground truth. Both are often violated in practice, so a reported 87% accuracy may not hold.

Why each option:

- A ✓ Correct: the dataset matches deployment data.
- B ✗ Test images must be unseen, not identical.
- C ✗ Most images contain one or a few classes.
- D ✗ Labels must be meaningful for learning.

In depth: Benchmarks like ImageNet drove progress, but a benchmark score assumes (1) the dataset distribution matches the application's data, and (2) the ground truth is unambiguous for every image, object or pixel. In practice deployment images differ (lighting, devices, populations), and many labels are ambiguous (is it a dog or a wolf? where exactly is the tumour boundary?). So an 87% benchmark accuracy may not mean 87% in the real world.

</details>

**Q95. [Week 4: Conv vs FC]**

Why is a convolutional layer more useful than a fully-connected layer for images?

- A. It exploits local spatial structure and shares weights across locations, so it has far fewer parameters and detects a pattern anywhere
- B. It has more parameters, so it always learns more
- C. It ignores the order of pixels
- D. It does not need training

<details><summary>Answer</summary>

**A.** Flattening an image into an MLP loses spatial structure (shuffled pixels give the same result) and needs huge weight matrices.

Why each option:

- A ✓ Correct: local structure plus weight sharing.
- B ✗ Conv layers have far fewer parameters.
- C ✗ That's what FC layers do; convolution respects order.
- D ✗ All layers need training.

In depth: A fully connected layer on a 224×224×3 image needs 150,528 weights per neuron and treats pixels as an unordered list: shuffling all pixels the same way gives identical results. A convolution slides a small filter (e.g. 3×3×3 = 27 weights) over the image. Local connectivity exploits the fact that neighbouring pixels are correlated. Weight sharing means one filter detects its pattern (a 'nose') anywhere, with far fewer parameters and better generalisation.

</details>

**Q96. [Week 4: Invariance]**

What type of invariance does a convolutional layer introduce?

- A. Translation invariance: a filter detects the same feature wherever it appears
- B. Rotation invariance
- C. Scale (size) invariance
- D. Colour invariance

<details><summary>Answer</summary>

**A.** Convolutions are translation invariant (strictly, equivariant), but not rotation or size invariant. That's one reason data augmentation helps.

Why each option:

- A ✓ Correct: translation invariance.
- B ✗ Rotation must be learned, e.g. through augmentation.
- C ✗ Scale must be learned or handled with multiscale designs.
- D ✗ Colour changes alter the filter responses.

In depth: Shifting the input shifts the activation map by the same amount (translation equivariance). After pooling or global aggregation the output becomes approximately translation invariant: a cat in the corner or the centre is still detected. Convolutions are not invariant to rotation or scale, because a filter tuned to an upright nose of a certain size won't match a rotated or much larger one. Data augmentation (rotations, scaling) and multiscale architectures compensate.

</details>

**Q97. [Week 4: Conv output size]**

What is the output spatial size of a convolution with input size N, filter size F, padding P and stride S?

- A. (N − F + 2P)/S + 1
- B. (N + F − P)·S
- C. N/S + F
- D. (N − 2P)/F + S

<details><summary>Answer</summary>

**A.** For example, a 7×7 input with a 3×3 filter, no padding, stride 1 gives 5×5. With stride 2 it gives 3×3.

Why each option:

- A ✓ Correct: (N − F + 2P)/S + 1.
- B ✗ Not a valid size formula.
- C ✗ Not a valid size formula.
- D ✗ Not a valid size formula.

In depth: Pad the input by P on each side, giving length N + 2P. A filter of size F has N + 2P − F + 1 valid positions at stride 1, and with stride S we take every S-th position: (N − F + 2P)/S + 1. Worked examples: 7×7, F=3, P=0, S=1 gives 5×5; S=2 gives 3×3. With F=3 and P=1 at S=1 the size is preserved ('same' padding). Memorise this formula; it appears in most CNN exam questions.

</details>

**Q98. [Week 4: Conv output size]**

A 7×7 input is convolved with a 3×3 filter, no padding, stride 3. What happens?

- A. (7 − 3)/3 + 1 is not an integer, so the filter does not fit evenly; this stride does not work without padding
- B. The output is 3×3
- C. The output is 5×5
- D. The output is 7×7

<details><summary>Answer</summary>

**A.** (7−3)/3 + 1 = 2.33. Choose hyperparameters so the formula gives an integer, e.g. by adding zero padding.

Why each option:

- A ✓ Correct: not an integer, so it doesn't fit evenly.
- B ✗ 3×3 is the result for stride 2.
- C ✗ 5×5 is the result for stride 1.
- D ✗ 7×7 would need padding 1 and stride 1.

In depth: (7 − 3 + 0)/3 + 1 = 4/3 + 1 ≈ 2.33. A non-integer means the filter positions don't tile the input evenly: the last stride would run off the edge. Frameworks like PyTorch silently floor it (giving 2×2 and ignoring the remaining pixels), but by design you should choose N, F, P and S so the result is an integer, for example by adding zero padding.

</details>

**Q99. [Week 4: Conv parameters]**

A conv layer applies 6 filters of size 5×5 to a 32×32×3 RGB image (with bias). How many learnable parameters does it have?

- A. 456
- B. 150
- C. 450
- D. 18,432

<details><summary>Answer</summary>

**A.** Each filter has 5·5·3 = 75 weights + 1 bias = 76, and 76 × 6 = 456. The filter depth always matches the input channels.

Why each option:

- A ✓ Correct: (5·5·3 + 1) × 6 = 456.
- B ✗ Forgets the 3 input channels and the biases: 5·5·6.
- C ✗ Forgets the biases: 75 × 6.
- D ✗ That's an output-size-style number, not parameters.

In depth: Each filter must cover all input channels: 5 × 5 × 3 = 75 weights, plus 1 bias = 76 parameters per filter. Six filters: 6 × 76 = 456. The output is 6 activation maps of size (32 − 5)/1 + 1 = 28, i.e. 28×28×6 with no padding. The parameter count is independent of the image size (32×32), which is the payoff of weight sharing.

</details>

**Q100. [Week 4: Tensor shape]**

A batch of 100 RGB images of 224×224 pixels is stored in PyTorch as which tensor shape?

- A. [100, 3, 224, 224]
- B. [224, 224, 3, 100]
- C. [3, 100, 224, 224]
- D. [100, 224, 3]

<details><summary>Answer</summary>

**A.** PyTorch uses B×C×H×W: batch, channels, height, width.

Why each option:

- A ✓ Correct: B×C×H×W.
- B ✗ That's channels-last with the batch at the end.
- C ✗ Batch must come first.
- D ✗ Missing a spatial dimension.

In depth: PyTorch convolution layers expect tensors in N×C×H×W order (batch, channels, height, width). So 100 RGB images of 224×224 are [100, 3, 224, 224]. Libraries like PIL or matplotlib use H×W×C for a single image, which is why transforms such as ToTensor() permute the axes. Getting this order wrong is a common source of shape errors.

</details>

**Q101. [Week 4: Pooling]**

Why are pooling layers used in CNNs, and how many learnable parameters do they have?

- A. To reduce the spatial dimension of activation maps (downsampling); they have no learnable parameters
- B. To add non-linearity; they have one weight per pixel
- C. To normalise the batch; they learn γ and β
- D. To increase the image resolution; they have many parameters

<details><summary>Answer</summary>

**A.** Pooling has 3 hyperparameters (type such as max or average, filter size, stride) but no parameters. Strided convolutions can also downsample.

Why each option:

- A ✓ Correct: downsampling with no parameters.
- B ✗ ReLU adds non-linearity; pooling has no weights.
- C ✗ That's batch norm.
- D ✗ Pooling reduces resolution (upsampling is transpose convolution).

In depth: Max pooling takes the maximum in each window (e.g. 2×2 with stride 2), halving H and W, so it keeps the strongest activations and adds a little translation robustness. Average pooling takes the mean. Pooling reduces computation and memory and enlarges the receptive field of later layers. It has hyperparameters (type, window size, stride) but no learnable parameters. A strided convolution is a learnable alternative for downsampling.

</details>

**Q102. [Week 4: Learned features]**

What do convolutional filters typically learn at different depths of a CNN?

- A. Early filters learn simple features (edges, contrast patterns); deeper filters combine them into complex parts and objects
- B. Early filters learn whole objects; deeper filters learn edges
- C. All filters learn the same thing
- D. Filters only learn colours

<details><summary>Answer</summary>

**A.** Stacking convolutions grows the receptive field, so deeper layers 'see' larger regions and more abstract patterns.

Why each option:

- A ✓ Correct: simple to complex with depth.
- B ✗ Reversed.
- C ✗ Filters specialise differently.
- D ✗ They learn spatial patterns, not just colours.

In depth: Visualising trained CNNs shows a hierarchy. First-layer filters resemble edge and colour-contrast detectors (Gabor-like). Middle layers respond to textures and simple parts (eyes, wheels). Deep layers respond to whole object parts or objects, because each layer's receptive field covers a larger area of the input. Deep Dream and feature visualisation optimise an input image to maximise a filter, revealing what it 'looks for'.

</details>

**Q103. [Week 4: Generalisation]**

Why is generalisation in computer vision so challenging?

- A. Objects vary in viewpoint, illumination, deformation, occlusion, background clutter and intra-class appearance
- B. Images contain too few pixels
- C. CNNs cannot be trained with gradient descent
- D. Every image has a unique label

<details><summary>Answer</summary>

**A.** Generalising means the features learned during training are also useful on unseen test data with all this variability.

Why each option:

- A ✓ Correct: large appearance variability.
- B ✗ Images have many pixels.
- C ✗ CNNs are trained with gradient descent.
- D ✗ Many images share a label.

In depth: The same object can produce wildly different pixel arrays: viewpoint, scale, deformation (a cat curled vs stretched), occlusion, illumination, background clutter, and variation within a class (chair styles). Meanwhile different classes can look similar. A model generalises when the features it learned on training images (e.g. 1.2M ImageNet images) remain discriminative on new images with all this variability. Augmentation, large diverse datasets and regularisation help.

</details>

**Q104. [Week 4: Dropout vs BatchNorm]**

What is the difference between dropout and batch normalisation layers?

- A. Dropout randomly zeroes activations during training to reduce overfitting; batch norm normalises activations to zero mean / unit variance per batch to speed up and stabilise training
- B. Both randomly remove neurons
- C. Batch norm randomly removes neurons; dropout normalises activations
- D. Neither behaves differently in training and testing

<details><summary>Answer</summary>

**A.** Batch norm speeds up training and makes it more robust to bad initialisation. Both layers behave differently in model.train() and model.eval().

Why each option:

- A ✓ Correct: regularisation vs normalisation.
- B ✗ Only dropout removes activations.
- C ✗ Swapped.
- D ✗ Both behave differently in train vs eval mode.

In depth: Dropout(p): during training each activation is zeroed with probability p (the rest are scaled by 1/(1−p)), which prevents co-adaptation, a bit like training an ensemble. At eval it is off. Batch norm: normalise each channel with the batch mean and variance, then apply learnable γ and β. It stabilises and speeds up training and allows higher learning rates. At eval it uses running averages of the statistics. So one regularises and the other normalises.

</details>

**Q105. [Week 4: Data augmentation]**

What is the role of data augmentation in computer vision?

- A. Apply random transformations (flips, crops, rotations, colour jitter) during training to increase diversity and improve generalisation
- B. Increase the size of the test set
- C. Remove noisy labels
- D. Replace the need for a loss function

<details><summary>Answer</summary>

**A.** Augmentation teaches invariances that convolutions don't provide, like rotation, scale and colour, and reduces overfitting.

Why each option:

- A ✓ Correct: random transformations for diversity and generalisation.
- B ✗ The test set should stay untouched.
- C ✗ Augmentation doesn't fix labels.
- D ✗ The loss is still required.

In depth: Augmentation applies random, label-preserving transformations each time an image is loaded: flips, crops, rotations, scaling, colour jitter, blur. The model effectively sees new variations every epoch, which (1) enlarges the training distribution, (2) teaches invariances that convolutions don't have by design (rotation, scale, colour), and (3) reduces overfitting. It's applied to training data only (torchvision.transforms), and the transformations must not change the true label.

</details>

**Q106. [Week 4: Transfer learning]**

What is transfer learning and how is it practically achieved?

- A. Reuse a model pre-trained on a large dataset (e.g. ImageNet ResNet), replace the final layer to match the new task, then freeze or fine-tune the weights
- B. Train a new model from random weights on the test set
- C. Copy the labels from one dataset to another
- D. Use k-means to transfer cluster centres

<details><summary>Answer</summary>

**A.** Pre-trained models (e.g. PyTorch ResNet) already contain useful generic features, so far less labelled data is needed.

Why each option:

- A ✓ Correct: reuse pre-trained weights, replace the head, freeze or fine-tune.
- B ✗ Never train on the test set.
- C ✗ Transfer learning reuses features, not labels.
- D ✗ k-means is unrelated.

In depth: Features learned on a large dataset (edges, textures, parts) are useful for many vision tasks. Practically: (1) load a pre-trained model (e.g. torchvision ResNet on ImageNet), (2) replace the final classification layer to match your number of classes, (3) either freeze the backbone and train only the new head (little data), or fine-tune all layers with a small learning rate (more data). This gives good accuracy with far less labelled data and compute.

</details>

**Q107. [Week 4: ResNet]**

What innovation did ResNet introduce that allowed much deeper CNNs?

- A. Residual (skip / shortcut) connections that add the block's input to its output, helping gradients flow
- B. Max pooling
- C. The sigmoid activation
- D. Fully connected layers only

<details><summary>Answer</summary>

**A.** Backpropagating through many layers applies the chain rule many times and the gradient vanishes. Skip connections give the gradient a direct path.

Why each option:

- A ✓ Correct: residual skip connections.
- B ✗ Max pooling existed long before.
- C ✗ Sigmoid makes vanishing gradients worse.
- D ✗ ResNet is convolutional.

In depth: A residual block computes y = F(x) + x: the layers only need to learn a residual correction F, and the identity shortcut passes x (and, backwards, the gradient) straight through. Without it, very deep plain networks trained worse than shallower ones because gradients vanish through many multiplications. With skip connections ResNets reached 50, 101 and 152 layers. Skip connections are now everywhere (DenseNet, UNet, transformers).

</details>

**Q108. [Week 4: UNet]**

What is a UNet and why does it work well for segmentation?

- A. A multiscale encoder-decoder CNN with skip connections: downsampling gives a large receptive field, upsampling restores resolution, skips keep fine detail
- B. A recurrent network for text
- C. A single convolution layer with a huge kernel
- D. A decision tree for pixels

<details><summary>Answer</summary>

**A.** Without pooling you would need ~100 layers of 3×3 convs to cover a 224×224 image. A UNet gets a global receptive field efficiently and outputs a per-pixel map.

Why each option:

- A ✓ Correct: a multiscale encoder-decoder with skips.
- B ✗ That's an RNN.
- C ✗ A huge kernel explodes the parameter count.
- D ✗ Not a tree model.

In depth: UNet (Ronneberger et al., 2015) has a contracting encoder (conv plus pooling: more channels, lower resolution, large receptive field to understand context) and an expanding decoder (upsampling or transpose conv back to full resolution). Skip connections concatenate encoder features at each scale into the decoder, so fine spatial detail lost by pooling is recovered. The output is a per-pixel map, ideal for segmentation (medical CT, MRI). It's also the standard denoiser in diffusion models.

</details>

**Q109. [Week 4: Receptive field]**

Why not simply remove all pooling and stack 3×3 convolutions to segment a 224×224 image?

- A. Each 3×3 conv only grows the receptive field by 2 pixels, so ~100+ layers are needed: hard to backpropagate and costly
- B. Convolutions cannot be stacked
- C. The output would be 1×1
- D. 3×3 kernels have no parameters

<details><summary>Answer</summary>

**A.** Bigger kernels (e.g. 11×11) reduce depth but make the parameter count grow quickly. Multiscale (UNet) is the practical answer.

Why each option:

- A ✓ Correct: too many layers needed for a global receptive field.
- B ✗ Convolutions stack fine.
- C ✗ With padding the size is kept.
- D ✗ Each 3×3 kernel has 9·C_in weights.

In depth: Each 3×3 conv (stride 1) grows the receptive field by 2 pixels: after L layers it is 2L + 1. To let every output pixel see the whole 224-pixel image you'd need L ≈ 112 layers, which is hard to backpropagate through and very expensive at full resolution. 11×11 kernels need ~22 layers but have ~13× more weights each. Downsampling grows the receptive field exponentially, and the UNet decoder restores resolution, which solves both problems.

</details>

<a id="week-5"></a>

## Week 5: Ethics and Performance

**Q110. [Week 5: Metrics]**

Which formula is correct?

- A. Precision = TP / (TP + FP)
- B. Recall = TN / (TN + FP)
- C. Specificity = TP / (TP + FN)
- D. Accuracy = TP / (TP + FP)

<details><summary>Answer</summary>

**A.** Precision: of predicted positives, how many are right. Recall = sensitivity = TP/(TP+FN). Specificity = TN/(TN+FP).

Why each option:

- A ✓ Correct.
- B ✗ TN/(TN+FP) is specificity, not recall.
- C ✗ TP/(TP+FN) is sensitivity/recall, not specificity.
- D ✗ TP/(TP+FP) is precision; accuracy needs TN too.

In depth: Confusion matrix for one positive class: TP (sick, predicted sick), FN (sick, predicted healthy), FP (healthy, predicted sick), TN (healthy, predicted healthy). Precision = TP/(TP+FP): of everything flagged positive, the fraction truly positive. Recall = sensitivity = TP/(TP+FN): of all real positives, the fraction found. Specificity = TN/(TN+FP): of all real negatives, the fraction correctly rejected. Accuracy = (TP+TN)/(TP+TN+FP+FN). Memory aid: Precision looks along the Predicted column; Recall and sensitivity look along the Real positive row.

</details>

**Q111. [Week 5: Metrics]**

Which two metrics are the same thing?

- A. Recall and sensitivity
- B. Precision and specificity
- C. Recall and specificity
- D. Accuracy and precision

<details><summary>Answer</summary>

**A.** Recall = sensitivity = true positive rate = TP/(TP+FN): how many of the actual positives were found.

Why each option:

- A ✓ Correct: both are TP/(TP+FN).
- B ✗ Precision uses FP in the denominator; specificity uses the negatives.
- C ✗ Recall is about positives; specificity is about negatives.
- D ✗ Accuracy uses all four cells.

In depth: Recall, sensitivity, true positive rate (TPR) and hit rate are all TP/(TP+FN): how many actual positives the model finds. Specificity = TN/(TN+FP) is the true negative rate, and 1 − specificity is the false positive rate used on the ROC x-axis. Precision = TP/(TP+FP) is also called positive predictive value (PPV). The lecture's question 'are there overlaps between these metrics?' expects recall = sensitivity.

</details>

**Q112. [Week 5: Metrics]**

A model gives TP = 40, FN = 10, FP = 20, TN = 930. What are its precision and recall?

- A. Precision ≈ 0.67, Recall = 0.80
- B. Precision = 0.80, Recall ≈ 0.67
- C. Precision = 0.97, Recall = 0.80
- D. Precision = 0.40, Recall = 0.20

<details><summary>Answer</summary>

**A.** Precision = 40/(40+20) = 0.667. Recall = 40/(40+10) = 0.8. Accuracy would be 970/1000 = 97%, which looks good but hides the errors.

Why each option:

- A ✓ Correct: P = 0.67, R = 0.80.
- B ✗ Swapped.
- C ✗ 0.97 is accuracy, not precision.
- D ✗ Uses wrong denominators.

In depth: Step 1, precision = TP/(TP+FP) = 40/(40+20) = 40/60 = 0.667. Step 2, recall = TP/(TP+FN) = 40/(40+10) = 40/50 = 0.80. Bonus: specificity = TN/(TN+FP) = 930/950 = 0.979, and accuracy = (40+930)/1000 = 0.97. F1 = 2PR/(P+R) = 2(0.667)(0.8)/(1.467) ≈ 0.727. The 97% accuracy hides that one in three positive predictions is a false alarm.

</details>

**Q113. [Week 5: Error types]**

A false positive is also called:

- A. A false alarm / Type I error
- B. A miss detection / Type II error
- C. A true negative
- D. Sensitivity

<details><summary>Answer</summary>

**A.** False positive = false alarm = Type I. False negative = miss detection = Type II. The costs of the two are often very different.

Why each option:

- A ✓ Correct: false alarm / Type I.
- B ✗ That's the false negative.
- C ✗ A true negative is a correct rejection.
- D ✗ Sensitivity is a rate, not an error type.

In depth: False positive: predicting the condition when it is absent, a false alarm. In statistics it's a Type I error (rejecting a true null hypothesis); the lecture describes it as 'overdoing a task'. False negative: missing a real case (miss detection, Type II error, 'not very good at a task'). Example of asymmetric costs: in a pregnancy test, an FP causes stress and an FN delays care; in elderly fall detection, an FN can be fatal.

</details>

**Q114. [Week 5: Operating point]**

For a disease screening test, which type of error should be minimised?

- A. False negatives (high sensitivity), to catch most disease cases
- B. False positives, to reduce false alarms
- C. Neither; maximise accuracy
- D. Both must be exactly zero

<details><summary>Answer</summary>

**A.** Screening aims to catch most cases. Diagnosis (e.g. a ~$2000 PET scan after screening) aims to minimise false positives.

Why each option:

- A ✓ Correct: minimise misses.
- B ✗ That's the goal for diagnosis, not screening.
- C ✗ Accuracy ignores the asymmetric costs.
- D ✗ Impossible in practice; there's always a trade-off.

In depth: Screening tests are cheap and applied to many people. Missing a case (FN) means the person is never followed up, so you choose a threshold with high sensitivity and accept more false alarms that later tests will filter. Diagnostic tests (e.g. a $2,000 PET scan for Alzheimer's after memory screening) must avoid false alarms, so they need high specificity. The operating point on the ROC curve is chosen by these costs, not by accuracy.

</details>

**Q115. [Week 5: ROC]**

How is a ROC curve built?

- A. Vary the decision threshold and plot sensitivity (TPR) against 1 − specificity (FPR) for each threshold
- B. Plot training loss against epochs
- C. Plot precision against the number of classes
- D. Plot predicted confidence against accuracy for each bin

<details><summary>Answer</summary>

**A.** Each threshold gives one operating point. The point closest to the top-left corner is often reported, and the area under the curve summarises performance.

Why each option:

- A ✓ Correct: sweep the threshold, plot TPR vs FPR.
- B ✗ That's a training curve.
- C ✗ That's not a ROC curve.
- D ✗ That's a calibration (reliability) diagram.

In depth: Step by step: (1) the model outputs a continuous score (probability) for each test sample. (2) Choose a threshold t: score ≥ t is positive. (3) Compute TPR = sensitivity and FPR = 1 − specificity for that t. (4) Sweep t from high to low; each t gives a point, from (0,0) at a very strict threshold to (1,1) when everything is positive. The diagonal is random guessing; the top-left corner (0,1) is perfect. AUC is the probability that a random positive scores higher than a random negative.

</details>

**Q116. [Week 5: Prevalence]**

What is the effect of low disease prevalence on the ROC curve?

- A. No effect on the ROC, since sensitivity and specificity are computed within each class; other metrics such as precision are needed
- B. The ROC curve becomes a straight line
- C. The ROC area always increases
- D. The ROC can no longer be computed

<details><summary>Answer</summary>

**A.** ROC is prevalence-independent. Precision (positive predictive value) drops when the disease is rare.

Why each option:

- A ✓ Correct: ROC is prevalence-independent.
- B ✗ Not true.
- C ✗ Not true.
- D ✗ It can still be computed.

In depth: TPR = TP/(TP+FN) uses only positives; FPR = FP/(FP+TN) uses only negatives. Changing how many positives exist relative to negatives doesn't change either rate, so the ROC curve is unchanged. Precision does change. Example: sensitivity = specificity = 98%. At 50% prevalence, PPV = 98%. At 1% prevalence in 10,000 people: TP = 98, FP = 0.02 × 9,900 = 198, so PPV = 98/296 ≈ 33%. That's why the lecture says low prevalence needs other metrics.

</details>

**Q117. [Week 5: Calibration]**

What is a calibration curve?

- A. A plot of the model's predicted confidence (binned) against the actual accuracy in each bin; a calibrated model lies on the diagonal
- B. A plot of the ROC curve for each class
- C. A plot of loss versus learning rate
- D. A histogram of class labels

<details><summary>Answer</summary>

**A.** To build it, test the model, store confidence and correctness, bin by confidence, and compute accuracy per bin. For the 50–60% bin a calibrated model would be ~55% accurate.

Why each option:

- A ✓ Correct: confidence vs accuracy per bin.
- B ✗ That's per-class ROC.
- C ✗ That's a learning-rate finder.
- D ✗ That's a label histogram.

In depth: Procedure: (1) run the model on a test set, and for each prediction store its confidence (max softmax probability) and whether it was correct. (2) Group predictions into confidence bins (0-10%, …, 90-100%). (3) For each bin compute the actual accuracy. (4) Plot accuracy against average confidence. A perfectly calibrated model lies on the diagonal: among predictions made with ~55% confidence, about 55% are correct. Below the diagonal is over-confident; above is under-confident. Expected Calibration Error (ECE) averages the gaps.

</details>

**Q118. [Week 5: Calibration]**

Deep neural classifiers are often described as poorly calibrated. What does this usually mean?

- A. They are over-confident: predicted confidence is higher than the actual accuracy
- B. They are always under-confident
- C. They always output 50% confidence
- D. They cannot output probabilities

<details><summary>Answer</summary>

**A.** On a calibration plot, over-confident models fall below the diagonal. Ensembles often improve calibration.

Why each option:

- A ✓ Correct: over-confident.
- B ✗ The usual problem is the opposite.
- C ✗ Not the case.
- D ✗ Softmax outputs are probabilities.

In depth: Modern deep networks trained with cross-entropy keep increasing their logits to push the training loss towards 0, so their softmax confidences become extreme (often 99%+) even when the accuracy at that confidence is lower. On a calibration plot they sit below the diagonal. Fixes include temperature scaling (divide logits by T > 1, tuned on validation data), label smoothing, and ensembles.

</details>

**Q119. [Week 5: Distribution shift]**

What is the difference between feature shift and class shift?

- A. Feature shift: the appearance of inputs changes (e.g. new camera, weather); class shift: novel classes appear that were not seen during training
- B. Feature shift: new classes appear; class shift: image brightness changes
- C. Both mean the learning rate changes
- D. They only occur during training

<details><summary>Answer</summary>

**A.** Both are safety concerns because the deployment data no longer matches the training distribution, and the model may fail silently.

Why each option:

- A ✓ Correct.
- B ✗ Swapped.
- C ✗ Unrelated to the learning rate.
- D ✗ They occur at deployment.

In depth: Feature (covariate) shift: p(x) changes but the classes are the same. Images look different because of a new camera, weather, lighting or demographics, which is a form of data drift. Class shift (novel classes, open set): a new category appears that the model has never seen, e.g. a fox shown to a cat/dog classifier. Both are safety concerns because the model will still output a prediction, often with high confidence, rather than saying 'I don't know'.

</details>

**Q120. [Week 5: Novel classes]**

Why do deep learning classifiers often assign high confidence to inputs from novel classes?

- A. Softmax must distribute probability among known classes, and a novel input far from the decision boundaries gets a large score for some known class
- B. Because they detect the new class and say so
- C. Because dropout is active at test time
- D. Because the loss is mean squared error

<details><summary>Answer</summary>

**A.** Most classifiers have no 'unknown' mechanism. Class scores grow with distance from the boundary, so out-of-distribution inputs can look very confident.

Why each option:

- A ✓ Correct: softmax has to choose among known classes, and far-away points get extreme scores.
- B ✗ They don't detect novelty by themselves.
- C ✗ Dropout is off at test time and isn't the cause.
- D ✗ Classifiers use cross-entropy.

In depth: A linear last layer gives each class a score proportional to the input's signed distance from that class's boundary. A novel input (a fox) can lie far from the boundaries in feature space, which makes some known class score very large. Softmax then normalises over known classes only, so all probability mass goes to one of them (e.g. 'dog, 97%'). There's no 'none of the above' output. Remedies: out-of-distribution detection, density models p(x), ensembles, and reject options.

</details>

**Q121. [Week 5: Adversarial attack]**

At a high level, how is an adversarial attack computed?

- A. Compute the gradient of the loss with respect to the input image at test time and add a small, imperceptible perturbation that pushes the prediction to a wrong class
- B. Poison the training data with a trigger pattern
- C. Retrain the model with a different learning rate
- D. Remove random pixels from the image

<details><summary>Answer</summary>

**A.** It exploits the backward pass and gradients, but on the input rather than the weights. It finds the shortest vector that moves the point across a decision boundary (e.g. panda → gibbon).

Why each option:

- A ✓ Correct: an input-gradient perturbation at test time.
- B ✗ That's a backdoor attack (training-time poisoning).
- C ✗ Retraining doesn't attack a model.
- D ✗ Random removal isn't optimised.

In depth: Training computes ∂L/∂weights to change the model. An attack computes ∂L/∂x, the gradient with respect to the input image, with the weights fixed. FGSM, for example: x_adv = x + ε·sign(∇ₓL(x, y_true)). This moves every pixel slightly in the direction that most increases the loss, which is the shortest path across the decision boundary. With ε tiny (imperceptible), 'panda 57.7%' becomes 'gibbon 99.3%'. In the real world, stickers on stop signs or printed patterns can work too.

</details>

**Q122. [Week 5: Backdoor attack]**

What is a backdoor attack?

- A. Training data is poisoned with a trigger pattern linked to a target label, so the model behaves normally on clean data but misclassifies whenever the trigger appears
- B. A gradient-based perturbation computed at test time
- C. Stealing a model's weights from a server
- D. Overfitting caused by too many epochs

<details><summary>Answer</summary>

**A.** It is hard to detect because tests on clean data look normal. It's a danger when using public or community datasets and pre-trained models.

Why each option:

- A ✓ Correct: training data poisoned with a trigger.
- B ✗ That's an adversarial attack.
- C ✗ That's model theft.
- D ✗ That's ordinary overfitting.

In depth: An attacker injects a few poisoned training samples: clean images plus a small trigger (e.g. a yellow square) relabelled to a target class. The network learns 'trigger means target' as a shortcut while still learning the real task. On clean test data it looks perfect, so validation doesn't reveal the problem; at deployment anyone who adds the trigger controls the prediction. It's relevant when using public or community datasets, downloaded pre-trained weights, or outsourced training.

</details>

**Q123. [Week 5: Ensembles]**

What is an ensemble, and what benefits does it usually give?

- A. Several different models (e.g. different random initialisations) whose predictions are averaged; often better calibration, accuracy and robustness
- B. One model trained for more epochs
- C. A model with more layers
- D. A single model evaluated on several test sets

<details><summary>Answer</summary>

**A.** Ensemble members must differ, for example through random weights, data subsets, or architectures. Otherwise averaging gains nothing.

Why each option:

- A ✓ Correct: averaging diverse models.
- B ✗ Training longer isn't an ensemble.
- C ✗ Depth isn't an ensemble.
- D ✗ Multiple test sets isn't an ensemble.

In depth: Train M models that differ (different random initialisations, data shuffles, bootstrap samples or architectures) and average their predicted probabilities. Errors that aren't shared partly cancel, so accuracy improves; the averaged probabilities are less extreme, so calibration improves; and disagreement between members signals uncertainty, which helps detect novel classes. The cost is M times the compute. Random forests are ensembles of trees.

</details>

**Q124. [Week 5: Explainable AI]**

Which statement about explainable AI is correct?

- A. Decision trees are explainable because their rules can be read; deep learning models are typically not and need tools like saliency maps
- B. Deep learning is inherently explainable
- C. Decision trees are never explainable
- D. Explainability is irrelevant in safety-critical applications

<details><summary>Answer</summary>

**A.** Explainability matters for justifying predictions (contestability, safety, e.g. mortgage decisions). CNN tools include gradient saliency, occlusion maps, Grad-CAM and feature visualisation.

Why each option:

- A ✓ Correct.
- B ✗ Deep learning is typically a black box.
- C ✗ Trees are the classic explainable model.
- D ✗ It matters most there.

In depth: Explainability means a human can understand why a prediction was made, which matters for contestability (a refused mortgage), safety-critical decisions (medicine) and debugging (turtle classified as cat?). Decision trees are explainable by design (read the rules), though very deep trees become hard to read. CNNs with millions of weights are black boxes, so post-hoc tools are used: gradient saliency maps, occlusion maps, Grad-CAM, and feature visualisation. The lecture also asks whether we always need explanations, since we use ibuprofen without fully knowing how it works.

</details>

**Q125. [Week 5: Saliency maps]**

How is a gradient-based saliency map computed for a CNN?

- A. Take the derivative of the class score with respect to the input image pixels to see which pixels most influence the prediction
- B. Take the derivative of the loss with respect to the learning rate
- C. Average all images of the class
- D. Count the number of filters in each layer

<details><summary>Answer</summary>

**A.** Training differentiates with respect to the weights. Saliency differentiates with respect to the image. Occlusion maps instead mask image regions and measure the change.

Why each option:

- A ✓ Correct: the gradient of the class score with respect to the input.
- B ✗ Not meaningful.
- C ✗ That's an average image.
- D ✗ Not an explanation method.

In depth: Compute ∂s_c/∂x, the derivative of the class score s_c with respect to every input pixel, using backpropagation down to the image instead of stopping at the weights. Large absolute gradients mark pixels where a small change most affects the prediction. Display |gradient| (max over channels) as a heat map. Alternatives: occlusion maps (slide a grey patch and record the score drop) and Grad-CAM (gradient-weighted activation maps of the last conv layer).

</details>

**Q126. [Week 5: train/eval mode]**

Why can model.train() vs model.eval() give drastically different validation accuracy?

- A. Dropout and batch norm behave differently: in eval mode dropout is disabled and batch norm uses running statistics instead of batch statistics
- B. eval() changes the weights
- C. train() removes the last layer
- D. There is never any difference

<details><summary>Answer</summary>

**A.** With Dropout(0.1) in train mode, 10% of activations are set to 0. Testing a single sample in train mode gives batch norm meaningless batch statistics.

Why each option:

- A ✓ Correct.
- B ✗ eval() changes behaviour, not weights.
- C ✗ No layers are removed.
- D ✗ The difference can be drastic.

In depth: model.train() and model.eval() switch layer behaviour. Dropout(0.1) in train mode randomly zeroes 10% of activations; in eval mode it's disabled and all units are used. BatchNorm in train mode normalises with the current batch's mean and variance (and updates running averages); in eval mode it uses the stored running averages. Testing in train mode gives random, batch-dependent outputs. With a single test sample, batch statistics are meaningless and predictions can collapse. Always call model.eval() (and torch.no_grad()) for validation.

</details>

**Q127. [Week 5: Bias case study]**

In the case study, a CNN could predict a patient's race from chest X-rays. Why is this an ethical concern?

- A. The model may use hidden demographic signals, which can introduce bias in clinical predictions that humans cannot easily see or explain
- B. Because chest X-rays cannot be used for ML
- C. Because the model was too accurate at detecting disease
- D. Because CNNs cannot process grey-scale images

<details><summary>Answer</summary>

**A.** Average scans and saliency maps did not clearly reveal why. This shows the need for responsible AI throughout the value chain.

Why each option:

- A ✓ Correct: hidden demographic shortcuts create invisible bias risk.
- B ✗ X-rays are widely used in ML.
- C ✗ Accuracy isn't the concern here.
- D ✗ CNNs handle grey-scale images fine.

In depth: Researchers found CNNs could predict self-reported race from chest X-rays with high accuracy, though radiologists can't, and average images and saliency maps didn't reveal why. The concern: a diagnostic model might also use this hidden signal as a shortcut, so its errors could be distributed unequally across groups (bias) in a way nobody can see or explain. It illustrates why responsible AI principles (fairness, transparency, accountability) must apply along the whole value chain.

</details>

<a id="week-6"></a>

## Week 6: Unsupervised Learning

**Q128. [Week 6: Supervised limits]**

Which is a limitation of supervised learning that unsupervised and self-supervised learning aim to address?

- A. Dependence on large amounts of high-quality, diverse labelled data, which is expensive or ambiguous to obtain
- B. It cannot use neural networks
- C. It cannot do classification
- D. It requires no data at all

<details><summary>Answer</summary>

**A.** Labels are expensive, can be wrong or ambiguous, and supervised models only learn features that discriminate the given labels.

Why each option:

- A ✓ Correct.
- B ✗ Supervised learning commonly uses neural nets.
- C ✗ Classification is its main use.
- D ✗ It requires labelled data.

In depth: The supervised learning limitations slides: (1) dependence on the quantity and diversity of labelled data (11M-parameter models need many examples per class). (2) The model learns only features that discriminate the given labels, which may be shortcuts. (3) Label quality: noisy or inconsistent annotations. (4) Labels are expensive (expert time in medicine) and some data is too ambiguous or complex to label. Unsupervised and self-supervised learning learn structure from the far more abundant unlabelled data.

</details>

**Q129. [Week 6: Learning paradigms]**

Which statement correctly distinguishes unsupervised, self-supervised and semi-supervised learning?

- A. Unsupervised finds patterns without labels; self-supervised creates labels from the data itself (pretext tasks); semi-supervised uses a few labelled plus many unlabelled samples
- B. All three require fully labelled data
- C. Self-supervised uses human labels for every sample
- D. Semi-supervised uses no data

<details><summary>Answer</summary>

**A.** Self-supervised is a form of unsupervised learning where the supervisory signal comes from the data (e.g. predict masked parts or the next token).

Why each option:

- A ✓ Correct.
- B ✗ Only supervised learning needs full labels.
- C ✗ Self-supervised creates its own targets.
- D ✗ It uses both labelled and unlabelled data.

In depth: Unsupervised: no labels at all; discover clusters (k-means, GMM), densities or low-dimensional structure (PCA, t-SNE). Self-supervised: still no human labels, but a supervised-style task is built from the data itself, e.g. predict the masked word, the next token, a missing patch, or whether two augmented views come from the same image. Semi-supervised: a small labelled set plus a large unlabelled set, e.g. pseudo-labelling and consistency regularisation.

</details>

**Q130. [Week 6: k-means]**

What are the steps of k-means clustering?

- A. Pick K random centroids → assign each point to the closest centroid → recompute each centroid as the mean of its points → repeat until stable
- B. Store all data → find K nearest neighbours → majority vote
- C. Fit a Gaussian per class with labels → compute likelihood
- D. Split the data on the best feature → recurse

<details><summary>Answer</summary>

**A.** K is a hyperparameter, not the same k as in kNN. k-means is unsupervised, while kNN needs labels.

Why each option:

- A ✓ Correct.
- B ✗ That's kNN (supervised).
- C ✗ That's a supervised Gaussian classifier.
- D ✗ That's a decision tree.

In depth: Algorithm (Lloyd's): (1) initialise K centroids, e.g. K random data points. (2) Assignment: put each point in the cluster of its nearest centroid (Euclidean distance). (3) Update: move each centroid to the mean of its assigned points. (4) Repeat 2-3 until assignments stop changing. Each step decreases the within-cluster sum of squares Σ‖x − μ_k‖², so it always converges, but only to a local minimum. Run several random restarts or use k-means++ initialisation.

</details>

**Q131. [Week 6: k-means]**

Which is a limitation of k-means?

- A. You must choose K, it assumes roughly spherical clusters, and it can get stuck in a local minimum depending on the initial centroids
- B. It is very slow and complicated to implement
- C. It requires labelled data
- D. It always finds the global optimum

<details><summary>Answer</summary>

**A.** Advantages: simple and efficient. Limitations: choosing K, cluster shape assumptions, and sensitivity to initialisation.

Why each option:

- A ✓ Correct.
- B ✗ It's simple and efficient (those are its advantages).
- C ✗ It's unsupervised.
- D ✗ It only finds local optima.

In depth: Limitations: (1) K must be chosen in advance (elbow method, silhouette score). (2) Using the distance to a single centroid implies roughly spherical, similar-size clusters, so elongated or nested shapes fail. (3) The result depends on initialisation and can be a poor local minimum. (4) It's sensitive to outliers and feature scaling. Advantages: simple, fast, easy to implement and interpret. GMMs relax the spherical assumption with covariances.

</details>

**Q132. [Week 6: GMM]**

What is a Gaussian Mixture Model (GMM)?

- A. A model of the data distribution as a weighted sum of Gaussians, each with its own mean and (co)variance, giving soft cluster membership probabilities
- B. A neural network with Gaussian activations
- C. A decision tree whose leaves are Gaussians
- D. A method that assigns each point to exactly one cluster by distance

<details><summary>Answer</summary>

**A.** Pearson (1894) fitted two Gaussians to Weldon's crab measurements by hand. Today GMMs are optimised with EM.

Why each option:

- A ✓ Correct.
- B ✗ Not a neural network.
- C ✗ Not a tree.
- D ✗ That's k-means (hard assignment).

In depth: A GMM models p(x) = Σ_k π_k N(x | μ_k, Σ_k), with mixing weights π_k summing to 1. Each Gaussian can have its own shape (covariance) and size. Each point gets a responsibility γ_ik = P(cluster k | x_i), a soft membership, rather than a hard label. Historically Pearson (1894) fitted two Gaussians to Weldon's crab measurements to test for two sub-species. GMMs are also used for brain tissue segmentation (grey matter, white matter, CSF intensities).

</details>

**Q133. [Week 6: EM]**

How does Expectation-Maximisation (EM) optimise a GMM?

- A. E-step: compute the probability that each sample belongs to each Gaussian; M-step: update means, variances and weights using those probabilities; repeat
- B. Compute the gradient of the loss and use Adam
- C. Solve the normal equations once
- D. Randomly assign labels and keep the best

<details><summary>Answer</summary>

**A.** EM is like k-means with soft assignments: k-means uses hard assignments to the nearest centroid, GMM-EM uses probabilities.

Why each option:

- A ✓ Correct.
- B ✗ EM isn't gradient descent with Adam.
- C ✗ There's no closed form in one step; EM iterates.
- D ✗ Random labels aren't EM.

In depth: Initialise μ_k, σ_k, π_k. E-step: for each sample compute responsibilities γ_ik = π_k N(x_i|μ_k, σ_k) / Σ_j π_j N(x_i|μ_j, σ_j). M-step: re-estimate with weighted MLE: N_k = Σ_i γ_ik; μ_k = Σ_i γ_ik x_i / N_k; σ²_k = Σ_i γ_ik (x_i − μ_k)² / N_k; π_k = N_k / N. Repeat until the log-likelihood stops increasing (it never decreases). k-means is the limit of EM with hard 0/1 responsibilities and equal spherical variances.

</details>

**Q134. [Week 6: Maximum likelihood]**

For n independent samples xᵢ assumed to come from N(μ, σ²), what are the maximum likelihood estimates?

- A. μ̂ = (1/n)Σxᵢ and σ̂² = (1/n)Σ(xᵢ − μ̂)²
- B. μ̂ = median(xᵢ) and σ̂² = max(xᵢ) − min(xᵢ)
- C. μ̂ = 0 and σ̂² = 1 always
- D. μ̂ = Σxᵢ and σ̂² = Σxᵢ²

<details><summary>Answer</summary>

**A.** Minimise the negative log-likelihood and set derivatives to 0. Note that σ̂² with 1/n is biased when μ̂ is estimated from the data; use 1/(n−1) for small datasets.

Why each option:

- A ✓ Correct: the sample mean and the 1/n variance.
- B ✗ The median and range aren't MLEs for a Gaussian.
- C ✗ Those are a standard normal, not estimates.
- D ✗ Missing the 1/n normalisation.

In depth: Derivation: the likelihood is L(μ, σ) = Πᵢ (1/√(2πσ²)) exp(−(xᵢ−μ)²/(2σ²)). The negative log-likelihood is NLL = (n/2)log(2πσ²) + Σᵢ(xᵢ−μ)²/(2σ²). (1) ∂NLL/∂μ = −Σᵢ(xᵢ−μ)/σ² = 0 gives μ̂ = (1/n)Σᵢxᵢ, the sample mean. (2) ∂NLL/∂σ = n/σ − Σᵢ(xᵢ−μ̂)²/σ³ = 0 gives σ̂² = (1/n)Σᵢ(xᵢ−μ̂)². Since μ̂ was estimated from the same data, σ̂² is biased low; the unbiased version divides by n − 1. If σ is fixed at 1, minimising the NLL over μ is exactly least squares, which links MLE to MSE.

</details>

**Q135. [Week 6: Maximum likelihood]**

What does maximum likelihood estimation seek?

- A. The parameter values that make the observed data most probable under the assumed model
- B. The parameters that maximise the loss
- C. The parameters that make the model as simple as possible
- D. The labels that best fit the parameters

<details><summary>Answer</summary>

**A.** In practice we minimise the negative log-likelihood, which turns a product of probabilities into a sum.

Why each option:

- A ✓ Correct.
- B ✗ We maximise likelihood, not loss.
- C ✗ That's regularisation or Occam's razor.
- D ✗ Labels are the data, not what's estimated.

In depth: Given data x₁…xₙ and a model family p(x|θ), the likelihood L(θ) = Πᵢ p(xᵢ|θ) says how probable the observed data is under θ. MLE picks θ̂ = argmax L(θ). Since products of small numbers underflow and are awkward to differentiate, we minimise the negative log-likelihood −Σᵢ log p(xᵢ|θ). Cross-entropy and MSE losses are both NLLs under categorical and Gaussian models, and the VAE and diffusion models are motivated by maximum likelihood.

</details>

**Q136. [Week 6: Clustering applications]**

Which is an application of unsupervised clustering mentioned in the lecture?

- A. Anomaly detection, recommender systems, and brain tissue segmentation from MRI
- B. Computing the gradient of a loss
- C. Positional encoding in transformers
- D. Choosing the learning rate

<details><summary>Answer</summary>

**A.** Clustering is useful to explore trends in data that cannot be labelled, or is too large to label.

Why each option:

- A ✓ Correct.
- B ✗ That's optimisation.
- C ✗ That's a transformer component.
- D ✗ That's hyperparameter tuning.

In depth: Anomaly detection: points far from every cluster (fraud, faults). Recommender systems: group users or items with similar behaviour. Exploring data that can't be labelled or is too big to label: customer segmentation, document topics. Medical imaging: GMM-based tissue segmentation of MRI into grey matter, white matter and CSF. Clustering also gives initial pseudo-labels for semi-supervised learning.

</details>

**Q137. [Week 6: Pseudo-labelling]**

What is pseudo-labelling in semi-supervised learning, and what is its main limitation?

- A. A model trained on labelled data predicts labels for unlabelled data, and confident predictions are added as training labels; wrong pseudo-labels can reinforce errors (confirmation bias)
- B. Humans relabel the test set; it is too slow
- C. Labels are randomly shuffled; it destroys accuracy
- D. It deletes unlabelled data; it wastes data

<details><summary>Answer</summary>

**A.** A confidence threshold is used, but over-confident wrong predictions still propagate.

Why each option:

- A ✓ Correct.
- B ✗ Not about humans relabelling.
- C ✗ Not random shuffling.
- D ✗ It uses unlabelled data.

In depth: Procedure: (1) train a model on the small labelled set. (2) Predict the unlabelled data. (3) Keep predictions above a confidence threshold as pseudo-labels. (4) Retrain on labelled plus pseudo-labelled data, and iterate. The limitation is confirmation bias: confident but wrong pseudo-labels are learned as truth and reinforced, especially when the initial model is weak or classes are imbalanced. Calibration matters, because deep nets are over-confident.

</details>

**Q138. [Week 6: Consistency regularisation]**

What is consistency regularisation?

- A. Encouraging the model to give the same prediction for different augmented (perturbed) versions of the same unlabelled input
- B. Forcing all weights to have the same value
- C. Using the same learning rate for every layer
- D. Training only on labelled data

<details><summary>Answer</summary>

**A.** The key consideration is choosing augmentations that don't change the true class.

Why each option:

- A ✓ Correct.
- B ✗ Not weight tying.
- C ✗ Unrelated to the learning rate.
- D ✗ It uses unlabelled data.

In depth: Idea: the label of an image shouldn't change if it's flipped, cropped or colour-jittered, so the model's prediction shouldn't either. Add an unsupervised loss on unlabelled data that penalises the difference between predictions on two perturbed versions (e.g. MSE or KL between outputs; FixMatch uses a weak-augmentation pseudo-label for the strong-augmentation prediction). Key considerations: the augmentations must preserve the class, the strength of the perturbation, and the weight of the consistency loss.

</details>

**Q139. [Week 6: Contrastive learning]**

In contrastive learning with data augmentation, how are positive and negative pairs created?

- A. Positives: random augmentations of the same reference image; negatives: other images from the dataset
- B. Positives: images with the same human label; negatives: images with no label
- C. Positives: the test set; negatives: the training set
- D. Positives and negatives are both random noise

<details><summary>Answer</summary>

**A.** A contrastive or margin loss pulls positive pairs together and pushes negatives apart in embedding space. It's often used as pretraining for downstream tasks.

Why each option:

- A ✓ Correct.
- B ✗ Uses no human labels.
- C ✗ Not dataset splits.
- D ✗ Pairs come from real images.

In depth: Pipeline (SimCLR-style): (1) take a reference image, create two random augmentations: a positive pair. (2) Other images in the batch are negatives. (3) Pass everything through an encoder (CNN) plus a projection to get embeddings. (4) A contrastive loss (InfoNCE, a margin loss) increases the similarity of positives and decreases it for negatives. After training, the encoder is a rich feature extractor; add a small head and fine-tune with few labels for downstream tasks.

</details>

**Q140. [Week 6: Contrastive learning]**

Which is a limitation of contrastive learning?

- A. Choosing which augmentations to apply and at what magnitude, and needing many good negative pairs with long training
- B. It requires every image to be labelled
- C. It can only be used with decision trees
- D. It has no hyperparameters

<details><summary>Answer</summary>

**A.** Too-strong augmentations destroy semantic content, and too-weak ones teach nothing. Good negatives are hard to get.

Why each option:

- A ✓ Correct.
- B ✗ It's self-supervised.
- C ✗ It uses neural encoders.
- D ✗ It has many hyperparameters.

In depth: Challenges listed: (1) what augmentations to apply and how strong; too weak teaches nothing and too strong destroys semantics (a crop missing the object). (2) It needs many, and hard, negatives, so large batches or memory banks and long training. (3) 'Negatives' may actually be the same class (false negatives). These motivate non-contrastive methods (BYOL, DINO, JEPA) that avoid explicit negatives.

</details>

**Q141. [Week 6: Self-supervised]**

Which is an example of predictive self-supervised learning?

- A. Masking parts of the input (words or image patches) and training the model to predict them
- B. Training a classifier on human-labelled ImageNet classes
- C. Running k-nearest neighbours on labelled data
- D. Grid search for hyperparameters

<details><summary>Answer</summary>

**A.** Other predictive tasks include next-token prediction (GPT), masked tokens (BERT), masked image patches (MAE), and predicting rotation or ordering.

Why each option:

- A ✓ Correct.
- B ✗ That's supervised learning.
- C ✗ That's supervised kNN.
- D ✗ That's hyperparameter tuning.

In depth: Predictive self-supervision hides part of the data and predicts it. Text: next-token prediction (GPT) and masked-token prediction (BERT). Images: masked autoencoders (MAE) reconstruct masked patches; other tasks predict rotation angle, colourisation or patch order. Video and audio: predict future frames. The model must understand context to predict well, so it learns useful representations. The limitation: pixel-level prediction can waste capacity on irrelevant detail, which is the motivation for JEPA.

</details>

**Q142. [Week 6: Foundation models]**

What makes a model a 'foundation model'?

- A. It is trained on huge amounts of (mostly unlabelled) data, typically with self-supervision, and can be used or adapted for many different downstream tasks
- B. It is the first layer of a CNN
- C. It is trained on a small labelled dataset for one task
- D. It is a model that cannot be fine-tuned

<details><summary>Answer</summary>

**A.** Examples include DINOv2 for vision and LLMs. Advantages are general features and little labelled data needed. Dangers include inherited biases, opacity, and concentration of power.

Why each option:

- A ✓ Correct.
- B ✗ That's a layer, not a model category.
- C ✗ That's a narrow task-specific model.
- D ✗ Adaptability is the defining feature.

In depth: Foundation models are large models pre-trained (usually self-supervised) on massive, broad data, then adapted to many downstream tasks by fine-tuning, linear probing, or prompting. Examples: DINOv2 (vision features usable for classification, segmentation and depth) and LLMs. Advantages: strong general features and very little task-specific labelled data needed. Dangers: they inherit biases from web-scale data, are opaque, are expensive to train (few organisations can), and errors propagate to every downstream use.

</details>

**Q143. [Week 6: Transfer learning]**

A model trained on Dataset A (1000 classes) is reused for Dataset B (10 classes). What is the difference between freezing and fine-tuning?

- A. Freezing: only the new final layer is trained; fine-tuning: all (or many) layers are updated, usually with a small learning rate
- B. Freezing updates all layers; fine-tuning updates none
- C. Both delete the pre-trained weights
- D. Fine-tuning changes the number of input channels

<details><summary>Answer</summary>

**A.** First replace the 1000-class head with a 10-class head, then balance new knowledge against old.

Why each option:

- A ✓ Correct.
- B ✗ Reversed.
- C ✗ Both keep the pre-trained weights.
- D ✗ Input channels are unchanged.

In depth: Step 1: replace the final layer (1000 outputs) with a new one (10 outputs). Its weights are random. Step 2, freezing: set requires_grad = False on the backbone and train only the head. This is fast, needs little data, and keeps the old knowledge. Fine-tuning: train all (or the upper) layers, typically with a smaller learning rate so pre-trained features aren't destroyed. It's better when Dataset B is large or different from A. It's a balance between new and old knowledge.

</details>

**Q144. [Week 6: t-SNE / UMAP]**

Which statement about t-SNE is correct?

- A. It is a non-linear method for visualising high-dimensional data in 2D/3D by matching neighbour similarities, using the KL divergence
- B. It is a linear projection identical to PCA
- C. It is a supervised classifier
- D. It is used to train GANs

<details><summary>Answer</summary>

**A.** t-SNE uses Gaussian neighbourhoods in high dimension and a Student-t in low dimension. UMAP is similar, usually faster, and preserves more global structure.

Why each option:

- A ✓ Correct.
- B ✗ PCA is linear; t-SNE is non-linear.
- C ✗ It's an unsupervised visualisation method.
- D ✗ Unrelated to GANs.

In depth: t-SNE converts high-dimensional distances into neighbour probabilities p_ij (Gaussian kernel), then places points in 2D with probabilities q_ij (heavy-tailed Student-t kernel) and moves them to minimise KL(P‖Q). It preserves local neighbourhoods and reveals clusters, but distances between clusters and cluster sizes aren't meaningful. It's non-parametric (no reusable mapping) and slow. UMAP (Uniform Manifold Approximation and Projection) is similar but faster and preserves more global structure. PCA is linear.

</details>

**Q145. [Week 6: KL divergence]**

Which property of the Kullback–Leibler divergence is true?

- A. It is not symmetric: D_KL(p‖q) ≠ D_KL(q‖p) in general, so it is a divergence, not a distance
- B. It is always negative
- C. It is symmetric and satisfies the triangle inequality
- D. It equals the mean squared error

<details><summary>Answer</summary>

**A.** D_KL(p‖q) = ∫p(x) log(p(x)/q(x)) dx ≥ 0, with equality only when p = q.

Why each option:

- A ✓ Correct: not symmetric.
- B ✗ KL ≥ 0.
- C ✗ Neither symmetric nor a metric.
- D ✗ It compares distributions, not samples.

In depth: D_KL(p‖q) = ∫p(x) log(p(x)/q(x)) dx. The lecture builds it up. Naïve 1, the difference p − q, can cancel. Naïve 2, the ratio p/q, should be 1. Naïve 3, log(p/q), should be 0, weighted by p to focus where the data is. By Gibbs' inequality, KL ≥ 0, with equality only if p = q. It's not symmetric: KL(p‖q) punishes q being small where p is large, while KL(q‖p) does the opposite. So it isn't a distance (no symmetry, no triangle inequality).

</details>

<a id="week-7"></a>

## Week 7: Language Models

**Q146. [Week 7: Time series]**

What is time-series data, and how does it differ from typical computer vision data?

- A. A sequence of data points in successive order over time, where order matters and each value depends on previous ones
- B. A set of independent images with no order
- C. Data with exactly two features
- D. Data that has no temporal structure

<details><summary>Answer</summary>

**A.** Text can be treated as time-series: token T is followed by T+1, T+2… Images have spatial structure instead.

Why each option:

- A ✓ Correct.
- B ✗ That's i.i.d. data without order.
- C ✗ Not a definition of time series.
- D ✗ Temporal structure is the defining feature.

In depth: A time series is x₁, x₂, …, x_T ordered in time, where each value is correlated with previous ones (autocorrelation, trends, seasonality). Shuffling the order destroys information. Unlike a single image classified independently, the model must remember context over time. Text can be treated as a time series of tokens (T, T+1, T+2…), which is why the same sequence models (RNN, LSTM, transformer) apply to language, audio, sensors and finance.

</details>

**Q147. [Week 7: Applications]**

Which is an application of time-series data?

- A. Forecasting stock prices or energy demand
- B. Semantic segmentation of a single photo
- C. Clustering iris flowers
- D. Computing a confusion matrix

<details><summary>Answer</summary>

**A.** Time-series tasks include forecasting (future values) and prediction or classification (e.g. activity recognition from sensor sequences).

Why each option:

- A ✓ Correct.
- B ✗ A single image isn't a time series.
- C ✗ Static tabular data.
- D ✗ An evaluation tool.

In depth: Forecasting predicts future values: stock prices, energy demand, weather, sales. Prediction or classification labels a sequence or each step: activity recognition from accelerometers, ECG arrhythmia detection, speech recognition, machine translation. Segmenting a single photo, clustering flowers and confusion matrices don't involve temporal order.

</details>

**Q148. [Week 7: Tokenisation]**

What is the trade-off between character-level and word-level tokens?

- A. Characters: small vocabulary (e.g. 65) but very long sequences; words/subwords: large vocabulary but shorter sequences
- B. Characters: large vocabulary and short sequences
- C. There is no trade-off
- D. Words always need fewer parameters than characters

<details><summary>Answer</summary>

**A.** Tokenisers like SentencePiece or BPE use subwords as a compromise. Modern LLMs have ~50,000+ tokens.

Why each option:

- A ✓ Correct.
- B ✗ Characters give a small vocabulary.
- C ✗ There is a clear trade-off.
- D ✗ A large vocabulary means large embedding and output matrices.

In depth: Character tokens: vocabulary ≈ 65 (Shakespeare) gives tiny embedding and output layers, no unknown words, but 1,115,000 tokens for the text, so long sequences and costly attention (quadratic in length). Word tokens: sequences are ~5× shorter, but the vocabulary is tens of thousands and rare words are unknown. Subword tokenisers (BPE, SentencePiece) are the compromise: frequent words are single tokens and rare words split into pieces. GPT-style vocabularies are ~50,000 tokens.

</details>

**Q149. [Week 7: Tokenisation]**

In the Shakespeare example, the text has 65 different symbols. Where does this number come from?

- A. 26 letters × 2 (upper and lower case) plus special characters such as space, punctuation and line feed
- B. The number of words in the text
- C. The embedding dimension
- D. The number of attention heads

<details><summary>Answer</summary>

**A.** Using one token per character gives a tiny vocabulary but a 1,115,000-token sequence.

Why each option:

- A ✓ Correct.
- B ✗ That's characters, not words.
- C ✗ The embedding dimension is a design choice.
- D ✗ Heads are an architecture choice.

In depth: Count it: 26 lowercase + 26 uppercase = 52 letters, plus space, newline (\n), and punctuation such as ! $ & ' , - . 3 : ; ?, giving 65 unique characters. Each gets an integer id (0-64) for the lookup table. A model then predicts a distribution over 65 symbols at each step.

</details>

**Q150. [Week 7: N-gram]**

How does a bigram language model generate text?

- A. Given the current token, it samples the next token from probabilities of pairs of tokens learned from (or counted in) the text
- B. It uses self-attention over the whole sequence
- C. It retrieves full sentences from the training set
- D. It uses a convolution over the image of the text

<details><summary>Answer</summary>

**A.** A learned bigram matrix gives 'still gibberish but not random' text. Larger N-grams quickly become intractable.

Why each option:

- A ✓ Correct.
- B ✗ That's a transformer.
- C ✗ That's retrieval.
- D ✗ Text isn't processed as an image here.

In depth: A bigram model stores a V×V matrix of P(next | current), counted from text or learned as weights (in the tutorial, a learned matrix with random initial weights). Generation: start with a token, sample the next from its row, repeat. It only sees one previous token, so output is 'not random but still gibberish'. N-grams with N > 2 condition on more context, but the table grows as V^N, which quickly becomes intractable. That motivates embeddings and neural models.

</details>

**Q151. [Week 7: Bag of words]**

What is a disadvantage of the Bag of Words representation?

- A. It loses the order of words (and sometimes context) and can be a very high-dimensional vector
- B. It is very hard to calculate
- C. It requires a GPU
- D. It cannot count repeated words

<details><summary>Answer</summary>

**A.** BoW is easy to calculate but ignores order. A lookup table of embeddings gives dense vectors with semantic meaning instead.

Why each option:

- A ✓ Correct.
- B ✗ It's very easy to calculate.
- C ✗ No GPU is needed.
- D ✗ Counting repeats is exactly what it does.

In depth: BoW represents a sentence as a vector of word counts over the vocabulary: 'I am very very tired today' sets very = 2. It's easy to compute and handles variable-length text, but (1) it loses order: 'not happy because raining' vs 'happy because not raining' can look the same; (2) the vector is as long as the vocabulary (very high-dimensional and sparse); (3) it has no notion of word similarity. Lookup-table embeddings fix (2) and (3), and positional encoding or recurrence fixes (1).

</details>

**Q152. [Week 7: Embeddings]**

What is a token embedding (lookup table)?

- A. Each token is associated with a learned vector in high dimension, where similar meanings give similar vectors
- B. A one-hot vector with the token count
- C. The position of the token in the sentence
- D. The probability of the next token

<details><summary>Answer</summary>

**A.** Word2Vec is a classic method. Embeddings are ubiquitous in LLMs (e.g. king − man + woman ≈ queen).

Why each option:

- A ✓ Correct.
- B ✗ One-hot vectors don't encode similarity.
- C ✗ That's positional encoding.
- D ✗ That's the model output.

In depth: An embedding table is a learnable matrix E of size V × d. Token id i selects row Eᵢ, a dense d-dimensional vector. During training, tokens used in similar contexts get similar vectors, so geometry encodes meaning: clusters of related words, and directions like king − man + woman ≈ queen. Word2Vec learns them by predicting context words; in LLMs they're learned jointly with the transformer. It's far more compact and meaningful than one-hot or BoW vectors.

</details>

**Q153. [Week 7: RNN]**

How is a recurrent neural network (RNN) different from a feed-forward neural network?

- A. It keeps a hidden state (memory) that is passed from one time step to the next, so it can process sequences of variable length
- B. It has no weights
- C. It processes all tokens in parallel with attention
- D. It can only process images

<details><summary>Answer</summary>

**A.** The same weights are applied at every step. Configurations include many-to-many (translation), many-to-one (sentiment), and one-to-many (generation).

Why each option:

- A ✓ Correct.
- B ✗ It has weights, shared over time.
- C ✗ That's a transformer.
- D ✗ It's mainly for sequences.

In depth: At each step t the RNN computes h_t = f(W_x x_t + W_h h_(t−1) + b) and the output y_t = g(W_y h_t). The hidden state h carries a summary of everything seen so far, and the same weights are reused at every step, so any sequence length works. Configurations: many-to-many (translation, tagging), many-to-one (sentiment), one-to-many (captioning, music generation). Training uses backpropagation through time.

</details>

**Q154. [Week 7: RNN limitations]**

Which is NOT a limitation of RNNs?

- A. They cannot process sequences at all
- B. They are computationally expensive because the recurrence cannot be parallelised
- C. They suffer from vanishing and exploding gradients
- D. They struggle to keep long-range context

<details><summary>Answer</summary>

**A.** RNNs do process sequences. Their limitations are sequential compute, unstable gradients, and short memory (e.g. forgetting what was bought at the start of a long sentence).

Why each option:

- A ✓ NOT a limitation, so this is the answer: RNNs process sequences.
- B ✗ A real limitation.
- C ✗ A real limitation.
- D ✗ A real limitation.

In depth: RNN limitations: (1) sequential computation: h_t needs h_(t−1), so you can't parallelise across time on GPUs, which makes training slow. (2) Vanishing or exploding gradients: backprop through time multiplies by W_h repeatedly. (3) Long-range context is lost; e.g. 'On the weekend, I went to the shops to buy an ice-cream … it was delicious', where the link spans many steps. Processing sequences is the RNN's purpose, not a limitation.

</details>

**Q155. [Week 7: LSTM]**

What regulates the cell state and hidden state in an LSTM?

- A. Gates (forget, input, output) that control what information is removed, added and output
- B. A softmax over the vocabulary
- C. Positional encodings
- D. Max pooling layers

<details><summary>Answer</summary>

**A.** The cell state is an ongoing memory. Gates (sigmoid-controlled) decide what to forget, what to write, and what to expose.

Why each option:

- A ✓ Correct.
- B ✗ Softmax is used at the output layer.
- C ✗ That's for transformers.
- D ✗ Pooling is a CNN component.

In depth: Gates are sigmoid layers outputting values in [0, 1] that multiply information element-wise. Forget gate f_t: how much of the old cell state to keep. Input gate i_t: how much of the new candidate c̃_t (tanh) to write. Output gate o_t: how much of tanh(c_t) to expose as the hidden state h_t. This learned control gives both long-term memory (cell state) and short-term memory (hidden state), hence 'Long Short-Term Memory'.

</details>

**Q156. [Week 7: LSTM limitations]**

What is a remaining limitation of LSTMs?

- A. They are still computationally expensive because they cannot be parallelised over the sequence
- B. They have no memory at all
- C. They cannot be trained with backpropagation
- D. They only work on images

<details><summary>Answer</summary>

**A.** LSTMs improve memory but are still sequential. Transformers remove the recurrence.

Why each option:

- A ✓ Correct.
- B ✗ They have a memory cell.
- C ✗ They're trained with BPTT.
- D ✗ They work on sequences.

In depth: LSTMs fix much of the vanishing gradient problem, but they're still recurrent: step t must wait for t − 1, so they can't use GPU parallelism across the sequence and are slow to train on large corpora. Very long-range dependencies remain hard, and each step does 4 times the computation of an RNN. Transformers replace recurrence with attention, processing all positions in parallel.

</details>

**Q157. [Week 7: Transformers]**

What is the key difference between the Transformer architecture and RNNs/LSTMs?

- A. Transformers have no recurrence: all tokens are processed simultaneously and self-attention captures context across the whole sequence
- B. Transformers process one token at a time with a hidden state
- C. Transformers use convolution only
- D. Transformers cannot be used for text

<details><summary>Answer</summary>

**A.** Parallel processing makes training much faster on GPUs, and attention captures global context in long sequences.

Why each option:

- A ✓ Correct.
- B ✗ That's an RNN.
- C ✗ Transformers use attention plus MLP layers.
- D ✗ They're the basis of LLMs.

In depth: RNNs and LSTMs pass information step by step through a hidden state. A transformer feeds all tokens at once and uses self-attention: every token directly computes a weighted combination of all other tokens. So (1) training parallelises across the sequence (fast on GPUs), and (2) any two positions are connected in one step, capturing global context without information degrading over distance. Order information is injected by positional encoding.

</details>

**Q158. [Week 7: Transformer limitations]**

Which is a limitation of Transformers?

- A. Very data hungry, and attention cost grows quadratically with sequence length (memory and compute)
- B. They cannot be parallelised
- C. They have no learnable parameters
- D. They cannot capture long-range context

<details><summary>Answer</summary>

**A.** Mitigations include the KV cache for generation and linear or windowed attention for long contexts.

Why each option:

- A ✓ Correct.
- B ✗ They're highly parallel.
- C ✗ They have billions of parameters.
- D ✗ Capturing long-range context is their strength.

In depth: (1) Data hungry: with few built-in assumptions (no recurrence, no locality), they need huge datasets to generalise. (2) Self-attention computes a T × T matrix, so memory and compute grow as O(T²) with context length; 128k-token contexts are expensive. (3) Huge compute and energy to train. Mitigations: the KV cache for inference, sliding-window or linear attention, and efficient kernels.

</details>

**Q159. [Week 7: Positional encoding]**

Why is a positional encoding needed in a Transformer?

- A. Self-attention is order-agnostic, so without positions 'I'm not happy because it is raining' and 'I'm happy because it is not raining' would look the same
- B. To normalise the token embeddings
- C. To reduce the vocabulary size
- D. To mask future tokens

<details><summary>Answer</summary>

**A.** A unique (sinusoidal or learned) vector is added to each position's embedding so the order of tokens matters.

Why each option:

- A ✓ Correct.
- B ✗ That's layer norm.
- C ✗ That's tokenisation.
- D ✗ That's the causal mask.

In depth: Self-attention is permutation-equivariant: it treats the input as a set. Shuffle the tokens and each output is the same, just shuffled. 'I'm not happy because it is raining' and 'I'm happy because it is not raining' contain the same tokens, so without position information they'd be indistinguishable. Adding a position vector PE(pos) to each token embedding makes identical tokens at different positions differ. The original transformer uses sinusoids: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d)).

</details>

**Q160. [Week 7: Attention]**

What does the attention mechanism compute?

- A. Softmax(QKᵀ/√d)·V: each token's query is compared with all keys, and the resulting weights mix the values to update the token
- B. The average of all token embeddings, ignoring queries and keys
- C. A convolution over the sequence
- D. The probability of each class directly

<details><summary>Answer</summary>

**A.** Each row of the attention matrix sums to 1. Adding the weighted values 'nudges' each token's embedding according to its context.

Why each option:

- A ✓ Correct.
- B ✗ Attention weights depend on queries and keys.
- C ✗ Not a convolution.
- D ✗ Class probabilities come from the final layer.

In depth: Each token embedding x is projected into a query q = W_q x, key k = W_k x and value v = W_v x. For T tokens stacked: Q, K, V are T × D. Scores QKᵀ (T × T) measure how relevant token j is to token i. Divide by √D (keeps the softmax from saturating), apply softmax row-wise (each row sums to 1), multiply by V: output = softmax(QKᵀ/√D)V. Each token's new representation is a context-weighted mix of values, added back to its embedding, so tokens get 'nudged' towards their meaning in context.

</details>

**Q161. [Week 7: Multi-head]**

With dim_emb = 1024, why use multi-head attention instead of one big attention?

- A. Splitting the dimension into several heads (e.g. 16 × 64) gives smaller matrices and lets each head capture a different type of relationship in parallel; outputs are concatenated
- B. It increases the vocabulary size
- C. It removes the need for positional encoding
- D. It makes the model recurrent

<details><summary>Answer</summary>

**A.** Heads are computed in parallel, concatenated, and passed through a final linear projection.

Why each option:

- A ✓ Correct.
- B ✗ Vocabulary is unrelated.
- C ✗ Positions are still needed.
- D ✗ Still not recurrent.

In depth: One head with D = 1024 creates large Q, K, V matrices and a single attention pattern. Multi-head splits D into h heads of size D/h (e.g. 16 heads × 64). Each head has its own projections and attention, so one head can track syntax and another coreference. Total compute is similar to one full-size head. The h outputs (T × 64 each) are concatenated back to T × 1024 and passed through a final linear projection W_o.

</details>

**Q162. [Week 7: Training LLMs]**

How are decoder-only transformers (like GPT) trained in practice?

- A. Self-supervised next-token prediction on huge text corpora: the target is the input shifted by one token, with a causal mask
- B. Supervised classification with human labels for every sentence
- C. k-means clustering of words
- D. Reinforcement learning from scratch with no text

<details><summary>Answer</summary>

**A.** 'Paris is the capital and largest city of France' gives many training pairs. BERT-style models instead predict masked tokens.

Why each option:

- A ✓ Correct: self-supervised next-token prediction.
- B ✗ Pre-training uses no human labels.
- C ✗ Not clustering.
- D ✗ LLMs are pre-trained on text.

In depth: Take a huge text corpus, tokenise it, and slice windows of length T. The input is tokens 1…T and the target is tokens 2…T+1 (shifted by one). A causal mask stops position t seeing the future, so the model learns T next-token predictions per window with cross-entropy loss. No human labels are needed; the text supervises itself. BERT-style encoders instead mask random tokens and predict them using both sides of the context. Instruction tuning and RLHF come after pre-training.

</details>

**Q163. [Week 7: Output layer]**

What does the final layer of a language transformer do?

- A. A linear classifier over every token in the vocabulary, followed by softmax to give next-token probabilities
- B. Max pooling over the sequence
- C. A k-means assignment of the token
- D. A positional encoding

<details><summary>Answer</summary>

**A.** Embeddings are converted back to token probabilities, and the next token is sampled and appended (autoregressive).

Why each option:

- A ✓ Correct.
- B ✗ Not pooling.
- C ✗ Not clustering.
- D ✗ PE is added at the input.

In depth: After N blocks, each position has a d-dimensional vector. A linear layer (d → V, e.g. V ≈ 50,000) produces one logit per vocabulary token, and softmax turns them into next-token probabilities. Generation picks (greedy) or samples (temperature, top-k) a token, appends it to the input, and repeats (autoregressive). It's literally a linear classifier over the vocabulary.

</details>

**Q164. [Week 7: KV cache]**

What is the purpose of the KV cache during LLM generation?

- A. Store the keys and values of previous tokens so they are not recomputed at every new generated token
- B. Store the training dataset
- C. Cache the gradients for backpropagation
- D. Store the vocabulary

<details><summary>Answer</summary>

**A.** During autoregressive generation only the new token's query needs to be computed against the cached keys and values.

Why each option:

- A ✓ Correct.
- B ✗ Not the dataset.
- C ✗ Gradients aren't computed at inference.
- D ✗ The vocabulary is the embedding table.

In depth: During generation, step t needs attention between the new token's query and the keys and values of all previous tokens. Those K and V vectors don't change once computed, so storing them avoids recomputing T projections per layer at each step. The cost per new token goes from O(T²) to O(T). The price is memory: the cache grows with context length × layers × heads, which is why long-context inference is memory-bound.

</details>

**Q165. [Week 7: Model size]**

According to the lecture, how many transformer layers do GPT-3 and DeepSeek-R1 have?

- A. GPT-3: 96 layers; DeepSeek-R1: 61 layers
- B. GPT-3: 12 layers; DeepSeek-R1: 6 layers
- C. Both have 1 layer
- D. GPT-3: 61 layers; DeepSeek-R1: 96 layers

<details><summary>Answer</summary>

**A.** An LLM stacks N attention blocks (attention + normalisation + feed-forward with residuals).

Why each option:

- A ✓ Correct: 96 and 61.
- B ✗ Far too small.
- C ✗ Both are deep.
- D ✗ Swapped.

In depth: These are facts from the slides: GPT-3 stacks 96 transformer blocks; DeepSeek-R1 has 61. Each block is multi-head attention + residual + layer norm + feed-forward MLP + residual + layer norm. Depth lets the model refine token representations many times. Exams may ask for such numbers, so note them on your one-page summary.

</details>

<a id="week-8"></a>

## Week 8: Autoencoder and VAE

**Q166. [Week 8: Discriminative vs generative]**

What is the difference between discriminative and generative modelling?

- A. Discriminative models learn p(y|x) (the decision); generative models learn the data distribution p(x) or p(x, y) = p(y|x)p(x)
- B. Discriminative models generate images; generative models classify
- C. Both only learn p(y)
- D. Generative models never use neural networks

<details><summary>Answer</summary>

**A.** Knowing p(x) also tells you when a decision is uncertain. High p(red|x) with low p(x) means the input is far from the data.

Why each option:

- A ✓ Correct.
- B ✗ Reversed.
- C ✗ Discriminative models learn p(y|x).
- D ✗ Modern generative models are neural.

In depth: A discriminative model learns p(y|x) directly: the boundary between classes (logistic regression, CNN classifiers). A generative model learns how the data itself is distributed, p(x) or the joint p(x, y). Using p(x, y) = p(y|x)p(x), knowing p(x) adds information: a point can have p(red|x) high (far on the red side of the boundary) while p(x) is tiny (unlike any data), so the decision is actually uncertain. Generative models can also sample new data from p(x).

</details>

**Q167. [Week 8: Manifold hypothesis]**

What is the manifold hypothesis?

- A. Many real-world high-dimensional datasets actually lie along low-dimensional manifolds embedded in the high-dimensional space
- B. All data is uniformly spread in high-dimensional space
- C. Neural networks must be shallow
- D. Images are random noise

<details><summary>Answer</summary>

**A.** A 3×32×32 image has 256^3072 possible values, but real images occupy a tiny, structured subset.

Why each option:

- A ✓ Correct.
- B ✗ Real data is concentrated, not uniform.
- C ✗ Unrelated.
- D ✗ Real images are highly structured.

In depth: Count the space: a 3×32×32 image with 256 levels per pixel has 256³⁰⁷² possible images, almost all of which look like noise. Real images occupy a tiny, structured, low-dimensional subset: a manifold with a few intrinsic degrees of freedom (pose, lighting, shape). The hypothesis justifies dimensionality reduction. PCA assumes the manifold is linear (a flat subspace); autoencoders and VAEs learn curved, non-linear manifolds.

</details>

**Q168. [Week 8: PCA]**

How are the principal components ordered?

- A. By decreasing eigenvalue: the first component captures the most variance
- B. Randomly
- C. By increasing variance
- D. Alphabetically by feature name

<details><summary>Answer</summary>

**A.** PCA computes an orthonormal coordinate system that maximises variance along each successive direction.

Why each option:

- A ✓ Correct.
- B ✗ There's a definite order.
- C ✗ Reversed.
- D ✗ Not meaningful.

In depth: PCA maximises variance step by step: u₁ = argmax over ‖u‖=1 of uᵀCu gives the top eigenvector, with variance λ₁ (the largest eigenvalue); u₂ is the best direction orthogonal to u₁, and so on. Keeping the first d components keeps the most variance possible for any d-dim linear projection; the fraction explained is (λ₁+…+λ_d)/Σλᵢ. That's how you pick d (e.g. 95% of variance).

</details>

**Q169. [Week 8: PCA vs AE]**

How does PCA relate to an autoencoder?

- A. PCA is a linear autoencoder: encode z = Uᵀx, decode x̂ = UUᵀx; an autoencoder generalises this with non-linear encoder and decoder networks
- B. PCA is a GAN
- C. They are unrelated
- D. An autoencoder is a linear version of PCA

<details><summary>Answer</summary>

**A.** Locally, a non-linear autoencoder manifold has a PCA-like tangent space.

Why each option:

- A ✓ Correct.
- B ✗ PCA isn't adversarial.
- C ✗ They're closely related.
- D ✗ Reversed: the AE is the non-linear generalisation.

In depth: Keep the top d eigenvectors as the columns of U (D × d). Encoder: z = Uᵀx (d numbers). Decoder: x̂ = Uz = UUᵀx (the projection back into the subspace). That's a linear autoencoder with tied weights, and a linear AE trained with MSE learns the same subspace. An autoencoder replaces Uᵀ and U with non-linear networks Enc and Dec, so it can follow curved manifolds; locally, its tangent space behaves like PCA.

</details>

**Q170. [Week 8: PCA limits]**

If the data space has 10⁶ dimensions (e.g. images) and we have 50,000 images, what is the largest subspace PCA can compute?

- A. At most 50,000 dimensions
- B. 10⁶ dimensions
- C. Exactly 2 dimensions
- D. 10¹² dimensions

<details><summary>Answer</summary>

**A.** The covariance matrix has rank at most equal to the number of samples, so at most 50,000 meaningful components.

Why each option:

- A ✓ Correct.
- B ✗ Not with only 50,000 samples.
- C ✗ No reason for 2.
- D ✗ Impossible.

In depth: Centred data X is N × D (50,000 × 10⁶). Its covariance has rank ≤ min(N − 1, D) ≈ 50,000. All the data lies in the span of the 50,000 samples, so at most ~50,000 non-zero eigenvalues and meaningful components exist; the remaining directions have zero variance in the sample. In practice you keep far fewer (e.g. a few hundred).

</details>

**Q171. [Week 8: Autoencoder]**

How is an autoencoder trained, and why is it self-supervised?

- A. It minimises the reconstruction error between input x and output x̂ through a bottleneck latent z; the input itself is the target, so no labels are needed
- B. It uses human labels as targets
- C. It maximises the classification accuracy
- D. It is trained with the KL divergence only

<details><summary>Answer</summary>

**A.** The encoder compresses to latent z and the decoder reconstructs. It's self-supervised because the data supervises itself.

Why each option:

- A ✓ Correct.
- B ✗ No labels are used.
- C ✗ Not a classifier.
- D ✗ KL is added in VAEs, not plain AEs.

In depth: Architecture: x goes through Enc to a low-dimensional bottleneck z, then through Dec to x̂. Loss: reconstruction error, e.g. MSE ‖x − x̂‖² (or BCE for binary pixels), minimised by backprop over both networks. The bottleneck forces the network to keep only the most important information: compression, so information is lost. Since the target is the input, any unlabelled dataset works, making it self-supervised. The latent z is a learned non-linear coordinate on the data manifold.

</details>

**Q172. [Week 8: AE variants]**

What is a denoising autoencoder?

- A. An autoencoder whose input is corrupted with noise (x + N(0,1)) and which is trained to reconstruct the clean x
- B. An autoencoder with no decoder
- C. An autoencoder trained to add noise
- D. A GAN discriminator

<details><summary>Answer</summary>

**A.** It learns robust features and to project noisy inputs back onto the data manifold.

Why each option:

- A ✓ Correct.
- B ✗ It has a decoder.
- C ✗ Reversed.
- D ✗ Not a discriminator.

In depth: Training pairs: input x̃ = x + noise (e.g. N(0,1) or masking), target x (clean). The network can't copy its input, so it must learn what clean data looks like and project corrupted points back onto the manifold. It learns robust features and is a practical denoiser. Conceptually it's the ancestor of diffusion models, which learn to denoise at many noise levels.

</details>

**Q173. [Week 8: AE variants]**

What is a sparse autoencoder?

- A. Instead of a small bottleneck, the latent can be as large as (or larger than) the input, but only a few latent units may be active (L1 penalty or top-k)
- B. An autoencoder with very few training samples
- C. An autoencoder with sparse input images
- D. An autoencoder whose weights are all zero

<details><summary>Answer</summary>

**A.** Sparse autoencoders are used for LLM interpretability because a few active features are easier to interpret.

Why each option:

- A ✓ Correct.
- B ✗ Not about dataset size.
- C ✗ The input is dense.
- D ✗ That wouldn't learn anything.

In depth: Instead of compressing through a narrow bottleneck, the latent can be large (e.g. 1×784 for a 1×28×28 MNIST image, the same size as the input). To prevent trivial copying, a sparsity constraint is added: an L1 penalty λ‖z‖₁ on activations, or top-k (keep the k largest activations, zero the rest). Each input is then explained by a few active features, which are easier to interpret. That's why sparse AEs are used to study the features inside LLMs.

</details>

**Q174. [Week 8: AE variants]**

What is a masked autoencoder?

- A. Parts (e.g. patches) of the input are masked and the model learns to reconstruct the missing parts
- B. An autoencoder that masks its weights
- C. An autoencoder with a causal mask over time
- D. An autoencoder trained only on labelled masks

<details><summary>Answer</summary>

**A.** It's a predictive self-supervised pretraining task that learns strong image features.

Why each option:

- A ✓ Correct.
- B ✗ Masks are on the data, not weights.
- C ✗ That's for autoregressive transformers.
- D ✗ No labels are used.

In depth: Randomly hide a large fraction of input patches (MAE masks ~75%), encode only the visible ones, and train the decoder to reconstruct the missing pixels. To fill in the gaps, the model must understand object structure and context, so the encoder learns strong representations for downstream tasks. It's a predictive self-supervised task, analogous to BERT's masked language modelling.

</details>

**Q175. [Week 8: AE as generator]**

Why is a standard autoencoder a poor generative model?

- A. The distribution of the latent z is unknown and irregular, so a randomly chosen z usually decodes into rubbish
- B. The decoder cannot output images
- C. Autoencoders cannot be trained
- D. The latent is always one-dimensional

<details><summary>Answer</summary>

**A.** The next best idea is to force p(z) to be what we want, which is what the VAE does.

Why each option:

- A ✓ Correct.
- B ✗ The decoder outputs images.
- C ✗ AEs train fine.
- D ✗ The latent can be any size.

In depth: The AE only learns to map training images to some codes and back. Nothing constrains how codes are spread: they may sit in scattered islands with gaps, at arbitrary scales. If you sample a random z, it's likely to land in a gap the decoder never saw, giving rubbish output. The VAE fixes this by forcing the latent distribution towards a known prior N(0, I), so sampling from the prior gives decodable codes.

</details>

**Q176. [Week 8: VAE loss]**

What is the loss function of a VAE?

- A. Reconstruction loss (e.g. MSE between x and x̂) + β × KL divergence between q(z|x) and the prior N(0, I)
- B. Only the KL divergence
- C. Binary cross-entropy between real and fake
- D. Hinge loss on the latent

<details><summary>Answer</summary>

**A.** This comes from maximising the Evidence Lower Bound (ELBO) of the log-likelihood.

Why each option:

- A ✓ Correct.
- B ✗ Missing the reconstruction term.
- C ✗ That's the GAN loss.
- D ✗ Hinge isn't used.

In depth: L = E_q[‖x − x̂‖²] + β·D_KL(q(z|x) ‖ N(0, I)). Reconstruction makes decoded samples match the input; KL pulls each encoded distribution N(μ, σ²) towards the standard normal so the latent space is filled smoothly and can be sampled. For Gaussian q and prior, the KL has a closed form per latent dimension: D_KL = ½Σ(μ² + σ² − log σ² − 1). β balances the two terms (β = 1 is the standard ELBO).

</details>

**Q177. [Week 8: Reparameterisation]**

What is the reparameterisation trick in a VAE?

- A. Sample z = μ + σ·ε with ε ~ N(0, 1), so sampling is differentiable with respect to μ and σ
- B. Set z = x directly
- C. Use a softmax over the latent
- D. Sample z uniformly from [0, 1]

<details><summary>Answer</summary>

**A.** Gradients can flow through μ and σ because the randomness is moved into ε.

Why each option:

- A ✓ Correct.
- B ✗ That's an autoencoder with no bottleneck.
- C ✗ Not used.
- D ✗ The prior is Gaussian.

In depth: Sampling z ~ N(μ, σ²) is random, so you can't backprop through 'draw a sample'. Rewrite: ε ~ N(0, 1) (no parameters), z = μ + σ·ε. Now z is a deterministic, differentiable function of μ and σ, with ∂z/∂μ = 1 and ∂z/∂σ = ε. Gradients from the reconstruction loss reach the encoder. Implementation: std = exp(0.5·logvar); z = mu + std * torch.randn_like(std).

</details>

**Q178. [Week 8: VAE implementation]**

In the MNIST VAE implementation, what does the encoder output?

- A. 2 × the latent dimension: μ and ln(σ²) (log-variance rather than σ, for numerical stability)
- B. Only the reconstruction x̂
- C. The class label
- D. A single scalar

<details><summary>Answer</summary>

**A.** Predicting log-variance keeps σ positive and stable.

Why each option:

- A ✓ Correct.
- B ✗ That's the decoder's job.
- C ✗ Not used in a plain VAE.
- D ✗ It needs a vector.

In depth: For a latent of dimension d, the encoder's last layer outputs 2d numbers: d for μ and d for log σ². Why log-variance? σ must be positive, and predicting log σ² lets the network output any real number; σ = exp(0.5·logvar) is always positive and numerically stable. The KL term also uses log σ² directly: −½Σ(1 + logvar − μ² − exp(logvar)).

</details>

**Q179. [Week 8: KL divergence]**

What does the Kullback–Leibler divergence measure?

- A. How one probability distribution differs from another reference distribution (non-negative, zero only if identical, not symmetric)
- B. The Euclidean distance between two vectors
- C. The accuracy of a classifier
- D. The number of parameters in a model

<details><summary>Answer</summary>

**A.** D_KL(p‖q) = ∫ p(x) log(p(x)/q(x)) dx. In the VAE it pushes q(z|x) towards N(0, I).

Why each option:

- A ✓ Correct.
- B ✗ Not a Euclidean distance.
- C ✗ Not accuracy.
- D ✗ Not a parameter count.

In depth: D_KL(p‖q) = ∫p(x) log(p(x)/q(x)) dx = E_p[log p − log q]. It's the expected extra information (in nats or bits) needed when using q to describe data that actually comes from p. Properties: ≥ 0, = 0 iff p = q, not symmetric. In the VAE, p = q_φ(z|x) (the encoder) and q = N(0, I) (the prior). In t-SNE it compares high- and low-dimensional neighbour distributions.

</details>

**Q180. [Week 8: ELBO]**

Why does the VAE maximise the Evidence Lower Bound (ELBO) rather than log p(x) directly?

- A. p(x) = ∫p(x, z)dz is intractable since the manifold z is unknown; since KL ≥ 0, the ELBO is a tractable lower bound on log p(x)
- B. Because log p(x) is always zero
- C. Because the ELBO is the classification accuracy
- D. Because maximum likelihood cannot be used with neural networks

<details><summary>Answer</summary>

**A.** An approximate posterior q_φ(z|x) (the encoder) and a decoder p_θ(x|z) make the bound computable.

Why each option:

- A ✓ Correct.
- B ✗ log p(x) is a real, informative quantity.
- C ✗ Not accuracy.
- D ✗ It is maximum likelihood, made tractable.

In depth: We want max log p(x), but p(x) = ∫p(x|z)p(z)dz integrates over the unknown latent manifold, which is intractable. Introduce the encoder q_φ(z|x). Then log p(x) = ELBO + D_KL(q_φ(z|x) ‖ p(z|x)). Since KL ≥ 0, log p(x) ≥ ELBO, where ELBO = E_q[log p_θ(x|z)] − D_KL(q_φ(z|x) ‖ p(z)). The first term is reconstruction (MSE for a Gaussian decoder), the second the prior-matching term. Maximising the ELBO trains the encoder and decoder together.

</details>

**Q181. [Week 8: β-VAE]**

What happens if the KL term weight β in a VAE is set too high (e.g. ×10)?

- A. The latent fits the prior very well, but reconstructions become poor/blurry; in the extreme the latent ignores the input (posterior collapse)
- B. Reconstructions become perfect and the latent ignores the prior
- C. Nothing changes
- D. Training becomes a GAN

<details><summary>Answer</summary>

**A.** With β too low the model behaves like an autoencoder: good reconstruction but poor fit of the latent to the prior.

Why each option:

- A ✓ Correct.
- B ✗ That describes β too small.
- C ✗ β strongly changes behaviour.
- D ✗ No discriminator is involved.

In depth: β sets the trade-off. β too large (×10): the KL dominates, every q(z|x) collapses towards N(0, I), the latent histograms look perfectly Gaussian, but the codes carry little information about x, so reconstructions are blurry or average (posterior collapse is the extreme case: μ = 0, σ = 1 for all inputs). β too small (÷10): it behaves like a plain autoencoder, with good reconstructions but a latent that doesn't match the prior, so random samples are poor. The lecture also notes the other extreme, σ → 0 ('autoencoder mode').

</details>

**Q182. [Week 8: Conditional VAE]**

What is a conditional VAE?

- A. A VAE where a condition c (e.g. class label) is given to the encoder and decoder, so you can choose what to generate
- B. A VAE that only trains when a condition is met
- C. A VAE without a decoder
- D. A VAE with a causal mask

<details><summary>Answer</summary>

**A.** It can even classify: run the VAE with each of the 10 conditions and pick the one with the lowest reconstruction error.

Why each option:

- A ✓ Correct.
- B ✗ 'Conditional' refers to an input condition.
- C ✗ It keeps both encoder and decoder.
- D ✗ Causal masks are for transformers.

In depth: Concatenate (or embed) a condition c, such as a one-hot class label, to the encoder input and the decoder input: q(z|x, c), p(x|z, c). The latent then only needs to encode the style (stroke thickness, slant), while c encodes the content (which digit). To generate a '7', sample z ~ N(0, I), set c = 7 and decode. It also fixes the multi-manifold ridge problem by giving each class its own Gaussian region. Classification trick: decode with each of the 10 conditions and pick the lowest reconstruction error.

</details>

**Q183. [Week 8: VAE as generator]**

Why can a trained VAE be used as a generative model?

- A. Because its latent is forced towards a known multivariate Gaussian, we can sample z ~ N(0, I) and decode it into a new sample
- B. Because it memorises all training images
- C. Because it uses a discriminator
- D. Because the encoder can be inverted analytically

<details><summary>Answer</summary>

**A.** That's why it works as a generator. Two manifolds (classes) can create a ridge in z, which conditional VAEs address.

Why each option:

- A ✓ Correct.
- B ✗ It doesn't memorise; it decodes new z.
- C ✗ That's a GAN.
- D ✗ Neural encoders aren't invertible analytically.

In depth: The KL term pushes the aggregate latent distribution towards N(0, I), so most of that region decodes into plausible images. Generation: z ~ N(0, I), x = Dec(z), with no encoder needed. You can also interpolate smoothly between codes. Limitations: samples are often blurry, the encoded μ manifold doesn't perfectly cover the prior, and multiple classes can create ridges in z.

</details>

<a id="week-9"></a>

## Week 9: GAN, Diffusion and Flow

**Q184. [Week 9: High dimension]**

For 10-dimensional samples from an isotropic N(0, 1), roughly what is the expected norm?

- A. About √10 ≈ 3.16
- B. About 0
- C. About 10
- D. About 100

<details><summary>Answer</summary>

**A.** The lecture shows all vectors having norms between 1.5 and 4.5, concentrated around √10.

Why each option:

- A ✓ Correct: √10 ≈ 3.16.
- B ✗ The density peaks at 0, but the mass doesn't.
- C ✗ That's ‖z‖², not ‖z‖.
- D ✗ Far too large.

In depth: Compute: E[‖z‖²] = Σ E[zᵢ²] = D × 1 = 10, so ‖z‖ ≈ √10 ≈ 3.16. The lecture's histogram shows nearly all norms between 1.5 and 4.5, with almost nothing near 0 even though the density is highest at the origin. Volume grows so fast with radius that the probability mass sits in a shell. This is the key intuition for latent spaces of VAEs and diffusion models.

</details>

**Q185. [Week 9: Curse of dimensionality]**

What is the curse of dimensionality?

- A. As dimension grows, the volume grows exponentially so data become sparse, distances become similar, and most probability mass sits far from the centre
- B. Neural networks cannot have more than 3 layers
- C. Training becomes faster with more dimensions
- D. High-dimensional data always lies on a line

<details><summary>Answer</summary>

**A.** Fortunately, actual data has structure (manifolds), unlike uniformly random vectors.

Why each option:

- A ✓ Correct.
- B ✗ Unrelated.
- C ✗ The opposite.
- D ✗ Not true.

In depth: Grid example from the lecture: 3 bins per dimension gives 3^D cells. With D = 7 there are 2,187 cells, so each has ~0.05% probability. Data therefore becomes exponentially sparse; most of a hypercube's volume is in its corners; for Gaussians, mass moves to a shell at radius √D; and pairwise distances between random points become nearly equal, so nearest neighbours lose meaning. Learning in high dimensions works only because real data lies on low-dimensional, structured manifolds.

</details>

**Q186. [Week 9: Interpolation]**

Why is linear interpolation between two Gaussian samples problematic in high dimension?

- A. Both endpoints have norm ≈ √D, but the midpoint has a smaller norm, which is very unlikely under the distribution
- B. The midpoint always has a larger norm
- C. Linear interpolation is impossible in high dimension
- D. Interpolated points are always identical to the endpoints

<details><summary>Answer</summary>

**A.** The midpoint falls inside the 'skin' of the hypersphere where there is almost no probability mass. Spherical interpolation stays on the shell.

Why each option:

- A ✓ Correct: the midpoint norm drops to ≈ 0.71√D.
- B ✗ It gets smaller, not larger.
- C ✗ It's possible but misleading.
- D ✗ Intermediate points differ.

In depth: Take x₁, x₂ with norms ≈ √D, nearly orthogonal. The midpoint m = (x₁ + x₂)/2 has ‖m‖² = (‖x₁‖² + ‖x₂‖² + 2x₁·x₂)/4 ≈ (D + D + 0)/4 = D/2, so ‖m‖ ≈ √D/√2 ≈ 0.71√D. That's well inside the shell where almost no samples exist, so decoding it gives atypical (washed-out) results. Spherical interpolation (slerp) follows the arc on the shell and keeps the norm ≈ √D.

</details>

**Q187. [Week 9: Normalisation]**

What does LayerNorm do compared with BatchNorm?

- A. BatchNorm normalises each channel across the batch; LayerNorm normalises the features of each sample independently
- B. LayerNorm normalises across the batch; BatchNorm normalises per sample
- C. Both are identical
- D. Neither has learnable parameters

<details><summary>Answer</summary>

**A.** Both have learnable γ and β. LayerNorm is common in transformers because it doesn't depend on batch size.

Why each option:

- A ✓ Correct.
- B ✗ Swapped.
- C ✗ They normalise along different axes.
- D ✗ Both have γ and β.

In depth: For a tensor [N, C, H, W]: BatchNorm2d computes the mean and variance per channel C over (N, H, W), so it depends on the batch; at test time it uses running averages. LayerNorm computes the mean and variance per sample over its features (C, H, W, or the embedding dimension in transformers), independent of the batch, so it behaves the same at train and test time and works with batch size 1 and variable-length sequences. Both then apply learnable γ (scale) and β (shift). Data normalisation (zero mean, unit std per input channel) is the preprocessing version.

</details>

**Q188. [Week 9: GAN]**

How is a GAN trained?

- A. A generator maps random noise to fake samples; a discriminator classifies real (1) vs fake (0) with binary cross-entropy; they are trained adversarially in alternation
- B. An encoder and decoder minimise reconstruction error
- C. A model predicts the noise added at each timestep
- D. A velocity field is regressed with MSE

<details><summary>Answer</summary>

**A.** Only the decoder or generator is needed to generate data. The discriminator drives the transformation from noise to data.

Why each option:

- A ✓ Correct.
- B ✗ That's an autoencoder.
- C ✗ That's DDPM.
- D ✗ That's flow matching.

In depth: Generator G: z ~ N(0, I) → fake x̃. Discriminator D: x → P(real). Alternate two steps. (1) D-step: BCE loss with real images labelled 1 and G(z) labelled 0, update D. (2) G-step: generate fakes, label them 1 (we want D fooled), compute BCE through D, update only G. It's a min-max game: min_G max_D E[log D(x)] + E[log(1 − D(G(z)))]. Challenges: instability and mode collapse (G produces a few samples only). After training, only G is kept.

</details>

**Q189. [Week 9: cGAN]**

What does a conditional GAN add?

- A. A condition (e.g. class) given to the generator (and discriminator) that modulates the features, so we can choose what to generate
- B. An encoder that reconstructs the input
- C. A noise schedule
- D. A causal mask

<details><summary>Answer</summary>

**A.** In the lecture, a 10-class classifier trained on real data is used to check that generated samples match the condition.

Why each option:

- A ✓ Correct.
- B ✗ That's VAE-GAN.
- C ✗ That's diffusion.
- D ✗ That's for transformers.

In depth: In a cGAN, G(z, c) and D(x, c) both receive the condition c (a class embedding, text features). In the lecture's implementation, the class modulates the generator's features. D must judge 'is this a real image of class c?', which forces G to produce the requested class. To evaluate it, a separate classifier trained on real data checks whether generated samples are recognised as the intended class.

</details>

**Q190. [Week 9: VAE-GAN]**

What is a VAE-GAN?

- A. A VAE whose decoder output is also judged by a discriminator, combining a structured latent with sharper GAN-like samples
- B. A GAN without a generator
- C. A diffusion model with two UNets
- D. A k-means variant

<details><summary>Answer</summary>

**A.** VAE reconstructions are often blurry, and the adversarial loss adds sharpness.

Why each option:

- A ✓ Correct.
- B ✗ It has a generator (the decoder).
- C ✗ Not diffusion.
- D ✗ Not clustering.

In depth: VAEs give a structured latent (encode, decode, sample) but blurry outputs, because MSE averages over plausible details. GANs give sharp outputs but no encoder and unstable training. A VAE-GAN keeps the encoder, decoder and KL term, and adds a discriminator on the decoder output (real vs reconstructed or generated), so the decoder is pushed to produce realistic, sharp textures. The loss is reconstruction + β·KL + adversarial.

</details>

**Q191. [Week 9: Score function]**

What is the score function and how can it be used to generate data?

- A. The gradient of the log data density, ∇ₓ log p(x); starting from random x and following it with Langevin dynamics (gradient steps plus small noise) moves samples towards high-density data
- B. The accuracy of a classifier; it is maximised with Adam
- C. The discriminator output of a GAN
- D. The KL divergence between two images

<details><summary>Answer</summary>

**A.** With the score you don't need p(x) itself. Langevin dynamics follows the score with a random walk.

Why each option:

- A ✓ Correct.
- B ✗ Not accuracy.
- C ✗ That's a GAN component.
- D ✗ KL is between distributions.

In depth: Score s(x) = ∇ₓ log p(x): a vector field pointing towards higher data density, and it doesn't need the normalising constant of p. If we know it (learned by a network), Langevin dynamics samples: x_(k+1) = x_k + (η/2)·s(x_k) + √η·ε, with ε ~ N(0, I). The gradient term climbs towards the modes; the noise term keeps samples diverse rather than collapsing to the peak. Start from random noise and iterate to get samples from p(x).

</details>

**Q192. [Week 9: Noise]**

Why is noise added to the data when learning the score?

- A. In sparse areas between data points the score cannot be estimated; adding noise (at multiple scales) spreads the data so the score is defined everywhere
- B. To make training faster by removing data
- C. To fool the discriminator
- D. Noise has no role in score-based models

<details><summary>Answer</summary>

**A.** In practice, start with lots of noise and decrease it progressively.

Why each option:

- A ✓ Correct.
- B ✗ It's not for removing data.
- C ✗ That's a GAN concept.
- D ✗ Noise is central.

In depth: With only training points, the estimated score is accurate near the data but undefined or arbitrary in the empty regions between them, which is exactly where sampling starts (random noise). Perturbing the data with Gaussian noise smears p(x) so the score is defined everywhere. Large noise covers the whole space but blurs detail; small noise is precise but local. So train on multiple noise scales and sample by starting at high noise and annealing down. That's the basis of score-based and diffusion models.

</details>

**Q193. [Week 9: DDPM]**

What is the DDPM model trained to predict?

- A. The noise ε that was added to the image at timestep t (MSE between predicted and true noise), which is related to the score
- B. The class label of the image
- C. Whether the image is real or fake
- D. The KL divergence to the prior

<details><summary>Answer</summary>

**A.** A key finding is that the score at t is related to the noise added at t. Sampling starts from noise and iterates down to t = 0.

Why each option:

- A ✓ Correct.
- B ✗ Not a classifier.
- C ✗ That's a GAN discriminator.
- D ✗ That's a VAE term.

In depth: Training loop: (1) take a clean image x₀. (2) Pick a random timestep t. (3) Sample ε ~ N(0, I) and form x_t = √(ᾱ_t)x₀ + √(1 − ᾱ_t)ε in one shot. (4) The network ε_θ(x_t, t) predicts ε. (5) Loss = ‖ε − ε_θ(x_t, t)‖² (MSE). The key finding: the score at time t is −ε/√(1 − ᾱ_t), so predicting noise is equivalent to learning the score. Sampling: start from x_T ~ N(0, I) and repeatedly remove the predicted noise (plus a little fresh noise, Langevin-style) until t = 0.

</details>

**Q194. [Week 9: Noise schedule]**

What is the noise schedule in a diffusion model?

- A. The function β(t) that sets how much noise is added at each timestep of the forward process (e.g. linear or cosine)
- B. The learning rate schedule of the optimiser
- C. The order in which images are shown to the model
- D. The number of GPUs used

<details><summary>Answer</summary>

**A.** The lecture used a linear β(t). Improved diffusion uses other options such as cosine.

Why each option:

- A ✓ Correct.
- B ✗ That's an optimiser schedule.
- C ✗ That's curriculum learning.
- D ✗ Hardware is unrelated.

In depth: β_t is the variance of noise added at step t: x_t = √(1 − β_t)·x_(t−1) + √(β_t)·ε. Define α_t = 1 − β_t and ᾱ_t = Π_(s≤t) α_s. Then ᾱ_t controls how much of the original signal remains, which allows jumping to any t directly. The lecture uses a linear schedule (β from 10⁻⁴ to 0.02 over T = 1000); improved DDPM uses a cosine schedule that destroys information more gradually. The schedule affects sample quality and how many steps are needed.

</details>

**Q195. [Week 9: Guidance]**

What is classifier-free guidance?

- A. The same diffusion model makes a conditional and an unconditional prediction at each step; their difference is scaled to push generation towards the condition, with no separate classifier
- B. Training a separate classifier on noisy images and adding its gradient
- C. Generating without any condition
- D. Using a GAN discriminator as guide

<details><summary>Answer</summary>

**A.** Classifier guidance uses Bayes' rule and needs a separate classifier's gradient ∇ log p(c|x). CFG gets this by subtracting the unconditional score from the conditional score.

Why each option:

- A ✓ Correct.
- B ✗ That's classifier guidance.
- C ✗ That's unconditional generation.
- D ✗ Not used.

In depth: Bayes: ∇log p(x|c) = ∇log p(x) + ∇log p(c|x). Classifier guidance gets ∇log p(c|x) from a separate classifier trained on noisy images. Classifier-free guidance flips it: ∇log p(c|x) = ∇log p(x|c) − ∇log p(x). Train one model that sometimes drops the condition (c = ∅), so it learns both scores. At sampling, combine: ε̂ = ε(x, ∅) + w·[ε(x, c) − ε(x, ∅)]. A guidance scale w > 1 pushes harder towards the prompt (better adherence, less diversity). This is what Stable Diffusion uses.

</details>

**Q196. [Week 9: Flow matching]**

How does conditional flow matching work?

- A. Pair random noise samples with data samples, train a network to regress the velocity along the path between them (MSE), then generate by solving the ODE dx/dt = u_θ(x, t)
- B. Train a discriminator to separate real and fake
- C. Add noise and use a random walk (Langevin) to sample
- D. Minimise the KL divergence to a Gaussian prior

<details><summary>Answer</summary>

**A.** Flow matching avoids the random walk. Generation still needs multiple steps because the learned velocity field is non-linear.

Why each option:

- A ✓ Correct.
- B ✗ That's a GAN.
- C ✗ That's score-based diffusion (Langevin).
- D ✗ That's the VAE KL term.

In depth: Pair x₀ ~ N(0, I) (noise) with x₁ (a data sample). Use the straight path x_t = (1 − t)x₀ + t·x₁, whose true velocity is dx_t/dt = x₁ − x₀. Train u_θ(x_t, t) with loss = MSE(u_θ(x_t, t), x₁ − x₀). Many random pairings average out to a smooth velocity field. Generation: start from noise and solve the ODE dx/dt = u_θ(x, t) from t = 0 to 1 with an ODE solver (Euler, RK). It's deterministic, with no random walk, but needs several steps because the learned field is curved. Rectified flow straightens the paths to allow fewer steps.

</details>

**Q197. [Week 9: Generative summary]**

Which statement correctly contrasts AE/VAE/GAN with diffusion/flow matching?

- A. AE/VAE/GAN: low-dimensional latent, one-shot generation, but many random z give bad samples; diffusion/flow: latent same dimension as x, multi-step generation, any random z gives a good sample
- B. Diffusion generates in one step; VAE needs many steps
- C. GANs use an ODE solver
- D. All of them use the same loss function

<details><summary>Answer</summary>

**A.** That's the summary from the lecture: one-shot decoders vs iterative transport of noise to data.

Why each option:

- A ✓ Correct.
- B ✗ Reversed.
- C ✗ Flow matching uses an ODE solver.
- D ✗ Their losses differ (reconstruction + KL, BCE, noise MSE, velocity MSE).

In depth: One-shot decoders (AE, VAE, GAN): the latent z is low-dimensional, and generation is one pass x = Dec(z), which is fast. But the decoder only produces good outputs for z in the learned region, and many random z give bad samples. Iterative transports (diffusion, flow matching): the 'latent' is noise of the same dimension as x. Generation moves it step by step (SDE or ODE) towards the data, which is slower but any random start gives a good sample. Latent diffusion combines both: diffusion in a VAE's latent.

</details>

**Q198. [Week 9: FID]**

How does the Fréchet Inception Distance (FID) measure generation quality?

- A. Real and generated images pass through a pre-trained Inception-v3; feature means and covariances are compared as Gaussians using the Fréchet distance (lower is better)
- B. It counts the pixels that differ between images
- C. It measures the discriminator loss
- D. It is the reconstruction MSE of a VAE

<details><summary>Answer</summary>

**A.** It's not perfect but widely used. Similar metrics include IS, KID and SWD.

Why each option:

- A ✓ Correct.
- B ✗ Pixel differences don't capture realism.
- C ✗ Not a sample-quality metric.
- D ✗ Reconstruction isn't generation quality.

In depth: Procedure: (1) resize real and generated images to 299×299 and pass them through a pre-trained Inception-v3 (48 layers, ~23.8M parameters, trained on ImageNet). (2) Take the 2048-dim pool3 features. (3) Fit a Gaussian to each set: (μ_r, Σ_r) and (μ_g, Σ_g). (4) FID = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_rΣ_g)^(1/2)). The first term compares mean features (are the samples realistic on average?); the trace term compares spread (diversity). Lower is better and 0 means identical statistics. Other metrics: IS, KID, SWD.

</details>

<a id="week-10"></a>

## Week 10: Implicit Models (NeRF, Splatting)

**Q199. [Week 10: Overfitting as a feature]**

What does 'overfitting as a feature' mean for implicit neural representations?

- A. Instead of generalising, an MLP is deliberately overfitted to one signal (image, shape, scene) so its weights become a representation of that signal
- B. Overfitting always improves test accuracy
- C. The model is trained only on the test set
- D. Dropout is used to increase overfitting in classifiers

<details><summary>Answer</summary>

**A.** Usually overfitting is bad. In an INR the goal is to fit one signal as precisely as possible.

Why each option:

- A ✓ Correct.
- B ✗ Overfitting hurts generalisation in normal ML.
- C ✗ Not about test-set training.
- D ✗ Dropout reduces overfitting.

In depth: Usually we train on many samples and want to generalise to new ones, so overfitting is bad. In implicit neural representations (INR) the dataset is a single signal: an image I(x, y), a shape's SDF, a 3D scene. Every coordinate is a training sample, and the goal is to memorise it as exactly as possible. The network's weights become a compact, continuous, differentiable encoding of that signal. NeRF and 4D video capture (4dv.ai) use this idea.

</details>

**Q200. [Week 10: Universal approximation]**

What does the universal approximation theorem state?

- A. A network with one hidden layer and enough units can approximate any continuous function arbitrarily closely
- B. Any network trains to the global minimum
- C. Deep networks never overfit
- D. Only convolutional networks can approximate functions

<details><summary>Answer</summary>

**A.** In theory yes. In practice, the more units, the harder the optimisation, so deeper networks are often used.

Why each option:

- A ✓ Correct.
- B ✗ Not a convergence guarantee.
- C ✗ Unrelated to overfitting.
- D ✗ Applies to MLPs.

In depth: A feed-forward network with one hidden layer of sigmoid or ReLU units can approximate any continuous function on a bounded domain to any accuracy ε, given enough units. Constructive intuition: pairs of steep sigmoids make 'bumps' (boxes), and summing many shifted, scaled bumps approximates any curve. The theorem says nothing about how many units are needed or whether gradient descent will find the weights. In practice deeper networks (several layers of 100 units) approximate better with fewer parameters.

</details>

**Q201. [Week 10: Sigmoid shaping]**

For a unit f(x) = sigmoid(w·x + b), what are the effects of w and b?

- A. w controls the steepness of the slope (high w → sharp step); b shifts the sigmoid along x
- B. w shifts the sigmoid; b controls the steepness
- C. Both only scale the output
- D. Neither changes the shape

<details><summary>Answer</summary>

**A.** Two sigmoid units with large w and opposite signs make a square (box) function. Many boxes can approximate any 1D function.

Why each option:

- A ✓ Correct.
- B ✗ Swapped.
- C ✗ They change the steepness and location.
- D ✗ They clearly change the shape.

In depth: sigmoid(wx + b) switches from 0 to 1 around x = −b/w. Increasing |w| makes the transition sharper (w = 50 is almost a step); the sign of w flips the direction. b shifts the location of the step. Build a box on [a, c]: h₁ = σ(w(x − a)), h₂ = σ(w(x − c)) with large w, output = h₁ − h₂, which is ≈ 1 between a and c and 0 outside. Ten such units can approximate a 1D curve with ~5 boxes; more units give finer approximation.

</details>

**Q202. [Week 10: SDF]**

What is a signed distance function (SDF) of a shape?

- A. At each location, the closest distance to the shape boundary: positive outside, negative inside
- B. The pixel intensity of the image
- C. The probability that a pixel belongs to a class
- D. The distance between two cameras

<details><summary>Answer</summary>

**A.** Thresholding at 0 gives the binary mask. The value gives the distance to the object, and the gradient gives its direction.

Why each option:

- A ✓ Correct.
- B ✗ That's an image function.
- C ✗ That's a segmentation probability.
- D ✗ Unrelated.

In depth: SDF(p) = ±distance from point p to the nearest point on the shape's boundary: > 0 outside, < 0 inside, = 0 exactly on the surface (the zero level set). Uses: threshold at 0 to get a binary mask at any resolution; the value gives the closest distance to the object (for collision checks and ray marching); the gradient ∇SDF has unit length and points away from the surface, giving the surface normal and the direction to the object. An MLP (x, y) → SDF learns a smooth version.

</details>

**Q203. [Week 10: SDF]**

How is the gradient of a learned SDF (MLP) computed?

- A. Using PyTorch autograd to differentiate the network output with respect to the input coordinates
- B. By finite differences on the training labels only
- C. By running k-means on the output
- D. It cannot be computed

<details><summary>Answer</summary>

**A.** The MLP is differentiable with respect to its inputs (x, y), so autograd gives ∇SDF at any location.

Why each option:

- A ✓ Correct.
- B ✗ Autograd is exact and works at any coordinate.
- C ✗ Not related.
- D ✗ It can be computed.

In depth: The MLP f(x, y) is a composition of differentiable operations, so set requires_grad = True on the input coordinate tensor, compute out = f(coords), then call torch.autograd.grad(out.sum(), coords). This returns ∂f/∂x and ∂f/∂y for every coordinate in one backward pass. It's the same machinery as training, but differentiating with respect to inputs instead of weights (like saliency maps). For a true SDF, ‖∇f‖ = 1, and an 'eikonal' loss can enforce it.

</details>

**Q204. [Week 10: Resolution]**

Why is an implicit representation said to have 'infinite resolution'?

- A. The trained MLP is a continuous function of coordinates, so it can be queried at any resolution without changing the number of parameters
- B. It stores every pixel at infinite precision
- C. It uses an infinite number of neurons
- D. It compresses images to zero bytes

<details><summary>Answer</summary>

**A.** The same model generated a 1000×1000 star. That's 100× more pixels with the same number of parameters (~105 floats).

Why each option:

- A ✓ Correct.
- B ✗ Values are stored as weights, not pixels.
- C ✗ A finite network.
- D ✗ Not zero bytes; it's a compact set of weights.

In depth: A pixel image stores values on a fixed grid (100×100 = 10,000 bytes). An INR stores weights of a function f(x, y), continuous in x and y. To render at 1000×1000, simply query f at a finer grid of coordinates: 100× more pixels, same ~105 floats (~450 bytes) of parameters. The level of detail is limited by what the network learned (it smoothly interpolates between training coordinates), but there's no fixed resolution.

</details>

**Q205. [Week 10: Spectral bias]**

What is spectral bias in coordinate-based MLPs, and how is it addressed?

- A. MLPs are biased towards learning low-frequency functions (blobby results); mapping inputs through high-frequency sinusoids (Fourier features / positional encoding) restores detail
- B. MLPs only learn high frequencies; fix by blurring the input
- C. MLPs ignore colour; fix by adding a colour channel
- D. It is caused by the optimiser; fix by using SGD

<details><summary>Answer</summary>

**A.** Neural Tangent Kernel theory explains why raw (x, y, z) inputs make the MLP act like a low-pass filter.

Why each option:

- A ✓ Correct.
- B ✗ It's the opposite bias.
- C ✗ Not about colour.
- D ✗ Not optimiser-specific.

In depth: Neural Tangent Kernel analysis shows MLPs fit low-frequency components of a target function first and much faster; high frequencies (sharp edges, fine texture) are learned very slowly. With raw (x, y, z) inputs the network acts like a low-pass filter, so results look blobby. Fix: Fourier feature mapping γ(p) = [sin(2⁰πp), cos(2⁰πp), …, sin(2^(L−1)πp), cos(2^(L−1)πp)]. High-frequency sinusoids make fine detail a simple function of the encoded input. The same idea as transformer positional encoding.

</details>

**Q206. [Week 10: NeRF]**

What does a NeRF MLP take as input and produce as output?

- A. Input: 3D position (x, y, z) and viewing direction (θ, φ), i.e. 5D; output: colour (RGB) and volume density σ
- B. Input: an image; output: a class label
- C. Input: noise; output: an image
- D. Input: text; output: next token

<details><summary>Answer</summary>

**A.** Colours along each camera ray are integrated with volume rendering to produce each pixel.

Why each option:

- A ✓ Correct.
- B ✗ That's a classifier.
- C ✗ That's a GAN or diffusion model.
- D ✗ That's an LLM.

In depth: F_θ: (x, y, z, θ, φ) → (r, g, b, σ). Density σ depends only on position (geometry); colour also depends on viewing direction, so specular reflections can change with viewpoint. To render a pixel: cast a ray from the camera through it, sample points along the ray, query the MLP at each, and combine the colours with volume rendering weights. Training minimises the MSE between rendered and real pixel colours over many posed images.

</details>

**Q207. [Week 10: NeRF encoding]**

In NeRF, positional encoding with L = 10 is applied to each component of the 3D position. What is the resulting input dimension?

- A. 60 (3 × 2L)
- B. 10
- C. 30
- D. 3

<details><summary>Answer</summary>

**A.** Each scalar maps to 2L sin/cos values: 3 × 20 = 60. For the viewing direction L = 4 is used (24 dims with a unit vector).

Why each option:

- A ✓ Correct: 3 × 2 × 10.
- B ✗ That's just L.
- C ✗ Forgets the sin/cos pair.
- D ✗ That's the raw input.

In depth: γ(p) maps each scalar to 2L values: L sines and L cosines at frequencies 2⁰…2^(L−1). Position has 3 components: 3 × 2 × 10 = 60 dimensions (the original paper also concatenates the raw xyz, giving 63). Direction as a 3D unit vector with L = 4: 3 × 2 × 4 = 24 (27 with raw input). Formula to remember: dim = (number of components) × 2L.

</details>

**Q208. [Week 10: NeRF]**

NeRF needs the camera pose for each training image. How is it obtained?

- A. Provided with the data, estimated beforehand with structure-from-motion tools such as COLMAP, or optimised jointly (e.g. BARF)
- B. It is always assumed to be the identity
- C. It is predicted by a GAN discriminator
- D. It is not needed

<details><summary>Answer</summary>

**A.** Without poses, rays cannot be cast through each pixel into the scene.

Why each option:

- A ✓ Correct.
- B ✗ Cameras are at different poses.
- C ✗ No discriminator in NeRF.
- D ✗ Rays require poses.

In depth: Each training image needs its extrinsics (where the camera is and where it points) and intrinsics (focal length) to cast the correct ray per pixel. Options: (1) given by the capture rig or synthetic data; (2) estimated beforehand with structure-from-motion, where COLMAP matches features across images and solves for poses plus a sparse point cloud; (3) optimised jointly with the radiance field, as in BARF (Bundle-Adjusting NeRF), which treats poses as learnable parameters.

</details>

**Q209. [Week 10: Volume rendering]**

In NeRF volume rendering, what is the transmittance T(t)?

- A. The probability that the ray travels from the camera to point t without being blocked
- B. The colour of the point at t
- C. The learning rate at step t
- D. The density of the medium at the camera

<details><summary>Answer</summary>

**A.** The pixel colour is a weighted sum of segment colours, with weights from density and transmittance (alpha compositing in discrete form).

Why each option:

- A ✓ Correct.
- B ✗ That's c(t).
- C ✗ Unrelated.
- D ✗ That's σ.

In depth: Along ray r(t) = o + t·d: the chance of hitting a particle in [t, t + dt] is σ(t)dt. Transmittance T(t) = exp(−∫₀ᵗ σ(s)ds) is the probability that light travels from the camera to t without being blocked. The colour is C = ∫ T(t)σ(t)c(t)dt. Discretised into N segments of length δᵢ: αᵢ = 1 − exp(−σᵢδᵢ), Tᵢ = Πⱼ<ᵢ(1 − αⱼ), and C = Σ Tᵢαᵢcᵢ, which is alpha compositing front to back. NeRF uses a coarse pass plus a fine pass that samples more points where the weights are high.

</details>

**Q210. [Week 10: Alpha compositing]**

In alpha compositing, what does α = 1 represent?

- A. A completely opaque object that contributes its full colour
- B. A completely transparent object
- C. Half transparency
- D. An invalid value

<details><summary>Answer</summary>

**A.** The final colour is a linear combination of the colours, weighted by their alpha (opacity) values.

Why each option:

- A ✓ Correct.
- B ✗ That's α = 0.
- C ✗ That's α = 0.5.
- D ✗ α = 1 is valid.

In depth: α ∈ [0, 1] is opacity. α = 1 means fully opaque: it contributes all its colour and blocks everything behind it. α = 0 means fully transparent: it contributes nothing. For K layers front to back: C = Σₖ cₖαₖ Πⱼ<ₖ(1 − αⱼ). Each layer's weight is its own opacity times the transparency of everything in front. NeRF and Gaussian splatting both render this way.

</details>

**Q211. [Week 10: Gaussian splatting]**

What is the key idea of 3D Gaussian splatting compared with NeRF?

- A. Represent the radiance field sparsely with 3D Gaussians only where density is non-zero, then project them into 2D Gaussians and sum their contributions directly
- B. Represent every point in space densely with an MLP
- C. Use a GAN to generate 3D scenes
- D. Use a decision tree to store voxels

<details><summary>Answer</summary>

**A.** Natural scenes are mostly air, so a sparse representation is far cheaper to render. Gaussians are culled outside the view frustum.

Why each option:

- A ✓ Correct.
- B ✗ That's NeRF.
- C ✗ Not GAN-based.
- D ✗ Not tree-based.

In depth: NeRF parameterises the radiance field densely: an MLP must be queried at many points along every ray, even in empty air, which is slow. Splatting parameterises it sparsely: millions of 3D Gaussians, each with position, covariance (shape and orientation), opacity and colour, placed only where there's matter. Rendering: cull Gaussians outside the view frustum, project each to a 2D Gaussian on the image, sort by depth, and alpha-composite. It's rasterisation rather than ray marching, so it's real-time and fully differentiable.

</details>

**Q212. [Week 10: Gaussian splatting]**

How is Gaussian splatting typically initialised and refined?

- A. Start from a Structure-from-Motion (SfM) point cloud, then iterate with cloning and splitting of Gaussians
- B. Start from random noise and run Langevin dynamics
- C. Start from a pre-trained ResNet
- D. Start from a single Gaussian at the origin and never change

<details><summary>Answer</summary>

**A.** Densification by cloning and splitting adds detail where needed.

Why each option:

- A ✓ Correct.
- B ✗ That's diffusion sampling.
- C ✗ Not used.
- D ✗ It needs many Gaussians.

In depth: COLMAP structure-from-motion gives camera poses and a sparse 3D point cloud. Each point becomes an initial Gaussian. Optimisation renders the views, compares them with the photos, and backpropagates to all Gaussian parameters. Adaptive density control: clone small Gaussians in under-reconstructed regions, split large Gaussians covering too much detail, and prune nearly transparent ones. This densification turns a sparse cloud into a detailed scene.

</details>

**Q213. [Week 10: Graphics]**

In the camera model, what do the extrinsic [R|T] and intrinsic K matrices do?

- A. [R|T] maps world coordinates to camera coordinates (rotation + translation); K maps camera coordinates to the image (screen) projection
- B. [R|T] stores the pixel colours; K stores the depth
- C. Both are learned by the NeRF MLP
- D. K maps world to camera; [R|T] maps camera to screen

<details><summary>Answer</summary>

**A.** Rays are then defined from the camera centre through each pixel: X = P + t·d, with t > 0.

Why each option:

- A ✓ Correct.
- B ✗ Matrices map coordinates, not colours.
- C ✗ They come from calibration or SfM.
- D ✗ Swapped.

In depth: A world point X_w becomes X_c = R·X_w + T (extrinsic: rotation R and translation T, describing camera pose), then the pixel u ~ K·X_c (intrinsic: focal lengths f_x, f_y and principal point c_x, c_y), followed by division by depth. Ray tracing does the inverse: for each pixel, build a ray X = P + t·d from the camera centre P through the pixel, with t > 0 (in front of the camera).

</details>

**Q214. [Week 10: Materials]**

How do diffuse, specular and mirror materials differ (BRDF)?

- A. Diffuse reflects incoming light equally in all directions; specular reflects more in a particular direction; mirrors reflect only in one direction
- B. All three reflect light equally in all directions
- C. Diffuse materials emit no light at all
- D. Mirrors absorb all light

<details><summary>Answer</summary>

**A.** The BRDF gives the fraction of light from direction s reflected towards direction v. Lambert's law adds the cosine (dot product with the normal) factor.

Why each option:

- A ✓ Correct.
- B ✗ They differ.
- C ✗ Diffuse surfaces reflect light.
- D ✗ Mirrors reflect, not absorb.

In depth: The rendering equation: outgoing light = emitted + ∫ BRDF(s → v) × incoming light(s) × (n·s) ds. The BRDF says what fraction of light from direction s is reflected towards view direction v. Diffuse (Lambertian): reflects equally in all directions, so it looks the same from every viewpoint. Specular: reflects mostly around the mirror direction (highlights). Mirror: only the exact reflection direction. Lambert's law gives the n·s factor: light hitting at a grazing angle spreads over more area, so it's dimmer. This is why NeRF's colour depends on viewing direction.

</details>

**Q215. [Week 10: Deep MLPs]**

A 5-layer, 256-unit ReLU MLP fitted to an image misses details. Which fix did the lecture suggest?

- A. Add batch normalisation to normalise features between layers and thus the gradients in deep networks
- B. Remove all activation functions
- C. Use fewer training pixels
- D. Replace the MLP with k-means

<details><summary>Answer</summary>

**A.** Deep networks can struggle to backpropagate gradients, and normalisation helps. Fourier features also address missing high-frequency detail.

Why each option:

- A ✓ Correct.
- B ✗ That makes it linear.
- C ✗ Less data doesn't help.
- D ✗ Not appropriate.

In depth: Deeper MLPs fitting I(x, y) can struggle: gradients don't propagate well through many layers, so some units barely train and fine detail is missing. The lecture's refresher: adding batch norm between layers normalises the features, which normalises the gradients and helps optimisation. Separately, missing high-frequency detail is the spectral bias, fixed by Fourier feature encoding of the input coordinates.

</details>

**Q216. [Week 10: Geometric transforms]**

How can an image represented by an MLP I(x, y) be rotated, scaled or warped?

- A. Transform the input coordinates (e.g. X' = (T + P + S)·R·X) before querying the MLP
- B. Retrain the MLP from scratch for every rotation
- C. Rotate the weights of the first layer only
- D. It cannot be transformed

<details><summary>Answer</summary>

**A.** Since the image is a function of coordinates, warping the coordinates warps the image, including projective transformations.

Why each option:

- A ✓ Correct.
- B ✗ No retraining is needed.
- C ✗ Not how it's done.
- D ✗ It can be transformed.

In depth: The MLP is f(x, y) → intensity. To warp the image, warp the coordinates: compute X' = M·X for each pixel coordinate and evaluate f(X'). With homogeneous coordinates, rotation R, translation T, scaling S and perspective P are all 3×3 matrices (e.g. X' = (T + P + S) @ R @ X in the lecture, with angle = π/6, scale = 1.5, perspective = −0.1). Projective transformations are exactly how 3D-to-2D camera projection works, which leads to NeRF.

</details>

<a id="week-11"></a>

## Week 11: Advanced Methods (ViT, DINO, Latent Diffusion)

**Q217. [Week 11: ViT]**

How does a Vision Transformer (ViT) apply the LLM architecture to images?

- A. The image is split into patches (the equivalent of tokens), each projected into an embedding vector with a 2D positional encoding, then passed through transformer blocks
- B. Each pixel is treated as a word in a dictionary of 256 symbols
- C. The image is converted to text with OCR first
- D. Convolutions replace all attention layers

<details><summary>Answer</summary>

**A.** It's the same principle as words to tokens to embeddings, using a sequence of patches instead of words.

Why each option:

- A ✓ Correct.
- B ✗ Pixels aren't treated as words.
- C ✗ No OCR.
- D ✗ ViT is attention-based.

In depth: ViT: split a 224×224 image into 16×16 patches, giving 14 × 14 = 196 patches. Flatten each (16·16·3 = 768 values) and apply a linear projection to the embedding dimension. Add a learned or 2D positional encoding (attention ignores order otherwise), often prepend a [CLS] token, then run standard transformer blocks. The [CLS] output feeds the classifier. Like LLMs, ViTs are data hungry, and DINO and DINOv3 use ViT backbones.

</details>

**Q218. [Week 11: LLM refresher]**

What is the typical pipeline of an LLM forward pass?

- A. Text → tokens → embeddings + positional encoding → N transformer blocks → probabilities for the next token over the vocabulary (~50,000)
- B. Text → pixels → convolutions → class label
- C. Tokens → k-means → decision tree
- D. Noise → generator → discriminator → text

<details><summary>Answer</summary>

**A.** In the lecture example, 9 tokens with 16-dim embeddings produce ~50,000 next-token probabilities.

Why each option:

- A ✓ Correct.
- B ✗ That's a CNN classifier.
- C ✗ Nonsense.
- D ✗ That's a GAN.

In depth: Example from the slide: 'QUT the university for the real…' is tokenised into 9 tokens (from ~50,000), each mapped to an embedding (9 × 16) plus positional encoding (9 × 16). N transformer blocks (GPT-3: 96) refine these vectors. The final linear layer plus softmax gives ~50,000 probabilities for the next token (e.g. 'world' 0.8). Sample, append, repeat.

</details>

**Q219. [Week 11: Attention matrix]**

After applying softmax to QKᵀ, what property does each row of the attention matrix A have?

- A. It sums to 1 (it is a set of weights over the tokens)
- B. It sums to 0
- C. All values are negative
- D. It is always the identity

<details><summary>Answer</summary>

**A.** A = softmax(QKᵀ/√D), then AV mixes the values. With a causal mask the future entries are 0.

Why each option:

- A ✓ Correct.
- B ✗ Softmax rows sum to 1.
- C ✗ Softmax outputs are ≥ 0.
- D ✗ Only if each token attends only to itself.

In depth: Softmax is applied row by row: for query token i, the scores against all keys j are exponentiated and normalised, so row i is a probability distribution over tokens (all ≥ 0, sum = 1). The output for token i is then Σⱼ Aᵢⱼvⱼ, a weighted average of values. With a causal mask, entries j > i are set to −∞ before softmax and become 0.

</details>

**Q220. [Week 11: DINO]**

Which technique does DINO use to learn image features without labels?

- A. Self-distillation: a student and a teacher (an EMA of the student) see different augmented views; cross-entropy matches their outputs, with centering and sharpening to avoid collapse
- B. A GAN discriminator that classifies real and fake
- C. Supervised training on ImageNet labels
- D. Next-token prediction on text

<details><summary>Answer</summary>

**A.** A stop-gradient (.detach()) is applied to the teacher. Centering prevents mode collapse, where the teacher outputs a constant.

Why each option:

- A ✓ Correct.
- B ✗ DINO isn't a GAN.
- C ✗ No labels are used.
- D ✗ That's for LLMs.

In depth: DINO (self-DIstillation with NO labels): two random augmentations x₁, x₂ of an image (global and local crops, colour jitter). The student network gets both; the teacher, whose weights are an EMA of the student, gets global views, and a stop-gradient (.detach()) is applied. The loss is cross-entropy between the teacher's softmax output (centred and sharpened with a low temperature) and the student's. Centering stops one dimension dominating, and sharpening stops a uniform output; together they prevent collapse. The learned ViT features are excellent: kNN on them gives top classification.

</details>

**Q221. [Week 11: DINO]**

In DINO, what is the purpose of centering the teacher's features?

- A. To avoid mode collapse, where the teacher collapses to a constant output
- B. To increase the learning rate
- C. To make the student a copy of the teacher
- D. To remove data augmentation

<details><summary>Answer</summary>

**A.** Centering and softmax sharpening balance each other. Interestingly, the best classification used kNN on DINO features.

Why each option:

- A ✓ Correct.
- B ✗ Unrelated to the learning rate.
- C ✗ The student learns from the teacher but isn't a copy.
- D ✗ Augmentation is essential.

In depth: Collapse is the classic failure of self-distillation: if teacher and student both output the same constant vector, the loss is minimal but nothing is learned. Centering subtracts a running mean (EMA) from the teacher's logits, so no single output dimension can dominate for all images. Sharpening (a low softmax temperature) prevents the other collapse, a uniform distribution. Their balance keeps the representations informative.

</details>

**Q222. [Week 11: VAE trade-off]**

What latent-size trade-off does a VAE face?

- A. Small latent (e.g. 64): blurry reconstructions and blurry random samples; large latent (e.g. 2048): sharp reconstructions, but random samples are rubbish
- B. Larger latent always gives better random samples
- C. Latent size has no effect
- D. Small latent gives sharp reconstructions

<details><summary>Answer</summary>

**A.** Most locations in a large latent space are invalid, so random z decodes poorly.

Why each option:

- A ✓ Correct.
- B ✗ Random samples get worse.
- C ✗ It has a big effect.
- D ✗ Reversed.

In depth: With a small latent (64 dims), the VAE must compress heavily, so reconstructions are blurry; but the latent is easy to match to N(0, I), so random samples are equally blurry but valid. With a large latent (2048), reconstructions are sharp, but the KL can't make the whole high-dimensional space valid, so most random z fall in regions the decoder never saw and generate rubbish. That's the motivation for latent diffusion: keep a fairly large latent for quality and use diffusion to find valid latent points.

</details>

**Q223. [Week 11: Latent diffusion]**

How does Stable Diffusion (latent diffusion) generate images?

- A. A VAE reduces the dimension; diffusion runs in the VAE latent space from Gaussian noise to a valid latent with a UNet (with transformers), conditioned on text; the decoder outputs the image
- B. Diffusion runs directly on full-resolution pixels with no autoencoder
- C. A GAN generates the image in one step from text
- D. An LLM writes the pixel values as text

<details><summary>Answer</summary>

**A.** Conditions (text, images, maps…) are encoded and fed into the UNet features. Diffusion finds the valid latent locations the VAE alone cannot.

Why each option:

- A ✓ Correct.
- B ✗ That's pixel-space diffusion; SD uses a latent.
- C ✗ That's a text-to-image GAN.
- D ✗ Not how it works.

In depth: Pipeline: (1) a VAE encoder compresses 512×512×3 images to a 64×64×4 latent (48× fewer values). (2) Training: forward diffusion adds noise to latents, and a UNet with transformer and cross-attention blocks learns to denoise, conditioned on a text embedding (from CLIP), with the conditions concatenated with or attending to the UNet features. (3) Generation: start from Gaussian noise in latent space, run reverse diffusion (SDE or ODE sampler) with classifier-free guidance, then decode with the VAE decoder. Other conditions (images, depth maps) can be added.

</details>

**Q224. [Week 11: Latent diffusion]**

Why combine a VAE with diffusion (or flow matching) in modern generative AI?

- A. Most latent locations of an AE/VAE are invalid; diffusion takes many small steps from low-dimensional noise to valid latents, while working in the latent keeps it cheap
- B. Because VAEs cannot reconstruct images
- C. Because diffusion cannot handle noise
- D. To remove the need for training

<details><summary>Answer</summary>

**A.** Most modern generative AI generates the latent of a VAE, then decodes it.

Why each option:

- A ✓ Correct.
- B ✗ VAEs reconstruct well with a large latent.
- C ✗ Diffusion is built on noise.
- D ✗ Both need training.

In depth: Two complementary problems. Pixel-space diffusion is expensive (every step processes the full image), and a VAE's latent is compact but mostly invalid for random sampling. Latent diffusion: the VAE handles perceptual compression (cheap and sharp reconstruction), and diffusion or flow matching learns to transport low-dimensional noise to the valid latent manifold with many small denoising steps. That's why 'most modern generative AI generates the latent of a VAE'.

</details>

**Q225. [Week 11: Architectures]**

Which architectural trend did the lecture highlight for image models?

- A. Transformer blocks inside UNets are now common (e.g. for medical imaging and diffusion denoisers)
- B. UNets have been replaced by decision trees
- C. Transformers cannot be used for images
- D. Skip connections are no longer used

<details><summary>Answer</summary>

**A.** Combining convolutional multiscale structure with attention is the state of the art.

Why each option:

- A ✓ Correct.
- B ✗ Not true.
- C ✗ ViT shows they can.
- D ✗ Skips are ubiquitous.

In depth: State-of-the-art segmentation and diffusion denoisers combine convolutional UNets (multiscale, local inductive bias, efficient at high resolution) with transformer and attention blocks (global context, cross-attention to conditions like text). Examples: TransUNet and Swin-UNet in medical imaging, and the Stable Diffusion UNet with attention at the lower resolutions. Pure Diffusion Transformers (DiT) are also emerging.

</details>

**Q226. [Week 11: Intelligence]**

What argument do Yann LeCun and others make about intelligence and generative models?

- A. Intelligence does not require a decoder (generating data) but is based on prediction, in a representation (feature) space
- B. Intelligence requires generating perfect images
- C. Only GANs can be intelligent
- D. Intelligence is impossible to model

<details><summary>Answer</summary>

**A.** Humans don't generate perfect images, and many narrow AI tasks work on features (Enc → z → classifier).

Why each option:

- A ✓ Correct.
- B ✗ That's the view being argued against.
- C ✗ Not the argument.
- D ✗ Not the argument.

In depth: Argument: generating every pixel is wasteful; humans don't produce perfect images, and many tasks only need features (Enc → z → classifier). Intelligence is about prediction: an agent that can predict how the world will evolve, in an abstract representation space, can plan and survive. So train encoders to predict representations of missing or future parts of the input (JEPA), not pixels, ignoring unpredictable detail such as leaves rustling.

</details>

**Q227. [Week 11: JEPA]**

What characterises a joint-embedding predictive architecture (JEPA) compared with contrastive learning?

- A. It predicts the embedding of a target from a context directly in representation space, without negative samples (non-contrastive)
- B. It requires carefully chosen negative pairs
- C. It reconstructs pixels with a decoder
- D. It is a supervised classifier

<details><summary>Answer</summary>

**A.** Contrastive learning is difficult because good positive and negative pairs are hard to get and training is long. JEPA avoids negatives.

Why each option:

- A ✓ Correct.
- B ✗ That's contrastive learning.
- C ✗ That's an autoencoder or MAE.
- D ✗ It's self-supervised.

In depth: JEPA: encode a context x and a target y (e.g. visible and masked image regions, or the present and future of a video). A predictor maps the context embedding to a prediction of the target embedding, and the loss is a distance in representation space. No decoder reconstructs pixels and no negative samples are needed (non-contrastive). Collapse is avoided with an EMA target encoder and stop-gradient (as in BYOL and DINO). It can be conditioned on actions for world models.

</details>

**Q228. [Week 11: Contrastive]**

In contrastive learning of an encoder, what are positive and negative pairs meant to achieve?

- A. Same object → same (close) features; different objects → different (far) features
- B. Same object → far features; different objects → close features
- C. All features should be identical
- D. Features should equal the pixels

<details><summary>Answer</summary>

**A.** The difficulty is generating good pairs and the long training. That's the motivation for non-contrastive methods.

Why each option:

- A ✓ Correct.
- B ✗ Reversed.
- C ✗ That's collapse.
- D ✗ No compression or learning.

In depth: The encoder should map two views of the same object to nearby embeddings (positive pair) and different objects to distant ones (negative pair), for example with the InfoNCE loss −log[exp(sim(z, z⁺)/τ) / Σ exp(sim(z, zₖ)/τ)]. Difficulties: finding good positives (the augmentation choice) and informative negatives (hard negatives, large batches), plus long training. That's why non-contrastive methods emerged.

</details>

<a id="week-12"></a>

## Week 12: Revision (cross-topic)

**Q229. [Week 12: Revision: methods]**

Which classic method separates classes with a line that minimises errors within a buffer zone (margin)?

- A. Support Vector Machine (SVM)
- B. k-means
- C. Gaussian Mixture Model
- D. k Nearest Neighbours

<details><summary>Answer</summary>

**A.** From the revision map: kNN uses neighbours, k-means uses points, GMM uses Gaussians, trees use conditions on each axis, random forests use ensembles of trees, and SVM uses a line with a buffer zone.

Why each option:

- A ✓ Correct.
- B ✗ Represents clusters by centre points.
- C ✗ Represents clusters by Gaussians.
- D ✗ Uses neighbours.

In depth: SVM finds the hyperplane that maximises the margin, the 'buffer zone' between the classes, by minimising errors and penalising points inside the margin (hinge loss). Only the points on or inside the margin (support vectors) define the boundary. Compare the revision map: kNN uses neighbours, k-means a centre point, GMM Gaussians, a decision tree axis conditions, a random forest an ensemble of trees, and a neuron a line without a margin.

</details>

**Q230. [Week 12: Revision: methods]**

Which method defines clusters by conditions on each axis (e.g. X > 1, Y > 0)?

- A. Decision tree
- B. SVM
- C. k-means
- D. Artificial neuron

<details><summary>Answer</summary>

**A.** Each split is an axis-aligned threshold, and a random forest is an ensemble of such trees.

Why each option:

- A ✓ Correct.
- B ✗ Uses a line with a margin.
- C ✗ Uses centre points.
- D ✗ Uses a single oblique line.

In depth: Each node of a decision tree tests one feature against a threshold (X > 1, Y > 0), so the boundaries are axis-aligned lines and the space is partitioned into rectangles ('Case 1, 2, 3' in the slide). A random forest averages many such trees. SVMs and neurons use oblique lines; k-means uses Voronoi cells around centres.

</details>

**Q231. [Week 12: Revision: errors]**

A Type II error (medical term) corresponds to:

- A. A false negative / miss detection: 'not very good at a task'
- B. A false positive / false alarm: 'overdo a task'
- C. A true positive
- D. A correct rejection

<details><summary>Answer</summary>

**A.** Type I = false positive (false alarm). Type II = false negative (miss).

Why each option:

- A ✓ Correct.
- B ✗ That's Type I.
- C ✗ A correct detection.
- D ✗ A correct rejection.

In depth: Type I = false positive (false alarm: 'overdo a task'). Type II = false negative (miss detection: 'not very good at a task'). Memory trick: Type II is the 'second' error, failing to detect something that is there. In screening you minimise Type II (high sensitivity); in diagnosis, Type I (high specificity).

</details>

**Q232. [Week 12: Revision: metrics]**

Why is accuracy rarely a good performance target?

- A. It is an amalgam of the two types of mistakes, which usually have very different costs
- B. It cannot be computed for binary problems
- C. It is always 100% on the training set
- D. It ignores correct predictions

<details><summary>Answer</summary>

**A.** Choose the performance target carefully, e.g. sensitivity for screening and specificity for diagnosis.

Why each option:

- A ✓ Correct.
- B ✗ It can be computed.
- C ✗ Not necessarily, and that isn't the reason.
- D ✗ It counts correct predictions.

In depth: Accuracy = (TP + TN)/total weights every error the same, but a missed cancer (FN) and a false alarm (FP) have very different costs. Under class imbalance it's dominated by the majority class. Choose metrics that match the application's goal: sensitivity, specificity, precision, recall, ROC/AUC, or a cost-weighted metric.

</details>

**Q233. [Week 12: Revision: experimental design]**

Ideally, where should the testing data come from?

- A. Secret data kept to benchmark the model after training, ideally not from the same source as training/validation data
- B. The same batch as the training data
- C. Randomly generated noise
- D. The validation data reused

<details><summary>Answer</summary>

**A.** Training and validation data are split (e.g. 90/10) to pick hyperparameters. Testing estimates real-world performance.

Why each option:

- A ✓ Correct.
- B ✗ That leaks training data.
- C ✗ Noise doesn't measure performance.
- D ✗ Validation was used for tuning.

In depth: The test set estimates how the model will do in the real world. It must not influence training or hyperparameter choice, or the estimate becomes optimistic. Best practice from the slide: training and validation come from the provided data (e.g. a 90/10 split, or k-fold); testing uses secret data, ideally from a different source (another site, scanner or time period), to measure true generalisation and robustness to data drift.

</details>

**Q234. [Week 12: Revision: typical approach]**

In the typical ML approach, when is gradient-based optimisation needed instead of an analytical solution?

- A. When no analytical solution exists (e.g. due to activation functions) and the parameter space is too large to inspect exhaustively
- B. Always, even for linear least squares
- C. Never; every model has a closed-form solution
- D. Only when there is no data

<details><summary>Answer</summary>

**A.** Get data → choose the simplest model → analytical solution? → can the parameter space be inspected (grid search)? → otherwise optimise → analyse results on the test set.

Why each option:

- A ✓ Correct.
- B ✗ OLS has a closed form.
- C ✗ ANNs don't.
- D ✗ Nothing to optimise without data.

In depth: The six-step recipe: (1) get data and inspect it; (2) choose the simplest possible model; (3) is there an analytical solution (like OLS for linear least squares)? Use it; (4) if not, can the parameter space be inspected (few parameters, so grid search)? (5) Otherwise optimise iteratively (gradient descent or Adam); for ANNs the activation functions make a closed form impossible and there are too many parameters; (6) analyse results on the test set.

</details>

**Q235. [Week 12: Revision: ANN geometry]**

In the geometric view of an ANN, what does each layer do?

- A. It maps the input feature space into another feature space where, ideally, classes become linearly separable for the next layer
- B. It sorts the training samples
- C. It computes the confusion matrix
- D. It normalises labels

<details><summary>Answer</summary>

**A.** In the end the output is a 1D problem where regression or classification is easy.

Why each option:

- A ✓ Correct.
- B ✗ No sorting.
- C ✗ That's evaluation.
- D ✗ Labels aren't normalised.

In depth: Each layer is a geometric transformation: a linear map (rotate, scale, project) plus a non-linear activation (bend, fold). Stacking them warps the input space so that classes which were tangled become linearly separable. In the end the output is a 1D problem, where everything from earlier weeks (regression, thresholding) applies.

</details>

**Q236. [Week 12: Revision: generative]**

Given p(x, y) = p(y|x)p(x), when is a classifier's decision uncertain even if p(red|x) is high?

- A. When p(x) is low: the input is far from the training data distribution
- B. When p(x) is high
- C. When p(y|x) equals 1 and p(x) equals 1
- D. Never; a high p(red|x) always means a certain decision

<details><summary>Answer</summary>

**A.** That's why learning p(x) (generative modelling) helps detect out-of-distribution inputs.

Why each option:

- A ✓ Correct.
- B ✗ High p(x) means typical data, which is reliable.
- C ✗ That's a certain decision.
- D ✗ Ignores p(x).

In depth: p(y|x) only describes which side of the boundary x is on. A point far from all training data can be deep on the red side (p(red|x) ≈ 1), yet p(x) ≈ 0 means the model has never seen anything like it, so the decision isn't trustworthy. Combining both (p(x, y) = p(y|x)p(x)) identifies certain decisions (high p(y|x) and high p(x)) vs uncertain ones. This motivates generative modelling for out-of-distribution detection.

</details>

**Q237. [Week 12: Revision: generative]**

Which pairing of method and way of learning p(x) is correct?

- A. GMM: maximum likelihood via EM; VAE: latent variables + approximate q(z|x), maximise ELBO; diffusion: learn the score (SDE); flow matching: learn a velocity field (ODE)
- B. GMM: ELBO; VAE: EM; diffusion: grid search; flow matching: k-means
- C. All four use a GAN discriminator
- D. None of them learn p(x)

<details><summary>Answer</summary>

**A.** That's the 'Finding p(x)' revision map in week 12.

Why each option:

- A ✓ Correct.
- B ✗ Scrambled.
- C ✗ Only GANs use discriminators.
- D ✗ All of them model p(x).

In depth: The 'Finding p(x)' map. Analytical model plus maximum likelihood: GMM with parameters μ, σ found by the EM method. Introduce latent variables and approximate p(z|x) with a network q(z|x), maximising the ELBO: VAE. Learn a vector field that is the gradient of log p(x) and follow it with a random walk (SDE): diffusion or score-based models. Start from a random location and follow a learned velocity field (ODE): flow matching.

</details>

**Q238. [Week 12: Revision: INR]**

Which summary of implicit neural representations is correct?

- A. An MLP can approximate any function; it can model a 2D image or 3D scan, the density and colour of a 3D scene from 2D projections (NeRF), or a sum of 3D Gaussians (splatting)
- B. INRs only work for text
- C. INRs require labelled classes
- D. INRs are a type of decision tree

<details><summary>Answer</summary>

**A.** That's the week 12 summary of week 10.

Why each option:

- A ✓ Correct.
- B ✗ INRs are for continuous signals.
- C ✗ They're fitted to signals, not classes.
- D ✗ They're neural networks.

In depth: The INR story in four steps: (1) an MLP can approximate any function (universal approximation); (2) overfit it to model a 2D image or 3D scan, f(x, y, z) → intensity; (3) NeRF models density and colour of a 3D scene, learned from a set of 2D photos via differentiable volume rendering; (4) Gaussian splatting replaces the dense MLP with a sum of 3D Gaussians placed only where matter is.

</details>

**Q239. [Week 12: Revision: key concepts]**

Which of the following is NOT a method to prevent overfitting in neural networks?

- A. Increasing the number of parameters while training longer on the same small dataset
- B. Dropout
- C. Data augmentation
- D. Early stopping on validation loss

<details><summary>Answer</summary>

**A.** More capacity and longer training on small data increases overfitting.

Why each option:

- A ✓ This increases overfitting, so it's the answer.
- B ✗ Prevents overfitting.
- C ✗ Prevents overfitting.
- D ✗ Prevents overfitting.

In depth: Overfitting happens when capacity is too large for the amount of data. Adding parameters and training longer on the same small dataset makes it worse. Dropout (random deactivation), data augmentation (more effective data), and early stopping (stop when validation loss rises) all reduce it, as do weight decay and more data.

</details>

**Q240. [Week 12: Revision: safety]**

Which list matches the three 'what could go wrong' failure modes repeated in weeks 1, 5 and 12?

- A. Unknown data (failing to generalise), data drift, and bias
- B. Learning rate, batch size, and epochs
- C. Convolution, pooling, and stride
- D. Precision, recall, and F1

<details><summary>Answer</summary>

**A.** 1) New data unlike training, 2) the data changes after deployment, 3) wrong mapping for subgroups.

Why each option:

- A ✓ Correct.
- B ✗ Training hyperparameters.
- C ✗ CNN components.
- D ✗ Performance metrics.

In depth: The three failure modes: (1) unknown data: a test case unlike any training data (failure to generalise, not enough data); (2) data drift: the distribution changes after deployment (new camera, new circumstances); (3) bias: the mapping is wrong for a subgroup (e.g. under-represented patients). They tie to safety (Week 5): calibration, OOD detection, monitoring and fairness evaluation.

</details>

**Q241. [Week 12: Revision: transformers]**

Text, images and other data can all be processed by the same transformer model. What is needed to make this possible?

- A. Convert the data into a sequence of tokens (words, patches…) projected to embeddings with positional encoding
- B. Convert all data to grey-scale images
- C. Use a separate decision tree for each data type
- D. Remove the attention mechanism

<details><summary>Answer</summary>

**A.** Words become tokens, images become patches. The same attention blocks then apply.

Why each option:

- A ✓ Correct.
- B ✗ Not needed.
- C ✗ Not how it works.
- D ✗ Attention is the core.

In depth: Transformers operate on a sequence of vectors. Text: tokens, then embeddings plus 1D positional encoding. Images: patches with a linear projection plus 2D positional encoding (ViT). Audio and video: spectrogram or space-time patches. Once data is a token sequence, the same attention blocks apply, which is why multimodal models can mix text and image tokens.

</details>

**Q242. [Week 12: Revision: CNN]**

Which set lists ANN/CNN key concepts that are examinable?

- A. Backpropagation, overfitting, dropout, convolution, pooling/stride, batch norm, ResNet/UNet, transfer learning
- B. Only the history of AI
- C. Only Python syntax
- D. Only GAN training

<details><summary>Answer</summary>

**A.** The week 12 revision slide groups these as ANN and CNN key concepts, plus adversarial and backdoor attacks, saliency maps and responsible AI.

Why each option:

- A ✓ Correct.
- B ✗ Far more is examinable.
- C ✗ Concepts, not syntax.
- D ✗ Too narrow.

In depth: The revision slide groups key ANN/CNN concepts: backpropagation, gradient descent and optimisers (momentum, Adam), activation functions, overfitting and dropout, convolution, pooling and stride, batch norm and layer norm, classic networks (convnet, VGG, ResNet, UNet), and transfer learning, distillation, self-supervision and semi-supervision. Safety topics: adversarial attacks, backdoor attacks, saliency maps and responsible AI. Use it as a checklist for your one-page summary.

</details>

**Q243. [Week 12: Revision: unsupervised]**

In unsupervised k-means, each cluster is represented by:

- A. One point at its centre; each data point is assigned to the closest centre, so the space is divided by straight lines
- B. A full Gaussian with covariance
- C. A set of axis-aligned conditions
- D. A line with a buffer zone

<details><summary>Answer</summary>

**A.** That's the simplest cluster model. GMM generalises it with Gaussians (shape and soft membership).

Why each option:

- A ✓ Correct.
- B ✗ That's a GMM.
- C ✗ That's a decision tree.
- D ✗ That's an SVM.

In depth: k-means represents each cluster by a single centroid (the mean of its points). Each data point belongs to its nearest centroid, so the boundaries between clusters are the perpendicular bisectors between centroids: straight lines that divide the space into Voronoi cells. That's why k-means struggles with elongated or curved clusters, and why GMMs (full covariance, soft assignment) generalise it.

</details>
