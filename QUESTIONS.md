# IFN680 Exam Practice: Question Bank

243 multiple choice questions generated from the IFN680 lectures (weeks 1–12), using the weekly review questions and the practice exam / Week 12 mock quiz as a guide.

Interactive version: open `index.html` (GitHub Pages).

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

</details>

**Q2. [Week 1: System 1 / System 2]** · _Practice exam_

'Thinking, Fast and Slow' (2011) describes how the brain works using two cognitive systems. Which statement is true?

- A. Brain System 1 is slower than System 2
- B. Brain System 1 is more logical than System 2
- C. Brain System 1 is more emotional than System 2
- D. Brain System 1 is more multitask than System 2

<details><summary>Answer</summary>

**D.** System 1 is automated, unconscious, relatively fast and multitask, and it solves known problems. System 2 is conscious reasoning: slow, one task at a time.

</details>

**Q3. [Week 3: ML vs DL]** · _Practice exam_

What is the difference between machine learning and deep learning?

- A. Deep learning is a subset of machine learning that involves neural networks with more than one hidden layer
- B. Deep learning is a type of machine learning that uses only supervised learning during training
- C. Deep learning is a type of machine learning that has neural networks with very wide layers
- D. There is no difference between machine learning and deep learning

<details><summary>Answer</summary>

**A.** 'Deep' refers to depth: multiple hidden layers stacked on top of each other.

</details>

**Q4. [Week 3: ANN]** · _Week 12 mock quiz (More practice)_

Which of the following statements about Artificial Neural Networks (ANNs) is correct?

- A. ANNs are only capable of linear transformations and cannot model non-linear relationships without external preprocessing
- B. The activation function in a neuron is applied to the input features before computing the weighted sum
- C. During backpropagation, the weights are updated using the chain rule to compute gradients, and optimisers like SGD are used to minimise the loss
- D. A neural network with more hidden layers will always perform better than a shallower network, regardless of the task or dataset size

<details><summary>Answer</summary>

**C.** A is wrong: non-linear activations let ANNs model non-linear relationships. B is wrong: z = W·x + b comes first, then a = σ(z). D is wrong: deeper networks can overfit, suffer vanishing or exploding gradients, and cost more.

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

</details>

**Q6. [Week 4: Conv output size]** · _Week 12 mock quiz Q10 (Practice exam)_

You are designing a CNN for image classification. Input images are 64×64 with 3 colour channels. Layers: (1) Conv 16 filters 3×3, stride 1, padding 1; (2) ReLU; (3) MaxPool 2×2 stride 2; (4) Conv 32 filters 3×3, stride 1, padding 1; (5) ReLU; (6) MaxPool 2×2 stride 2. What is the shape of the output after the second max-pooling layer?

- A. [16, 16, 16]
- B. [32, 16, 16]
- C. [32, 32, 32]
- D. [32, 8, 8]

<details><summary>Answer</summary>

**B.** [3,64,64] → conv → [16,64,64] → pool → [16,32,32] → conv → [32,32,32] → pool → [32,16,16].

</details>

**Q7. [Week 5: Metrics]** · _Practice exam_

In a medical diagnosis task with high class imbalance (rare disease), which metric would be the most appropriate to evaluate model performance?

- A. Accuracy
- B. Sensitivity and specificity
- C. Mean squared error
- D. Training loss

<details><summary>Answer</summary>

**B.** With 1% prevalence, predicting 'healthy' for everyone gives 99% accuracy. Sensitivity and specificity measure each type of error separately.

</details>

**Q8. [Week 6: Semi-supervised]** · _Practice exam_

Which of the following statements about semi-supervised learning is correct?

- A. Semi-supervised learning leverages both labelled and unlabelled data to improve model performance
- B. Semi-supervised learning uses only labelled data to train models, ignoring unlabelled data entirely
- C. Semi-supervised learning assumes that unlabelled data carries no useful information and is primarily used to augment small labelled datasets with synthetic data
- D. Semi-supervised learning is only applicable to classification tasks and cannot be used for regression

<details><summary>Answer</summary>

**A.** It's suitable when labels are scarce or expensive but unlabelled data is plentiful.

</details>

**Q9. [Week 7: LSTM]** · _Practice exam_

What problem do LSTM networks primarily solve compared to vanilla RNNs?

- A. Faster training speed
- B. Lower memory requirements
- C. Elimination of the need for activation functions
- D. Vanishing gradient problem for long sequences

<details><summary>Answer</summary>

**D.** The LSTM cell state, regulated by gates, keeps memory over longer sequences and is more robust to vanishing gradients.

</details>

**Q10. [Week 7: Attention concepts]** · _Week 12 mock quiz Q12 (Practice exam)_

Consider 4 concepts: 1. Multi-Head Attention, 2. Value (V), 3. Self-attention, 4. Causal Masking. And 4 statements: A. Each token in a sequence attends to all other tokens in the same sequence. B. Ensures each token can only attend to previous tokens in autoregressive models. C. Determines how much each value contributes to the final output based on attention scores. D. Splits the input into multiple smaller attention heads to capture different relationships in parallel. Which combination is correct?

- A. 1-D, 2-C, 3-A, 4-B
- B. 1-A, 2-B, 3-C, 4-D
- C. 1-D, 2-B, 3-A, 4-C
- D. 1-B, 2-D, 3-C, 4-A

<details><summary>Answer</summary>

**A.** Multi-head attention splits into heads (D). Values are what gets weighted and mixed (C). Self-attention is all-to-all within the sequence (A). Causal masking blocks future tokens (B).

</details>

**Q11. [Week 8: PCA]** · _Practice exam_

Considering the principal component analysis method, which statement is NOT true?

- A. Each component direction is perpendicular to every other one
- B. The sum of the variance of all the components should be equal to the total variance of the data
- C. Each component direction is an eigenvector of the covariance matrix
- D. One has to use singular value decomposition (SVD) to compute a PCA

<details><summary>Answer</summary>

**D.** SVD is one way to compute PCA, but you can also do an eigendecomposition of the covariance matrix. It is not required.

</details>

**Q12. [Week 9: High dimension]** · _Practice exam_

Considering a D-dimensional multivariate normal distribution (D > 100), which statement is NOT true?

- A. The norm is expected to be on average sqrt(D)
- B. Two random samples would be orthogonal
- C. The vectors are most likely to be aligned with the orthonormal axes
- D. It is almost impossible to sample a vector with a norm lower than 1

<details><summary>Answer</summary>

**C.** In high dimension, samples lie in a thin shell of radius ≈ √D and random pairs are almost perpendicular. They are not aligned with the axes.

</details>

**Q13. [Week 9: Diffusion]** · _Practice exam_

Which of the following statements about diffusion models is correct?

- A. Diffusion models generate data by iteratively adding noise to a random sample until it matches the target distribution
- B. The forward process gradually adds Gaussian noise to the data over a series of timesteps, while the reverse process learns to denoise the data
- C. Diffusion models are trained using a single-step process where the model directly predicts the final denoised sample from pure noise
- D. Diffusion models are only applicable to discrete data (e.g. text) and cannot be used for continuous data like images or audio

<details><summary>Answer</summary>

**B.** The forward process (an encoder with no training) adds noise. The reverse process is learned and denoises iteratively.

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

</details>

**Q16. [Week 3: Parameters]** · _Mock-style_

How many learnable parameters does an MLP with 2 inputs, one hidden layer of 10 units and 1 output have (with biases)?

- A. 41
- B. 30
- C. 20
- D. 31

<details><summary>Answer</summary>

**A.** Hidden layer: 2·10 + 10 = 30. Output layer: 10·1 + 1 = 11. Total 41.

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

</details>

**Q18. [Week 4: Conv output size]** · _Mock-style_

Input [3,128,128]. Apply: (1) Conv 8 filters 3×3, stride 2, padding 1; (2) ReLU; (3) MaxPool 2×2 stride 2. What is the output shape?

- A. [8, 32, 32]
- B. [8, 64, 64]
- C. [3, 32, 32]
- D. [8, 16, 16]

<details><summary>Answer</summary>

**A.** Conv: floor((128 + 2 − 3)/2) + 1 = 64, giving [8,64,64]. ReLU doesn't change the shape. Pool halves it to [8,32,32].

</details>

**Q19. [Week 4: Parameters]** · _Mock-style_

The feature map [32,16,16] is flattened and fed to nn.Linear(…, 10). How many parameters does this linear layer have?

- A. 81,930
- B. 81,920
- C. 8,202
- D. 5,130

<details><summary>Answer</summary>

**A.** 32·16·16 = 8,192 inputs. 8,192 × 10 weights + 10 biases = 81,930.

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

</details>

**Q21. [Week 5: Metrics]** · _Mock-style_

A classifier on 100 healthy patients gives TN = 90 and FP = 10, and on 20 sick patients gives TP = 15 and FN = 5. What are sensitivity and specificity?

- A. Sensitivity = 0.75, Specificity = 0.90
- B. Sensitivity = 0.90, Specificity = 0.75
- C. Sensitivity = 0.60, Specificity = 0.90
- D. Sensitivity = 0.875, Specificity = 0.75

<details><summary>Answer</summary>

**A.** Sensitivity = TP/(TP+FN) = 15/20 = 0.75. Specificity = TN/(TN+FP) = 90/100 = 0.90.

</details>

**Q22. [Week 6: Clustering]** · _Mock-style_

Which statement about k-means is NOT true?

- A. K is a hyperparameter chosen by the user
- B. It can converge to a local minimum depending on initialisation
- C. It requires labelled data to compute the centroids
- D. Each point is assigned to its closest centroid

<details><summary>Answer</summary>

**C.** k-means is unsupervised. kNN is the method that needs labels.

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

</details>

**Q25. [Week 7: Attention]** · _Mock-style_

A transformer processes a sequence of T = 9 tokens with embedding dimension 16. What is the size of the attention matrix A = softmax(QKᵀ) for one head?

- A. 9 × 9
- B. 16 × 16
- C. 9 × 16
- D. 16 × 9

<details><summary>Answer</summary>

**A.** Q and K are T × D, so QKᵀ is T × T: one weight per pair of tokens. Its cost grows quadratically with T.

</details>

**Q26. [Week 7: Multi-head]** · _Mock-style_

An embedding dimension of 768 is split into 12 attention heads. What is the dimension per head?

- A. 64
- B. 12
- C. 768
- D. 9,216

<details><summary>Answer</summary>

**A.** 768 / 12 = 64. The heads' outputs are concatenated back to 768.

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

</details>

**Q28. [Week 9: GAN]** · _Mock-style_

Which statement about GANs is NOT true?

- A. The discriminator is trained to output 1 for real images and 0 for fake images
- B. The generator maps a random latent vector to a sample
- C. The discriminator is required at generation (inference) time
- D. A conditional GAN feeds a condition such as the class to the generator

<details><summary>Answer</summary>

**C.** After training, only the generator is needed to create data. The discriminator only drives training.

</details>

**Q29. [Week 9: Flow matching]** · _Mock-style_

Which statement about flow matching is NOT true?

- A. It learns a velocity field between noise and data
- B. Sampling solves an ODE with an ODE solver
- C. Sampling requires a random walk with Langevin dynamics
- D. Several steps are needed because the learnt velocity field is non-linear

<details><summary>Answer</summary>

**C.** Langevin random walks are used by score-based diffusion. Flow matching follows a deterministic ODE.

</details>

**Q30. [Week 9: High dimension]** · _Mock-style_

Samples are drawn from an isotropic normal distribution N(0, I) in D = 400 dimensions. What is the typical norm of a sample?

- A. About 20
- B. About 1
- C. About 400
- D. About 0

<details><summary>Answer</summary>

**A.** The norm concentrates around √D = √400 = 20, and random pairs are nearly orthogonal.

</details>

**Q31. [Week 10: NeRF encoding]** · _Mock-style_

In NeRF, the viewing direction is a 3D unit vector encoded with L = 4 frequencies (sin and cos). What is the encoded dimension?

- A. 24
- B. 8
- C. 12
- D. 60

<details><summary>Answer</summary>

**A.** 3 components × 2L = 3 × 8 = 24. For position, L = 10 gives 60.

</details>

**Q32. [Week 10: Gaussian splatting]** · _Mock-style_

Which statement about Gaussian splatting is NOT true?

- A. It represents the scene densely with an MLP queried at every point in space
- B. Projected 3D Gaussians become 2D Gaussians on the image
- C. It is typically initialised from a Structure-from-Motion point cloud
- D. Gaussians are cloned and split during optimisation

<details><summary>Answer</summary>

**A.** Dense MLP queries are NeRF. Splatting is sparse: Gaussians only where density is non-zero.

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

</details>

**Q35. [Week 1: Classification]**

Which of the following is a classification task?

- A. Predicting whether a chest X-ray shows pneumonia or not
- B. Predicting next year's krill density from water temperature
- C. Predicting the price of a house in dollars
- D. Predicting tomorrow's temperature in degrees

<details><summary>Answer</summary>

**A.** Classification predicts a discrete class label. The other three predict a continuous value, which makes them regression.

</details>

**Q36. [Week 1: Regression]**

Which of the following is a regression task?

- A. Estimating a patient's heart rate (beats per minute) from a video
- B. Deciding if a loan should be approved or denied
- C. Identifying which of 1000 object classes is in an image
- D. Detecting if an email is spam

<details><summary>Answer</summary>

**A.** Regression outputs a continuous number. Loan approval, object recognition and spam detection all output a class.

</details>

**Q37. [Week 1: GOFAI vs connectionism]**

What is the main difference between classic AI (GOFAI, symbolism) and modern AI (machine learning, connectionism)?

- A. In GOFAI a human expert writes the rules or analytical model; in ML the machine learns parameters of a generic model from data
- B. GOFAI uses neural networks while ML uses decision rules
- C. GOFAI requires large datasets while ML requires none
- D. There is no difference; they are two names for the same approach

<details><summary>Answer</summary>

**A.** Symbolic AI encodes human-written logic and physics models. Connectionism learns from data, which is what made it take off once computers became fast and data became plentiful.

</details>

**Q38. [Week 1: Supervised vs unsupervised]**

Which statement correctly contrasts supervised and unsupervised learning?

- A. Supervised learning uses labelled input-output pairs; unsupervised learning finds structure (e.g. clusters) in unlabelled data
- B. Supervised learning needs no data; unsupervised learning needs labelled data
- C. Unsupervised learning always gives higher accuracy than supervised learning
- D. Supervised learning can only do regression, unsupervised can only do classification

<details><summary>Answer</summary>

**A.** Labels are the key difference. Labelling is expensive, and it's a big business, which is why unsupervised and self-supervised methods matter.

</details>

**Q39. [Week 1: System 1 / System 2]**

According to the lecture, which describes Brain System 2?

- A. Conscious thinking based on reasoning that can solve unknown problems, but is relatively slow and does one task at a time
- B. Automated unconscious actions learned through repetition
- C. Exact and vast memory that always gives the same answer
- D. A fast multitask system that only solves known problems

<details><summary>Answer</summary>

**A.** System 2 is conscious reasoning. Current ML is closer to System 1: with enough data it learns skills, but we don't know how to make a machine reason about problems it has never seen.

</details>

**Q40. [Week 1: Data drift]**

An AI system works well at deployment, but months later the hospital replaces its camera and accuracy drops. This is an example of:

- A. Data drift
- B. Bias
- C. Overfitting to the test set
- D. Underfitting

<details><summary>Answer</summary>

**A.** Data drift: over time the data changes (new camera, new circumstances), so inputs map to different locations than the ones learned during training.

</details>

**Q41. [Week 1: Generalisation]**

A new patient's scan does not resemble anything seen during training, and the model gives a wrong answer. Which failure mode is this?

- A. Lack of generalisation (unknown data / not enough training data)
- B. Data drift caused by changing equipment
- C. Label noise
- D. Vanishing gradients

<details><summary>Answer</summary>

**A.** 'Unknown data': the model fails to generalise because the training data did not cover this case.

</details>

**Q42. [Week 1: Bias]**

A cancer model was trained with healthy white volunteers and white cancer patients. Black patients with cancer get mapped close to healthy white volunteers and are missed. What is this an example of?

- A. Bias
- B. Data drift
- C. Adversarial attack
- D. Backdoor attack

<details><summary>Answer</summary>

**A.** Bias: the learned mapping is wrong for a subgroup that was under-represented in training.

</details>

**Q43. [Week 1: ML as maps]**

In the lecture's 'machine learning = creating maps' view, what is an embedding / latent variable?

- A. The location (vector of numbers) that the model maps an input to, where similar inputs end up close together
- B. The raw pixel values of the input image
- C. The label provided by a human annotator
- D. The learning rate used during training

<details><summary>Answer</summary>

**A.** Layers of additions and multiplications map, for example, 36 million pixel numbers to a location on a map. That location is the embedding, or latent variable.

</details>

**Q44. [Week 1: Generative AI]**

In the 'map' view of ML, how does a chatbot such as ChatGPT generate text?

- A. It guesses the next location (token) on the map given the previous ones
- B. It retrieves complete answers from a database of human-written responses
- C. It runs hand-written grammar rules
- D. It clusters the question with k-means

<details><summary>Answer</summary>

**A.** LLMs are autoregressive: they predict the next token's location on the map, append it, and repeat.

</details>

**Q45. [Week 1: History]**

What triggered the 2012 paradigm shift in AI?

- A. Convolutional neural networks trained on ImageNet using GPUs: data-driven models with no hand-crafted model
- B. The invention of decision trees
- C. The invention of symbolic logic programming
- D. The first chess program beating a human

<details><summary>Answer</summary>

**A.** In 2012 CNNs won the ImageNet challenge (14M images, 20,000+ categories). Compute (GPUs) and plenty of data made data-driven models win.

</details>

**Q46. [Week 1: AGI / ASI]**

Which statement best distinguishes AGI from ASI?

- A. AGI is human-level general intelligence across many tasks; ASI would surpass human intelligence
- B. AGI is narrow AI for one task; ASI is AI for two tasks
- C. AGI and ASI are two image classification benchmarks
- D. ASI is a weaker version of AGI that runs on phones

<details><summary>Answer</summary>

**A.** Artificial General Intelligence means human-level ability across tasks. Artificial Super Intelligence means beyond human. Open questions include whether scaling alone can get there, and the ethics.

</details>

**Q47. [Week 1: Intelligence]**

Which point did the lecture raise when questioning whether current AI is truly 'intelligent'?

- A. The human brain uses about 20 W and can learn new skills from few examples, whereas AI needs huge data and energy
- B. AI cannot perform any useful task
- C. Computers have too little memory to store a neural network
- D. AI systems always explain their reasoning

<details><summary>Answer</summary>

**A.** Intelligence questions include learning new skills on its own, solving never-seen problems, and efficiency: the brain runs on ~20 W.

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

</details>

**Q49. [Week 2: Correlation]**

Why do we check data correlation (e.g. Pearson's coefficient) before performing linear regression?

- A. To check there is a linear relationship between input and output worth modelling
- B. To compute the learning rate
- C. To remove the need for a test set
- D. Because regression only works on uncorrelated data

<details><summary>Answer</summary>

**A.** Pearson's correlation measures linear correlation. Spearman's measures monotonic, possibly non-linear, correlation.

</details>

**Q50. [Week 2: Correlation]**

Which coefficient should you use to detect a non-linear but monotonic relationship?

- A. Spearman's rank correlation
- B. Pearson's correlation
- C. Mean squared error
- D. Cross-entropy

<details><summary>Answer</summary>

**A.** Spearman's correlation works on ranks, so it captures any monotonic relationship, not only linear ones.

</details>

**Q51. [Week 2: OLS]**

What is the ordinary least squares (OLS) solution for the parameters W in the linear system AW = B?

- A. W = (AᵀA)⁻¹AᵀB
- B. W = A⁻¹Bᵀ
- C. W = AᵀB(AAᵀ)
- D. W = (A + B)⁻¹

<details><summary>Answer</summary>

**A.** Setting the derivative of the squared error to 0 gives the normal equations AᵀAW = AᵀB, so W = (AᵀA)⁻¹AᵀB (the pseudo-inverse).

</details>

**Q52. [Week 2: Linear regression]**

For the model y = mx + c, what are the parameters and how are they found?

- A. m and c, found by minimising the error between predictions and ground truth (e.g. least squares)
- B. x and y, found by grid search on the test set
- C. The learning rate and number of epochs, found analytically
- D. The polynomial order, found by OLS

<details><summary>Answer</summary>

**A.** Parameters (m, c) are chosen to minimise the residual error. This can be done analytically (OLS), by grid search, or by gradient descent.

</details>

**Q53. [Week 2: Assumptions]**

Which is NOT one of the four main assumptions of linear regression?

- A. The input features must follow a uniform distribution
- B. Linearity between input and output
- C. Independence of the residuals
- D. Homoscedasticity (constant variance of residuals)

<details><summary>Answer</summary>

**A.** The four assumptions are linearity, independence, homoscedasticity, and normality of residuals (centred at 0). Nothing requires uniform inputs.

</details>

**Q54. [Week 2: Assumptions]**

What does homoscedasticity mean in linear regression?

- A. The residuals have constant variance at every level of the input
- B. The residuals follow a uniform distribution
- C. The inputs are all on the same scale
- D. The output is always positive

<details><summary>Answer</summary>

**A.** Homoscedasticity is constant residual variance. If the spread of errors grows with x, the assumption is violated.

</details>

**Q55. [Week 2: Polynomial regression]**

How can polynomial regression (e.g. y = w₀ + w₁x + w₂x²) be solved as an ordinary least squares problem?

- A. Build a matrix A whose columns are 1, x, x², …; the model is still linear in the parameters, so W = (AᵀA)⁻¹AᵀB applies
- B. It cannot. Polynomial regression always needs gradient descent
- C. Take the log of the output and use a decision tree
- D. Use k-means to find the coefficients

<details><summary>Answer</summary>

**A.** The model is non-linear in x but linear in the weights, so the same OLS solution works with polynomial feature columns.

</details>

**Q56. [Week 2: Polynomial regression]**

What is the difference between a linear regression model and a 1st order polynomial regression model?

- A. There is no difference; both are y = w₀ + w₁x
- B. The 1st order polynomial includes an x² term
- C. Linear regression has no intercept
- D. The polynomial model must be solved with gradient descent

<details><summary>Answer</summary>

**A.** A 1st order polynomial is a straight line, which is exactly linear regression.

</details>

**Q57. [Week 2: Hyperparameters]**

What is the hyperparameter of a polynomial regression model?

- A. The polynomial order (degree)
- B. The coefficients w₀…w_N
- C. The residuals
- D. The ground-truth values

<details><summary>Answer</summary>

**A.** The order is chosen with validation data. The coefficients are parameters fitted on training data.

</details>

**Q58. [Week 2: Overfitting]**

How can overfitting be identified?

- A. Error is low on the training data but much higher on validation (unseen) data
- B. Error is high on both training and validation data
- C. Training error keeps increasing
- D. Validation error is lower than training error

<details><summary>Answer</summary>

**A.** An overfitted model (e.g. a high-order polynomial) memorises training points and fails to generalise. Compare performance on data not used for fitting.

</details>

**Q59. [Week 2: Experimental design]**

In best-practice experimental design, what is the validation set used for?

- A. Selecting hyperparameters (e.g. polynomial order, K in kNN)
- B. Reporting the final performance of the model
- C. Fitting the model parameters
- D. Nothing; it is optional decoration

<details><summary>Answer</summary>

**A.** Training data fits parameters, validation data selects hyperparameters, and testing data (kept secret, ideally from another source) estimates final performance.

</details>

**Q60. [Week 2: Gradient descent]**

Gradient descent on the krill data (one variable ~0.05, another ~30) did not follow the steepest slope. What was the easy fix?

- A. Normalise the data so variables have a similar scale
- B. Increase the polynomial order
- C. Use the test set for training
- D. Remove the bias term

<details><summary>Answer</summary>

**A.** With different scales, the gradient is dominated by one variable. After normalisation, gradient descent converges quickly.

</details>

**Q61. [Week 2: Grid search]**

How does grid search find the parameters of a model?

- A. It evaluates the loss for every combination of parameter values on a predefined grid and keeps the best
- B. It follows the negative gradient of the loss
- C. It solves the normal equations analytically
- D. It randomly removes neurons

<details><summary>Answer</summary>

**A.** Grid search is exhaustive and simple, but its cost grows exponentially with the number of parameters.

</details>

**Q62. [Week 2: kNN]**

Which sequence describes the K Nearest Neighbours classifier?

- A. Normalise data → store training data → compute distance to training points → take the K closest → majority vote
- B. Initialise K centroids → assign points → update centroids → repeat
- C. Fit a line → compute residuals → apply softmax
- D. Split on the feature that best separates classes → recurse

<details><summary>Answer</summary>

**A.** kNN stores the training data and classifies by majority among the K nearest points. The second option describes k-means, which is unsupervised.

</details>

**Q63. [Week 2: kNN]**

Which is a limitation of KNN?

- A. Prediction is slow and memory-heavy for large datasets because all training data must be stored and compared
- B. It cannot be interpreted at all
- C. It requires gradient descent to train
- D. It can only be used for regression

<details><summary>Answer</summary>

**A.** kNN is easy to interpret, but it is slow at inference, sensitive to feature scale, and suffers in high dimension. It works for classification and regression.

</details>

**Q64. [Week 2: kNN]**

How should the value of K in KNN be chosen?

- A. Check performance for a range of K values on the validation dataset
- B. Always use K = 1
- C. Choose K that maximises accuracy on the test set
- D. K must equal the number of features

<details><summary>Answer</summary>

**A.** K is a hyperparameter, so it's selected on validation data, never on the test set.

</details>

**Q65. [Week 2: kNN regression]**

How can KNN be used for a regression task?

- A. Predict the average (or weighted average) of the target values of the K nearest neighbours
- B. It cannot be used for regression
- C. Use the majority class of the neighbours
- D. Fit a decision tree on the neighbours

<details><summary>Answer</summary>

**A.** For regression, replace the majority vote with an average of the neighbours' values.

</details>

**Q66. [Week 2: Decision trees]**

Which is an advantage of decision trees?

- A. They are interpretable: the decision-making follows explicit rules (e.g. income < $35,000)
- B. They never overfit
- C. They always produce smooth, curved boundaries
- D. They require no training data

<details><summary>Answer</summary>

**A.** Trees are interpretable and handle mixed feature types, but they overfit easily and produce axis-aligned boundaries.

</details>

**Q67. [Week 2: Pruning]**

How can overfitting in decision trees be prevented?

- A. Pruning: limit tree growth (max depth, min samples per leaf) or remove branches that do not improve validation performance
- B. Grow the tree until every leaf has one sample
- C. Increase the learning rate
- D. Use more features at every split

<details><summary>Answer</summary>

**A.** Pre-pruning stops growth early. Post-pruning grows the full tree and then removes branches that don't help on validation data.

</details>

**Q68. [Week 2: Random forest]**

What is the difference between a decision tree and a random forest?

- A. A random forest is an ensemble of many trees trained on random subsets of data/features whose predictions are combined by voting
- B. A random forest is a single, very deep tree
- C. A random forest uses neural networks at each node
- D. There is no difference

<details><summary>Answer</summary>

**A.** Several are better than one: averaging many diverse trees reduces overfitting and variance.

</details>

**Q69. [Week 2: Confusion matrix]**

What does a confusion matrix show?

- A. Counts of predictions for each (ground-truth class, predicted class) pair; the diagonal holds correct predictions
- B. The loss value at each epoch
- C. The correlation between features
- D. The weights of a linear model

<details><summary>Answer</summary>

**A.** To build it, run the model on a dataset and count how often each true class is predicted as each class. Off-diagonal cells are the different types of errors.

</details>

**Q70. [Week 2: Simpson's paradox]**

Simpson's paradox is listed as a limitation for linear regression. What is it?

- A. A trend that appears in several groups of data disappears or reverses when the groups are combined
- B. Gradient descent diverging when the learning rate is too high
- C. The residuals not being normally distributed
- D. Having more features than samples

<details><summary>Answer</summary>

**A.** Pooling sub-groups can hide or reverse a relationship, so a single linear fit over everything can mislead.

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

</details>

**Q72. [Week 3: Class score]**

How is the class score of a linear classifier related to the decision boundary?

- A. The score is the (scaled) signed distance of the input from the boundary: positive on one side, negative on the other
- B. The score is the number of misclassified samples
- C. The score is always between 0 and 1
- D. The score is unrelated to the parameters

<details><summary>Answer</summary>

**A.** A neuron projects the input onto w (dot product). The sign gives the class and the magnitude grows with distance from the boundary.

</details>

**Q73. [Week 3: Loss functions]**

What is the purpose of a loss function?

- A. To give a single number measuring how good the current parameters are, which we minimise to train the model
- B. To initialise the weights
- C. To choose the number of layers
- D. To normalise the input data

<details><summary>Answer</summary>

**A.** The loss (cost, objective) quantifies prediction error. Optimisation then finds the parameters that minimise it.

</details>

**Q74. [Week 3: Loss functions]**

Which loss functions are most commonly used for classification and regression respectively?

- A. Cross-entropy for classification; mean squared error for regression
- B. Mean squared error for classification; hinge loss for regression
- C. Accuracy for classification; R² for regression
- D. KL divergence for both

<details><summary>Answer</summary>

**A.** Cross-entropy (softmax loss) and hinge loss are used for classification. MSE is used for regression.

</details>

**Q75. [Week 3: Hinge vs cross-entropy]**

What is a key difference between hinge loss and cross-entropy loss?

- A. Hinge loss becomes zero once the correct class score exceeds others by a margin; cross-entropy keeps pushing the correct class probability towards 1
- B. Cross-entropy is only for regression
- C. Hinge loss requires a softmax
- D. They are mathematically identical

<details><summary>Answer</summary>

**A.** Hinge (multiclass SVM) loss sums over incorrect classes with a margin. Cross-entropy is the negative log-likelihood of the softmax probability of the correct class.

</details>

**Q76. [Week 3: Softmax]**

What does the softmax function do?

- A. Normalises a vector of class scores into probabilities that are positive and sum to 1
- B. Sets all negative values to zero
- C. Selects the largest weight in the network
- D. Computes the gradient of the loss

<details><summary>Answer</summary>

**A.** softmax(sᵢ) = exp(sᵢ)/Σexp(sⱼ). It approximates a max function and turns scores into a probability distribution.

</details>

**Q77. [Week 3: Gradient]**

How is the gradient of the loss used to update model parameters in gradient descent?

- A. Parameters move a small step (learning rate) in the direction opposite to the gradient: w ← w − η∇L
- B. Parameters move in the direction of the gradient to increase the loss
- C. The gradient is added to the input data
- D. The gradient replaces the parameters directly

<details><summary>Answer</summary>

**A.** The gradient points uphill, so we step downhill, scaled by the learning rate.

</details>

**Q78. [Week 3: GD variants]**

What distinguishes gradient descent, stochastic gradient descent and mini-batch gradient descent?

- A. GD uses the whole dataset per update, SGD uses one sample, mini-batch uses a small batch of samples
- B. GD uses one sample, SGD uses the whole dataset, mini-batch uses none
- C. They differ only in the activation function
- D. SGD has no learning rate

<details><summary>Answer</summary>

**A.** When data doesn't fit in memory, updates are computed on samples or batches. Mini-batch is the usual compromise.

</details>

**Q79. [Week 3: Momentum]**

What does adding momentum to gradient descent achieve?

- A. Like a heavy ball rolling in a canyon, it smooths updates, reducing oscillations and accelerating along consistent directions
- B. It guarantees the global minimum is found
- C. It removes the need for a loss function
- D. It sets the learning rate to zero at the end of training

<details><summary>Answer</summary>

**A.** Momentum keeps an exponential moving average of past gradients, which damps oscillation across a narrow valley.

</details>

**Q80. [Week 3: Adam]**

What is the main difference between plain SGD and Adam?

- A. Adam combines momentum (moving average of gradients) with an adaptive per-parameter learning rate (moving average of squared gradients, as in RMSProp)
- B. Adam does not use gradients
- C. Adam uses a fixed learning rate for every parameter with no momentum
- D. Adam only works for linear regression

<details><summary>Answer</summary>

**A.** Adam has two momentum parameters (β₁ for the first moment, β₂ for the second moment, typically 0.9 and 0.999). It is the most used optimiser.

</details>

**Q81. [Week 3: Hyperparameters]**

Which definition is correct?

- A. An epoch is one full pass over the training dataset
- B. Batch size is the number of layers in the network
- C. The learning rate is the number of epochs
- D. An epoch is one parameter update

<details><summary>Answer</summary>

**A.** Batch size is the number of samples per update. The learning rate is the step size. An epoch is one pass through all training samples.

</details>

**Q82. [Week 3: Learning rate]**

What happens when the learning rate is too small or too large?

- A. Too small: training takes many steps; too large: updates overshoot and the loss may oscillate or diverge
- B. Too small: the model overfits immediately; too large: training is perfectly stable
- C. The learning rate has no effect on training
- D. Too large always finds the global minimum faster

<details><summary>Answer</summary>

**A.** A common strategy is a large learning rate at the start to get close quickly, then decrease it to settle in the minimum.

</details>

**Q83. [Week 3: Neuron]**

What is the difference between a linear classifier and an artificial neuron?

- A. A neuron is a linear classifier (w·x + b) followed by a non-linear activation function
- B. A neuron has no weights
- C. A linear classifier uses backpropagation while a neuron does not
- D. There is no relationship between them

<details><summary>Answer</summary>

**A.** One neuron gives one linear boundary. The activation gives a smoother transition and lets stacked layers represent non-linear functions.

</details>

**Q84. [Week 3: Activation functions]**

What happens to a multi-layer neural network with no activation functions?

- A. It collapses to a single linear transformation, no more expressive than one linear layer
- B. It becomes a universal approximator
- C. It can only output zeros
- D. It trains faster and represents more complex functions

<details><summary>Answer</summary>

**A.** A product of matrices is a matrix. Non-linear activations make each layer a non-linear transformation.

</details>

**Q85. [Week 3: Activation functions]**

Which three are common activation functions for neural networks?

- A. ReLU, sigmoid, tanh
- B. Softmax, MSE, cross-entropy
- C. Adam, SGD, RMSProp
- D. Dropout, batch norm, pooling

<details><summary>Answer</summary>

**A.** ReLU, sigmoid and tanh are classic activations. The other options are losses, optimisers, and layers.

</details>

**Q86. [Week 3: Vanishing gradient]**

Why can sigmoid activations cause vanishing gradients in deep networks?

- A. The sigmoid saturates for large |x|, giving near-zero derivatives that shrink further when multiplied through many layers
- B. The sigmoid has a derivative of exactly 1 everywhere
- C. Sigmoid outputs negative values only
- D. Sigmoid cannot be differentiated

<details><summary>Answer</summary>

**A.** Backpropagation multiplies derivatives through the chain rule. Many small factors make the gradient vanish for early layers.

</details>

**Q87. [Week 3: Backpropagation]**

What is backpropagation?

- A. An application of the chain rule to compute the gradient of the loss with respect to every weight, layer by layer from output to input
- B. The forward computation of the network's prediction
- C. A method to initialise weights
- D. A data augmentation technique

<details><summary>Answer</summary>

**A.** The forward pass computes outputs and the loss. The backward pass propagates gradients back through the layers with the chain rule.

</details>

**Q88. [Week 3: Depth and width]**

In neural network terminology, what are network depth and layer width?

- A. Depth = number of layers; width = number of neurons (units) in a layer
- B. Depth = number of neurons; width = number of layers
- C. Depth = batch size; width = learning rate
- D. Depth = number of epochs; width = number of classes

<details><summary>Answer</summary>

**A.** Increasing either increases capacity: more parameters and more complex functions, but also more risk of overfitting.

</details>

**Q89. [Week 3: Overfitting]**

Which set of techniques helps mitigate or identify overfitting in neural networks?

- A. Regularisation, dropout, and early stopping based on the validation loss
- B. Increasing the number of parameters and training longer
- C. Removing the validation set
- D. Using the test set to choose the number of epochs

<details><summary>Answer</summary>

**A.** More parameters bring more capacity to overfit. Watch the validation loss (early stopping) and regularise (weight penalty, dropout).

</details>

**Q90. [Week 3: EMA]**

Why is an exponential moving average (EMA) attractive compared with a standard moving average?

- A. It needs to memorise only the previous averaged value (yₜ = αyₜ₋₁ + (1−α)xₜ) instead of a window of 2N+1 values
- B. It has no hyperparameter
- C. It gives equal weight to all past values
- D. It can only be used on images

<details><summary>Answer</summary>

**A.** EMA is a recursive filter with very low memory. It is the core of momentum and Adam, and of EMA teachers (e.g. DINO).

</details>

**Q91. [Week 3: Last layer]**

What is the relationship between the last layer of a neural network and a linear classifier?

- A. The last layer is a linear classifier operating on the features produced by earlier layers, which transform the space to make classes linearly separable
- B. The last layer is always a convolution
- C. The last layer performs k-means clustering
- D. They are unrelated

<details><summary>Answer</summary>

**A.** Each layer maps one feature space into another. Ideally the final space is linearly separable, so a linear classifier suffices.

</details>

**Q92. [Week 3: Initialisation]**

How are the weights W and b of a neural network typically initialised?

- A. Randomly drawn from a uniform or normal distribution (small values)
- B. All set to exactly zero
- C. All set to one
- D. Copied from the input data

<details><summary>Answer</summary>

**A.** Random initialisation breaks symmetry, so neurons learn different features. With all-equal weights every neuron would get the same gradient.

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

</details>

**Q94. [Week 4: CV assumptions]**

Which is a major assumption made when using datasets to develop computer vision models?

- A. The dataset distribution matches the distribution of the data encountered in the end application
- B. Images in the test set are identical to training images
- C. Every image contains every class
- D. Labels are allowed to be random

<details><summary>Answer</summary>

**A.** Other assumptions include no ambiguity in the ground truth. Both are often violated in practice, so a reported 87% accuracy may not hold.

</details>

**Q95. [Week 4: Conv vs FC]**

Why is a convolutional layer more useful than a fully-connected layer for images?

- A. It exploits local spatial structure and shares weights across locations, so it has far fewer parameters and detects a pattern anywhere
- B. It has more parameters, so it always learns more
- C. It ignores the order of pixels
- D. It does not need training

<details><summary>Answer</summary>

**A.** Flattening an image into an MLP loses spatial structure (shuffled pixels give the same result) and needs huge weight matrices.

</details>

**Q96. [Week 4: Invariance]**

What type of invariance does a convolutional layer introduce?

- A. Translation invariance: a filter detects the same feature wherever it appears
- B. Rotation invariance
- C. Scale (size) invariance
- D. Colour invariance

<details><summary>Answer</summary>

**A.** Convolutions are translation invariant (strictly, equivariant), but not rotation or size invariant. That's one reason data augmentation helps.

</details>

**Q97. [Week 4: Conv output size]**

What is the output spatial size of a convolution with input size N, filter size F, padding P and stride S?

- A. (N − F + 2P)/S + 1
- B. (N + F − P)·S
- C. N/S + F
- D. (N − 2P)/F + S

<details><summary>Answer</summary>

**A.** For example, a 7×7 input with a 3×3 filter, no padding, stride 1 gives 5×5. With stride 2 it gives 3×3.

</details>

**Q98. [Week 4: Conv output size]**

A 7×7 input is convolved with a 3×3 filter, no padding, stride 3. What happens?

- A. (7 − 3)/3 + 1 is not an integer, so the filter does not fit evenly; this stride does not work without padding
- B. The output is 3×3
- C. The output is 5×5
- D. The output is 7×7

<details><summary>Answer</summary>

**A.** (7−3)/3 + 1 = 2.33. Choose hyperparameters so the formula gives an integer, e.g. by adding zero padding.

</details>

**Q99. [Week 4: Conv parameters]**

A conv layer applies 6 filters of size 5×5 to a 32×32×3 RGB image (with bias). How many learnable parameters does it have?

- A. 456
- B. 150
- C. 450
- D. 18,432

<details><summary>Answer</summary>

**A.** Each filter has 5·5·3 = 75 weights + 1 bias = 76, and 76 × 6 = 456. The filter depth always matches the input channels.

</details>

**Q100. [Week 4: Tensor shape]**

A batch of 100 RGB images of 224×224 pixels is stored in PyTorch as which tensor shape?

- A. [100, 3, 224, 224]
- B. [224, 224, 3, 100]
- C. [3, 100, 224, 224]
- D. [100, 224, 3]

<details><summary>Answer</summary>

**A.** PyTorch uses B×C×H×W: batch, channels, height, width.

</details>

**Q101. [Week 4: Pooling]**

Why are pooling layers used in CNNs, and how many learnable parameters do they have?

- A. To reduce the spatial dimension of activation maps (downsampling); they have no learnable parameters
- B. To add non-linearity; they have one weight per pixel
- C. To normalise the batch; they learn γ and β
- D. To increase the image resolution; they have many parameters

<details><summary>Answer</summary>

**A.** Pooling has 3 hyperparameters (type such as max or average, filter size, stride) but no parameters. Strided convolutions can also downsample.

</details>

**Q102. [Week 4: Learned features]**

What do convolutional filters typically learn at different depths of a CNN?

- A. Early filters learn simple features (edges, contrast patterns); deeper filters combine them into complex parts and objects
- B. Early filters learn whole objects; deeper filters learn edges
- C. All filters learn the same thing
- D. Filters only learn colours

<details><summary>Answer</summary>

**A.** Stacking convolutions grows the receptive field, so deeper layers 'see' larger regions and more abstract patterns.

</details>

**Q103. [Week 4: Generalisation]**

Why is generalisation in computer vision so challenging?

- A. Objects vary in viewpoint, illumination, deformation, occlusion, background clutter and intra-class appearance
- B. Images contain too few pixels
- C. CNNs cannot be trained with gradient descent
- D. Every image has a unique label

<details><summary>Answer</summary>

**A.** Generalising means the features learned during training are also useful on unseen test data with all this variability.

</details>

**Q104. [Week 4: Dropout vs BatchNorm]**

What is the difference between dropout and batch normalisation layers?

- A. Dropout randomly zeroes activations during training to reduce overfitting; batch norm normalises activations to zero mean / unit variance per batch to speed up and stabilise training
- B. Both randomly remove neurons
- C. Batch norm randomly removes neurons; dropout normalises activations
- D. Neither behaves differently in training and testing

<details><summary>Answer</summary>

**A.** Batch norm speeds up training and makes it more robust to bad initialisation. Both layers behave differently in model.train() and model.eval().

</details>

**Q105. [Week 4: Data augmentation]**

What is the role of data augmentation in computer vision?

- A. Apply random transformations (flips, crops, rotations, colour jitter) during training to increase diversity and improve generalisation
- B. Increase the size of the test set
- C. Remove noisy labels
- D. Replace the need for a loss function

<details><summary>Answer</summary>

**A.** Augmentation teaches invariances that convolutions don't provide, like rotation, scale and colour, and reduces overfitting.

</details>

**Q106. [Week 4: Transfer learning]**

What is transfer learning and how is it practically achieved?

- A. Reuse a model pre-trained on a large dataset (e.g. ImageNet ResNet), replace the final layer to match the new task, then freeze or fine-tune the weights
- B. Train a new model from random weights on the test set
- C. Copy the labels from one dataset to another
- D. Use k-means to transfer cluster centres

<details><summary>Answer</summary>

**A.** Pre-trained models (e.g. PyTorch ResNet) already contain useful generic features, so far less labelled data is needed.

</details>

**Q107. [Week 4: ResNet]**

What innovation did ResNet introduce that allowed much deeper CNNs?

- A. Residual (skip / shortcut) connections that add the block's input to its output, helping gradients flow
- B. Max pooling
- C. The sigmoid activation
- D. Fully connected layers only

<details><summary>Answer</summary>

**A.** Backpropagating through many layers applies the chain rule many times and the gradient vanishes. Skip connections give the gradient a direct path.

</details>

**Q108. [Week 4: UNet]**

What is a UNet and why does it work well for segmentation?

- A. A multiscale encoder-decoder CNN with skip connections: downsampling gives a large receptive field, upsampling restores resolution, skips keep fine detail
- B. A recurrent network for text
- C. A single convolution layer with a huge kernel
- D. A decision tree for pixels

<details><summary>Answer</summary>

**A.** Without pooling you would need ~100 layers of 3×3 convs to cover a 224×224 image. A UNet gets a global receptive field efficiently and outputs a per-pixel map.

</details>

**Q109. [Week 4: Receptive field]**

Why not simply remove all pooling and stack 3×3 convolutions to segment a 224×224 image?

- A. Each 3×3 conv only grows the receptive field by 2 pixels, so ~100+ layers are needed: hard to backpropagate and costly
- B. Convolutions cannot be stacked
- C. The output would be 1×1
- D. 3×3 kernels have no parameters

<details><summary>Answer</summary>

**A.** Bigger kernels (e.g. 11×11) reduce depth but make the parameter count grow quickly. Multiscale (UNet) is the practical answer.

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

</details>

**Q111. [Week 5: Metrics]**

Which two metrics are the same thing?

- A. Recall and sensitivity
- B. Precision and specificity
- C. Recall and specificity
- D. Accuracy and precision

<details><summary>Answer</summary>

**A.** Recall = sensitivity = true positive rate = TP/(TP+FN): how many of the actual positives were found.

</details>

**Q112. [Week 5: Metrics]**

A model gives TP = 40, FN = 10, FP = 20, TN = 930. What are its precision and recall?

- A. Precision ≈ 0.67, Recall = 0.80
- B. Precision = 0.80, Recall ≈ 0.67
- C. Precision = 0.97, Recall = 0.80
- D. Precision = 0.40, Recall = 0.20

<details><summary>Answer</summary>

**A.** Precision = 40/(40+20) = 0.667. Recall = 40/(40+10) = 0.8. Accuracy would be 970/1000 = 97%, which looks good but hides the errors.

</details>

**Q113. [Week 5: Error types]**

A false positive is also called:

- A. A false alarm / Type I error
- B. A miss detection / Type II error
- C. A true negative
- D. Sensitivity

<details><summary>Answer</summary>

**A.** False positive = false alarm = Type I. False negative = miss detection = Type II. The costs of the two are often very different.

</details>

**Q114. [Week 5: Operating point]**

For a disease screening test, which type of error should be minimised?

- A. False negatives (high sensitivity), to catch most disease cases
- B. False positives, to reduce false alarms
- C. Neither; maximise accuracy
- D. Both must be exactly zero

<details><summary>Answer</summary>

**A.** Screening aims to catch most cases. Diagnosis (e.g. a ~$2000 PET scan after screening) aims to minimise false positives.

</details>

**Q115. [Week 5: ROC]**

How is a ROC curve built?

- A. Vary the decision threshold and plot sensitivity (TPR) against 1 − specificity (FPR) for each threshold
- B. Plot training loss against epochs
- C. Plot precision against the number of classes
- D. Plot predicted confidence against accuracy for each bin

<details><summary>Answer</summary>

**A.** Each threshold gives one operating point. The point closest to the top-left corner is often reported, and the area under the curve summarises performance.

</details>

**Q116. [Week 5: Prevalence]**

What is the effect of low disease prevalence on the ROC curve?

- A. No effect on the ROC, since sensitivity and specificity are computed within each class; other metrics such as precision are needed
- B. The ROC curve becomes a straight line
- C. The ROC area always increases
- D. The ROC can no longer be computed

<details><summary>Answer</summary>

**A.** ROC is prevalence-independent. Precision (positive predictive value) drops when the disease is rare.

</details>

**Q117. [Week 5: Calibration]**

What is a calibration curve?

- A. A plot of the model's predicted confidence (binned) against the actual accuracy in each bin; a calibrated model lies on the diagonal
- B. A plot of the ROC curve for each class
- C. A plot of loss versus learning rate
- D. A histogram of class labels

<details><summary>Answer</summary>

**A.** To build it, test the model, store confidence and correctness, bin by confidence, and compute accuracy per bin. For the 50–60% bin a calibrated model would be ~55% accurate.

</details>

**Q118. [Week 5: Calibration]**

Deep neural classifiers are often described as poorly calibrated. What does this usually mean?

- A. They are over-confident: predicted confidence is higher than the actual accuracy
- B. They are always under-confident
- C. They always output 50% confidence
- D. They cannot output probabilities

<details><summary>Answer</summary>

**A.** On a calibration plot, over-confident models fall below the diagonal. Ensembles often improve calibration.

</details>

**Q119. [Week 5: Distribution shift]**

What is the difference between feature shift and class shift?

- A. Feature shift: the appearance of inputs changes (e.g. new camera, weather); class shift: novel classes appear that were not seen during training
- B. Feature shift: new classes appear; class shift: image brightness changes
- C. Both mean the learning rate changes
- D. They only occur during training

<details><summary>Answer</summary>

**A.** Both are safety concerns because the deployment data no longer matches the training distribution, and the model may fail silently.

</details>

**Q120. [Week 5: Novel classes]**

Why do deep learning classifiers often assign high confidence to inputs from novel classes?

- A. Softmax must distribute probability among known classes, and a novel input far from the decision boundaries gets a large score for some known class
- B. Because they detect the new class and say so
- C. Because dropout is active at test time
- D. Because the loss is mean squared error

<details><summary>Answer</summary>

**A.** Most classifiers have no 'unknown' mechanism. Class scores grow with distance from the boundary, so out-of-distribution inputs can look very confident.

</details>

**Q121. [Week 5: Adversarial attack]**

At a high level, how is an adversarial attack computed?

- A. Compute the gradient of the loss with respect to the input image at test time and add a small, imperceptible perturbation that pushes the prediction to a wrong class
- B. Poison the training data with a trigger pattern
- C. Retrain the model with a different learning rate
- D. Remove random pixels from the image

<details><summary>Answer</summary>

**A.** It exploits the backward pass and gradients, but on the input rather than the weights. It finds the shortest vector that moves the point across a decision boundary (e.g. panda → gibbon).

</details>

**Q122. [Week 5: Backdoor attack]**

What is a backdoor attack?

- A. Training data is poisoned with a trigger pattern linked to a target label, so the model behaves normally on clean data but misclassifies whenever the trigger appears
- B. A gradient-based perturbation computed at test time
- C. Stealing a model's weights from a server
- D. Overfitting caused by too many epochs

<details><summary>Answer</summary>

**A.** It is hard to detect because tests on clean data look normal. It's a danger when using public or community datasets and pre-trained models.

</details>

**Q123. [Week 5: Ensembles]**

What is an ensemble, and what benefits does it usually give?

- A. Several different models (e.g. different random initialisations) whose predictions are averaged; often better calibration, accuracy and robustness
- B. One model trained for more epochs
- C. A model with more layers
- D. A single model evaluated on several test sets

<details><summary>Answer</summary>

**A.** Ensemble members must differ, for example through random weights, data subsets, or architectures. Otherwise averaging gains nothing.

</details>

**Q124. [Week 5: Explainable AI]**

Which statement about explainable AI is correct?

- A. Decision trees are explainable because their rules can be read; deep learning models are typically not and need tools like saliency maps
- B. Deep learning is inherently explainable
- C. Decision trees are never explainable
- D. Explainability is irrelevant in safety-critical applications

<details><summary>Answer</summary>

**A.** Explainability matters for justifying predictions (contestability, safety, e.g. mortgage decisions). CNN tools include gradient saliency, occlusion maps, Grad-CAM and feature visualisation.

</details>

**Q125. [Week 5: Saliency maps]**

How is a gradient-based saliency map computed for a CNN?

- A. Take the derivative of the class score with respect to the input image pixels to see which pixels most influence the prediction
- B. Take the derivative of the loss with respect to the learning rate
- C. Average all images of the class
- D. Count the number of filters in each layer

<details><summary>Answer</summary>

**A.** Training differentiates with respect to the weights. Saliency differentiates with respect to the image. Occlusion maps instead mask image regions and measure the change.

</details>

**Q126. [Week 5: train/eval mode]**

Why can model.train() vs model.eval() give drastically different validation accuracy?

- A. Dropout and batch norm behave differently: in eval mode dropout is disabled and batch norm uses running statistics instead of batch statistics
- B. eval() changes the weights
- C. train() removes the last layer
- D. There is never any difference

<details><summary>Answer</summary>

**A.** With Dropout(0.1) in train mode, 10% of activations are set to 0. Testing a single sample in train mode gives batch norm meaningless batch statistics.

</details>

**Q127. [Week 5: Bias case study]**

In the case study, a CNN could predict a patient's race from chest X-rays. Why is this an ethical concern?

- A. The model may use hidden demographic signals, which can introduce bias in clinical predictions that humans cannot easily see or explain
- B. Because chest X-rays cannot be used for ML
- C. Because the model was too accurate at detecting disease
- D. Because CNNs cannot process grey-scale images

<details><summary>Answer</summary>

**A.** Average scans and saliency maps did not clearly reveal why. This shows the need for responsible AI throughout the value chain.

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

</details>

**Q129. [Week 6: Learning paradigms]**

Which statement correctly distinguishes unsupervised, self-supervised and semi-supervised learning?

- A. Unsupervised finds patterns without labels; self-supervised creates labels from the data itself (pretext tasks); semi-supervised uses a few labelled plus many unlabelled samples
- B. All three require fully labelled data
- C. Self-supervised uses human labels for every sample
- D. Semi-supervised uses no data

<details><summary>Answer</summary>

**A.** Self-supervised is a form of unsupervised learning where the supervisory signal comes from the data (e.g. predict masked parts or the next token).

</details>

**Q130. [Week 6: k-means]**

What are the steps of k-means clustering?

- A. Pick K random centroids → assign each point to the closest centroid → recompute each centroid as the mean of its points → repeat until stable
- B. Store all data → find K nearest neighbours → majority vote
- C. Fit a Gaussian per class with labels → compute likelihood
- D. Split the data on the best feature → recurse

<details><summary>Answer</summary>

**A.** K is a hyperparameter, not the same k as in kNN. k-means is unsupervised, while kNN needs labels.

</details>

**Q131. [Week 6: k-means]**

Which is a limitation of k-means?

- A. You must choose K, it assumes roughly spherical clusters, and it can get stuck in a local minimum depending on the initial centroids
- B. It is very slow and complicated to implement
- C. It requires labelled data
- D. It always finds the global optimum

<details><summary>Answer</summary>

**A.** Advantages: simple and efficient. Limitations: choosing K, cluster shape assumptions, and sensitivity to initialisation.

</details>

**Q132. [Week 6: GMM]**

What is a Gaussian Mixture Model (GMM)?

- A. A model of the data distribution as a weighted sum of Gaussians, each with its own mean and (co)variance, giving soft cluster membership probabilities
- B. A neural network with Gaussian activations
- C. A decision tree whose leaves are Gaussians
- D. A method that assigns each point to exactly one cluster by distance

<details><summary>Answer</summary>

**A.** Pearson (1894) fitted two Gaussians to Weldon's crab measurements by hand. Today GMMs are optimised with EM.

</details>

**Q133. [Week 6: EM]**

How does Expectation-Maximisation (EM) optimise a GMM?

- A. E-step: compute the probability that each sample belongs to each Gaussian; M-step: update means, variances and weights using those probabilities; repeat
- B. Compute the gradient of the loss and use Adam
- C. Solve the normal equations once
- D. Randomly assign labels and keep the best

<details><summary>Answer</summary>

**A.** EM is like k-means with soft assignments: k-means uses hard assignments to the nearest centroid, GMM-EM uses probabilities.

</details>

**Q134. [Week 6: Maximum likelihood]**

For n independent samples xᵢ assumed to come from N(μ, σ²), what are the maximum likelihood estimates?

- A. μ̂ = (1/n)Σxᵢ and σ̂² = (1/n)Σ(xᵢ − μ̂)²
- B. μ̂ = median(xᵢ) and σ̂² = max(xᵢ) − min(xᵢ)
- C. μ̂ = 0 and σ̂² = 1 always
- D. μ̂ = Σxᵢ and σ̂² = Σxᵢ²

<details><summary>Answer</summary>

**A.** Minimise the negative log-likelihood and set derivatives to 0. Note that σ̂² with 1/n is biased when μ̂ is estimated from the data; use 1/(n−1) for small datasets.

</details>

**Q135. [Week 6: Maximum likelihood]**

What does maximum likelihood estimation seek?

- A. The parameter values that make the observed data most probable under the assumed model
- B. The parameters that maximise the loss
- C. The parameters that make the model as simple as possible
- D. The labels that best fit the parameters

<details><summary>Answer</summary>

**A.** In practice we minimise the negative log-likelihood, which turns a product of probabilities into a sum.

</details>

**Q136. [Week 6: Clustering applications]**

Which is an application of unsupervised clustering mentioned in the lecture?

- A. Anomaly detection, recommender systems, and brain tissue segmentation from MRI
- B. Computing the gradient of a loss
- C. Positional encoding in transformers
- D. Choosing the learning rate

<details><summary>Answer</summary>

**A.** Clustering is useful to explore trends in data that cannot be labelled, or is too large to label.

</details>

**Q137. [Week 6: Pseudo-labelling]**

What is pseudo-labelling in semi-supervised learning, and what is its main limitation?

- A. A model trained on labelled data predicts labels for unlabelled data, and confident predictions are added as training labels; wrong pseudo-labels can reinforce errors (confirmation bias)
- B. Humans relabel the test set; it is too slow
- C. Labels are randomly shuffled; it destroys accuracy
- D. It deletes unlabelled data; it wastes data

<details><summary>Answer</summary>

**A.** A confidence threshold is used, but over-confident wrong predictions still propagate.

</details>

**Q138. [Week 6: Consistency regularisation]**

What is consistency regularisation?

- A. Encouraging the model to give the same prediction for different augmented (perturbed) versions of the same unlabelled input
- B. Forcing all weights to have the same value
- C. Using the same learning rate for every layer
- D. Training only on labelled data

<details><summary>Answer</summary>

**A.** The key consideration is choosing augmentations that don't change the true class.

</details>

**Q139. [Week 6: Contrastive learning]**

In contrastive learning with data augmentation, how are positive and negative pairs created?

- A. Positives: random augmentations of the same reference image; negatives: other images from the dataset
- B. Positives: images with the same human label; negatives: images with no label
- C. Positives: the test set; negatives: the training set
- D. Positives and negatives are both random noise

<details><summary>Answer</summary>

**A.** A contrastive or margin loss pulls positive pairs together and pushes negatives apart in embedding space. It's often used as pretraining for downstream tasks.

</details>

**Q140. [Week 6: Contrastive learning]**

Which is a limitation of contrastive learning?

- A. Choosing which augmentations to apply and at what magnitude, and needing many good negative pairs with long training
- B. It requires every image to be labelled
- C. It can only be used with decision trees
- D. It has no hyperparameters

<details><summary>Answer</summary>

**A.** Too-strong augmentations destroy semantic content, and too-weak ones teach nothing. Good negatives are hard to get.

</details>

**Q141. [Week 6: Self-supervised]**

Which is an example of predictive self-supervised learning?

- A. Masking parts of the input (words or image patches) and training the model to predict them
- B. Training a classifier on human-labelled ImageNet classes
- C. Running k-nearest neighbours on labelled data
- D. Grid search for hyperparameters

<details><summary>Answer</summary>

**A.** Other predictive tasks include next-token prediction (GPT), masked tokens (BERT), masked image patches (MAE), and predicting rotation or ordering.

</details>

**Q142. [Week 6: Foundation models]**

What makes a model a 'foundation model'?

- A. It is trained on huge amounts of (mostly unlabelled) data, typically with self-supervision, and can be used or adapted for many different downstream tasks
- B. It is the first layer of a CNN
- C. It is trained on a small labelled dataset for one task
- D. It is a model that cannot be fine-tuned

<details><summary>Answer</summary>

**A.** Examples include DINOv2 for vision and LLMs. Advantages are general features and little labelled data needed. Dangers include inherited biases, opacity, and concentration of power.

</details>

**Q143. [Week 6: Transfer learning]**

A model trained on Dataset A (1000 classes) is reused for Dataset B (10 classes). What is the difference between freezing and fine-tuning?

- A. Freezing: only the new final layer is trained; fine-tuning: all (or many) layers are updated, usually with a small learning rate
- B. Freezing updates all layers; fine-tuning updates none
- C. Both delete the pre-trained weights
- D. Fine-tuning changes the number of input channels

<details><summary>Answer</summary>

**A.** First replace the 1000-class head with a 10-class head, then balance new knowledge against old.

</details>

**Q144. [Week 6: t-SNE / UMAP]**

Which statement about t-SNE is correct?

- A. It is a non-linear method for visualising high-dimensional data in 2D/3D by matching neighbour similarities, using the KL divergence
- B. It is a linear projection identical to PCA
- C. It is a supervised classifier
- D. It is used to train GANs

<details><summary>Answer</summary>

**A.** t-SNE uses Gaussian neighbourhoods in high dimension and a Student-t in low dimension. UMAP is similar, usually faster, and preserves more global structure.

</details>

**Q145. [Week 6: KL divergence]**

Which property of the Kullback–Leibler divergence is true?

- A. It is not symmetric: D_KL(p‖q) ≠ D_KL(q‖p) in general, so it is a divergence, not a distance
- B. It is always negative
- C. It is symmetric and satisfies the triangle inequality
- D. It equals the mean squared error

<details><summary>Answer</summary>

**A.** D_KL(p‖q) = ∫p(x) log(p(x)/q(x)) dx ≥ 0, with equality only when p = q.

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

</details>

**Q147. [Week 7: Applications]**

Which is an application of time-series data?

- A. Forecasting stock prices or energy demand
- B. Semantic segmentation of a single photo
- C. Clustering iris flowers
- D. Computing a confusion matrix

<details><summary>Answer</summary>

**A.** Time-series tasks include forecasting (future values) and prediction or classification (e.g. activity recognition from sensor sequences).

</details>

**Q148. [Week 7: Tokenisation]**

What is the trade-off between character-level and word-level tokens?

- A. Characters: small vocabulary (e.g. 65) but very long sequences; words/subwords: large vocabulary but shorter sequences
- B. Characters: large vocabulary and short sequences
- C. There is no trade-off
- D. Words always need fewer parameters than characters

<details><summary>Answer</summary>

**A.** Tokenisers like SentencePiece or BPE use subwords as a compromise. Modern LLMs have ~50,000+ tokens.

</details>

**Q149. [Week 7: Tokenisation]**

In the Shakespeare example, the text has 65 different symbols. Where does this number come from?

- A. 26 letters × 2 (upper and lower case) plus special characters such as space, punctuation and line feed
- B. The number of words in the text
- C. The embedding dimension
- D. The number of attention heads

<details><summary>Answer</summary>

**A.** Using one token per character gives a tiny vocabulary but a 1,115,000-token sequence.

</details>

**Q150. [Week 7: N-gram]**

How does a bigram language model generate text?

- A. Given the current token, it samples the next token from probabilities of pairs of tokens learned from (or counted in) the text
- B. It uses self-attention over the whole sequence
- C. It retrieves full sentences from the training set
- D. It uses a convolution over the image of the text

<details><summary>Answer</summary>

**A.** A learned bigram matrix gives 'still gibberish but not random' text. Larger N-grams quickly become intractable.

</details>

**Q151. [Week 7: Bag of words]**

What is a disadvantage of the Bag of Words representation?

- A. It loses the order of words (and sometimes context) and can be a very high-dimensional vector
- B. It is very hard to calculate
- C. It requires a GPU
- D. It cannot count repeated words

<details><summary>Answer</summary>

**A.** BoW is easy to calculate but ignores order. A lookup table of embeddings gives dense vectors with semantic meaning instead.

</details>

**Q152. [Week 7: Embeddings]**

What is a token embedding (lookup table)?

- A. Each token is associated with a learned vector in high dimension, where similar meanings give similar vectors
- B. A one-hot vector with the token count
- C. The position of the token in the sentence
- D. The probability of the next token

<details><summary>Answer</summary>

**A.** Word2Vec is a classic method. Embeddings are ubiquitous in LLMs (e.g. king − man + woman ≈ queen).

</details>

**Q153. [Week 7: RNN]**

How is a recurrent neural network (RNN) different from a feed-forward neural network?

- A. It keeps a hidden state (memory) that is passed from one time step to the next, so it can process sequences of variable length
- B. It has no weights
- C. It processes all tokens in parallel with attention
- D. It can only process images

<details><summary>Answer</summary>

**A.** The same weights are applied at every step. Configurations include many-to-many (translation), many-to-one (sentiment), and one-to-many (generation).

</details>

**Q154. [Week 7: RNN limitations]**

Which is NOT a limitation of RNNs?

- A. They cannot process sequences at all
- B. They are computationally expensive because the recurrence cannot be parallelised
- C. They suffer from vanishing and exploding gradients
- D. They struggle to keep long-range context

<details><summary>Answer</summary>

**A.** RNNs do process sequences. Their limitations are sequential compute, unstable gradients, and short memory (e.g. forgetting what was bought at the start of a long sentence).

</details>

**Q155. [Week 7: LSTM]**

What regulates the cell state and hidden state in an LSTM?

- A. Gates (forget, input, output) that control what information is removed, added and output
- B. A softmax over the vocabulary
- C. Positional encodings
- D. Max pooling layers

<details><summary>Answer</summary>

**A.** The cell state is an ongoing memory. Gates (sigmoid-controlled) decide what to forget, what to write, and what to expose.

</details>

**Q156. [Week 7: LSTM limitations]**

What is a remaining limitation of LSTMs?

- A. They are still computationally expensive because they cannot be parallelised over the sequence
- B. They have no memory at all
- C. They cannot be trained with backpropagation
- D. They only work on images

<details><summary>Answer</summary>

**A.** LSTMs improve memory but are still sequential. Transformers remove the recurrence.

</details>

**Q157. [Week 7: Transformers]**

What is the key difference between the Transformer architecture and RNNs/LSTMs?

- A. Transformers have no recurrence: all tokens are processed simultaneously and self-attention captures context across the whole sequence
- B. Transformers process one token at a time with a hidden state
- C. Transformers use convolution only
- D. Transformers cannot be used for text

<details><summary>Answer</summary>

**A.** Parallel processing makes training much faster on GPUs, and attention captures global context in long sequences.

</details>

**Q158. [Week 7: Transformer limitations]**

Which is a limitation of Transformers?

- A. Very data hungry, and attention cost grows quadratically with sequence length (memory and compute)
- B. They cannot be parallelised
- C. They have no learnable parameters
- D. They cannot capture long-range context

<details><summary>Answer</summary>

**A.** Mitigations include the KV cache for generation and linear or windowed attention for long contexts.

</details>

**Q159. [Week 7: Positional encoding]**

Why is a positional encoding needed in a Transformer?

- A. Self-attention is order-agnostic, so without positions 'I'm not happy because it is raining' and 'I'm happy because it is not raining' would look the same
- B. To normalise the token embeddings
- C. To reduce the vocabulary size
- D. To mask future tokens

<details><summary>Answer</summary>

**A.** A unique (sinusoidal or learned) vector is added to each position's embedding so the order of tokens matters.

</details>

**Q160. [Week 7: Attention]**

What does the attention mechanism compute?

- A. Softmax(QKᵀ/√d)·V: each token's query is compared with all keys, and the resulting weights mix the values to update the token
- B. The average of all token embeddings, ignoring queries and keys
- C. A convolution over the sequence
- D. The probability of each class directly

<details><summary>Answer</summary>

**A.** Each row of the attention matrix sums to 1. Adding the weighted values 'nudges' each token's embedding according to its context.

</details>

**Q161. [Week 7: Multi-head]**

With dim_emb = 1024, why use multi-head attention instead of one big attention?

- A. Splitting the dimension into several heads (e.g. 16 × 64) gives smaller matrices and lets each head capture a different type of relationship in parallel; outputs are concatenated
- B. It increases the vocabulary size
- C. It removes the need for positional encoding
- D. It makes the model recurrent

<details><summary>Answer</summary>

**A.** Heads are computed in parallel, concatenated, and passed through a final linear projection.

</details>

**Q162. [Week 7: Training LLMs]**

How are decoder-only transformers (like GPT) trained in practice?

- A. Self-supervised next-token prediction on huge text corpora: the target is the input shifted by one token, with a causal mask
- B. Supervised classification with human labels for every sentence
- C. k-means clustering of words
- D. Reinforcement learning from scratch with no text

<details><summary>Answer</summary>

**A.** 'Paris is the capital and largest city of France' gives many training pairs. BERT-style models instead predict masked tokens.

</details>

**Q163. [Week 7: Output layer]**

What does the final layer of a language transformer do?

- A. A linear classifier over every token in the vocabulary, followed by softmax to give next-token probabilities
- B. Max pooling over the sequence
- C. A k-means assignment of the token
- D. A positional encoding

<details><summary>Answer</summary>

**A.** Embeddings are converted back to token probabilities, and the next token is sampled and appended (autoregressive).

</details>

**Q164. [Week 7: KV cache]**

What is the purpose of the KV cache during LLM generation?

- A. Store the keys and values of previous tokens so they are not recomputed at every new generated token
- B. Store the training dataset
- C. Cache the gradients for backpropagation
- D. Store the vocabulary

<details><summary>Answer</summary>

**A.** During autoregressive generation only the new token's query needs to be computed against the cached keys and values.

</details>

**Q165. [Week 7: Model size]**

According to the lecture, how many transformer layers do GPT-3 and DeepSeek-R1 have?

- A. GPT-3: 96 layers; DeepSeek-R1: 61 layers
- B. GPT-3: 12 layers; DeepSeek-R1: 6 layers
- C. Both have 1 layer
- D. GPT-3: 61 layers; DeepSeek-R1: 96 layers

<details><summary>Answer</summary>

**A.** An LLM stacks N attention blocks (attention + normalisation + feed-forward with residuals).

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

</details>

**Q167. [Week 8: Manifold hypothesis]**

What is the manifold hypothesis?

- A. Many real-world high-dimensional datasets actually lie along low-dimensional manifolds embedded in the high-dimensional space
- B. All data is uniformly spread in high-dimensional space
- C. Neural networks must be shallow
- D. Images are random noise

<details><summary>Answer</summary>

**A.** A 3×32×32 image has 256^3072 possible values, but real images occupy a tiny, structured subset.

</details>

**Q168. [Week 8: PCA]**

How are the principal components ordered?

- A. By decreasing eigenvalue: the first component captures the most variance
- B. Randomly
- C. By increasing variance
- D. Alphabetically by feature name

<details><summary>Answer</summary>

**A.** PCA computes an orthonormal coordinate system that maximises variance along each successive direction.

</details>

**Q169. [Week 8: PCA vs AE]**

How does PCA relate to an autoencoder?

- A. PCA is a linear autoencoder: encode z = Uᵀx, decode x̂ = UUᵀx; an autoencoder generalises this with non-linear encoder and decoder networks
- B. PCA is a GAN
- C. They are unrelated
- D. An autoencoder is a linear version of PCA

<details><summary>Answer</summary>

**A.** Locally, a non-linear autoencoder manifold has a PCA-like tangent space.

</details>

**Q170. [Week 8: PCA limits]**

If the data space has 10⁶ dimensions (e.g. images) and we have 50,000 images, what is the largest subspace PCA can compute?

- A. At most 50,000 dimensions
- B. 10⁶ dimensions
- C. Exactly 2 dimensions
- D. 10¹² dimensions

<details><summary>Answer</summary>

**A.** The covariance matrix has rank at most equal to the number of samples, so at most 50,000 meaningful components.

</details>

**Q171. [Week 8: Autoencoder]**

How is an autoencoder trained, and why is it self-supervised?

- A. It minimises the reconstruction error between input x and output x̂ through a bottleneck latent z; the input itself is the target, so no labels are needed
- B. It uses human labels as targets
- C. It maximises the classification accuracy
- D. It is trained with the KL divergence only

<details><summary>Answer</summary>

**A.** The encoder compresses to latent z and the decoder reconstructs. It's self-supervised because the data supervises itself.

</details>

**Q172. [Week 8: AE variants]**

What is a denoising autoencoder?

- A. An autoencoder whose input is corrupted with noise (x + N(0,1)) and which is trained to reconstruct the clean x
- B. An autoencoder with no decoder
- C. An autoencoder trained to add noise
- D. A GAN discriminator

<details><summary>Answer</summary>

**A.** It learns robust features and to project noisy inputs back onto the data manifold.

</details>

**Q173. [Week 8: AE variants]**

What is a sparse autoencoder?

- A. Instead of a small bottleneck, the latent can be as large as (or larger than) the input, but only a few latent units may be active (L1 penalty or top-k)
- B. An autoencoder with very few training samples
- C. An autoencoder with sparse input images
- D. An autoencoder whose weights are all zero

<details><summary>Answer</summary>

**A.** Sparse autoencoders are used for LLM interpretability because a few active features are easier to interpret.

</details>

**Q174. [Week 8: AE variants]**

What is a masked autoencoder?

- A. Parts (e.g. patches) of the input are masked and the model learns to reconstruct the missing parts
- B. An autoencoder that masks its weights
- C. An autoencoder with a causal mask over time
- D. An autoencoder trained only on labelled masks

<details><summary>Answer</summary>

**A.** It's a predictive self-supervised pretraining task that learns strong image features.

</details>

**Q175. [Week 8: AE as generator]**

Why is a standard autoencoder a poor generative model?

- A. The distribution of the latent z is unknown and irregular, so a randomly chosen z usually decodes into rubbish
- B. The decoder cannot output images
- C. Autoencoders cannot be trained
- D. The latent is always one-dimensional

<details><summary>Answer</summary>

**A.** The next best idea is to force p(z) to be what we want, which is what the VAE does.

</details>

**Q176. [Week 8: VAE loss]**

What is the loss function of a VAE?

- A. Reconstruction loss (e.g. MSE between x and x̂) + β × KL divergence between q(z|x) and the prior N(0, I)
- B. Only the KL divergence
- C. Binary cross-entropy between real and fake
- D. Hinge loss on the latent

<details><summary>Answer</summary>

**A.** This comes from maximising the Evidence Lower Bound (ELBO) of the log-likelihood.

</details>

**Q177. [Week 8: Reparameterisation]**

What is the reparameterisation trick in a VAE?

- A. Sample z = μ + σ·ε with ε ~ N(0, 1), so sampling is differentiable with respect to μ and σ
- B. Set z = x directly
- C. Use a softmax over the latent
- D. Sample z uniformly from [0, 1]

<details><summary>Answer</summary>

**A.** Gradients can flow through μ and σ because the randomness is moved into ε.

</details>

**Q178. [Week 8: VAE implementation]**

In the MNIST VAE implementation, what does the encoder output?

- A. 2 × the latent dimension: μ and ln(σ²) (log-variance rather than σ, for numerical stability)
- B. Only the reconstruction x̂
- C. The class label
- D. A single scalar

<details><summary>Answer</summary>

**A.** Predicting log-variance keeps σ positive and stable.

</details>

**Q179. [Week 8: KL divergence]**

What does the Kullback–Leibler divergence measure?

- A. How one probability distribution differs from another reference distribution (non-negative, zero only if identical, not symmetric)
- B. The Euclidean distance between two vectors
- C. The accuracy of a classifier
- D. The number of parameters in a model

<details><summary>Answer</summary>

**A.** D_KL(p‖q) = ∫ p(x) log(p(x)/q(x)) dx. In the VAE it pushes q(z|x) towards N(0, I).

</details>

**Q180. [Week 8: ELBO]**

Why does the VAE maximise the Evidence Lower Bound (ELBO) rather than log p(x) directly?

- A. p(x) = ∫p(x, z)dz is intractable since the manifold z is unknown; since KL ≥ 0, the ELBO is a tractable lower bound on log p(x)
- B. Because log p(x) is always zero
- C. Because the ELBO is the classification accuracy
- D. Because maximum likelihood cannot be used with neural networks

<details><summary>Answer</summary>

**A.** An approximate posterior q_φ(z|x) (the encoder) and a decoder p_θ(x|z) make the bound computable.

</details>

**Q181. [Week 8: β-VAE]**

What happens if the KL term weight β in a VAE is set too high (e.g. ×10)?

- A. The latent fits the prior very well, but reconstructions become poor/blurry; in the extreme the latent ignores the input (posterior collapse)
- B. Reconstructions become perfect and the latent ignores the prior
- C. Nothing changes
- D. Training becomes a GAN

<details><summary>Answer</summary>

**A.** With β too low the model behaves like an autoencoder: good reconstruction but poor fit of the latent to the prior.

</details>

**Q182. [Week 8: Conditional VAE]**

What is a conditional VAE?

- A. A VAE where a condition c (e.g. class label) is given to the encoder and decoder, so you can choose what to generate
- B. A VAE that only trains when a condition is met
- C. A VAE without a decoder
- D. A VAE with a causal mask

<details><summary>Answer</summary>

**A.** It can even classify: run the VAE with each of the 10 conditions and pick the one with the lowest reconstruction error.

</details>

**Q183. [Week 8: VAE as generator]**

Why can a trained VAE be used as a generative model?

- A. Because its latent is forced towards a known multivariate Gaussian, we can sample z ~ N(0, I) and decode it into a new sample
- B. Because it memorises all training images
- C. Because it uses a discriminator
- D. Because the encoder can be inverted analytically

<details><summary>Answer</summary>

**A.** That's why it works as a generator. Two manifolds (classes) can create a ridge in z, which conditional VAEs address.

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

</details>

**Q185. [Week 9: Curse of dimensionality]**

What is the curse of dimensionality?

- A. As dimension grows, the volume grows exponentially so data become sparse, distances become similar, and most probability mass sits far from the centre
- B. Neural networks cannot have more than 3 layers
- C. Training becomes faster with more dimensions
- D. High-dimensional data always lies on a line

<details><summary>Answer</summary>

**A.** Fortunately, actual data has structure (manifolds), unlike uniformly random vectors.

</details>

**Q186. [Week 9: Interpolation]**

Why is linear interpolation between two Gaussian samples problematic in high dimension?

- A. Both endpoints have norm ≈ √D, but the midpoint has a smaller norm, which is very unlikely under the distribution
- B. The midpoint always has a larger norm
- C. Linear interpolation is impossible in high dimension
- D. Interpolated points are always identical to the endpoints

<details><summary>Answer</summary>

**A.** The midpoint falls inside the 'skin' of the hypersphere where there is almost no probability mass. Spherical interpolation stays on the shell.

</details>

**Q187. [Week 9: Normalisation]**

What does LayerNorm do compared with BatchNorm?

- A. BatchNorm normalises each channel across the batch; LayerNorm normalises the features of each sample independently
- B. LayerNorm normalises across the batch; BatchNorm normalises per sample
- C. Both are identical
- D. Neither has learnable parameters

<details><summary>Answer</summary>

**A.** Both have learnable γ and β. LayerNorm is common in transformers because it doesn't depend on batch size.

</details>

**Q188. [Week 9: GAN]**

How is a GAN trained?

- A. A generator maps random noise to fake samples; a discriminator classifies real (1) vs fake (0) with binary cross-entropy; they are trained adversarially in alternation
- B. An encoder and decoder minimise reconstruction error
- C. A model predicts the noise added at each timestep
- D. A velocity field is regressed with MSE

<details><summary>Answer</summary>

**A.** Only the decoder or generator is needed to generate data. The discriminator drives the transformation from noise to data.

</details>

**Q189. [Week 9: cGAN]**

What does a conditional GAN add?

- A. A condition (e.g. class) given to the generator (and discriminator) that modulates the features, so we can choose what to generate
- B. An encoder that reconstructs the input
- C. A noise schedule
- D. A causal mask

<details><summary>Answer</summary>

**A.** In the lecture, a 10-class classifier trained on real data is used to check that generated samples match the condition.

</details>

**Q190. [Week 9: VAE-GAN]**

What is a VAE-GAN?

- A. A VAE whose decoder output is also judged by a discriminator, combining a structured latent with sharper GAN-like samples
- B. A GAN without a generator
- C. A diffusion model with two UNets
- D. A k-means variant

<details><summary>Answer</summary>

**A.** VAE reconstructions are often blurry, and the adversarial loss adds sharpness.

</details>

**Q191. [Week 9: Score function]**

What is the score function and how can it be used to generate data?

- A. The gradient of the log data density, ∇ₓ log p(x); starting from random x and following it with Langevin dynamics (gradient steps plus small noise) moves samples towards high-density data
- B. The accuracy of a classifier; it is maximised with Adam
- C. The discriminator output of a GAN
- D. The KL divergence between two images

<details><summary>Answer</summary>

**A.** With the score you don't need p(x) itself. Langevin dynamics follows the score with a random walk.

</details>

**Q192. [Week 9: Noise]**

Why is noise added to the data when learning the score?

- A. In sparse areas between data points the score cannot be estimated; adding noise (at multiple scales) spreads the data so the score is defined everywhere
- B. To make training faster by removing data
- C. To fool the discriminator
- D. Noise has no role in score-based models

<details><summary>Answer</summary>

**A.** In practice, start with lots of noise and decrease it progressively.

</details>

**Q193. [Week 9: DDPM]**

What is the DDPM model trained to predict?

- A. The noise ε that was added to the image at timestep t (MSE between predicted and true noise), which is related to the score
- B. The class label of the image
- C. Whether the image is real or fake
- D. The KL divergence to the prior

<details><summary>Answer</summary>

**A.** A key finding is that the score at t is related to the noise added at t. Sampling starts from noise and iterates down to t = 0.

</details>

**Q194. [Week 9: Noise schedule]**

What is the noise schedule in a diffusion model?

- A. The function β(t) that sets how much noise is added at each timestep of the forward process (e.g. linear or cosine)
- B. The learning rate schedule of the optimiser
- C. The order in which images are shown to the model
- D. The number of GPUs used

<details><summary>Answer</summary>

**A.** The lecture used a linear β(t). Improved diffusion uses other options such as cosine.

</details>

**Q195. [Week 9: Guidance]**

What is classifier-free guidance?

- A. The same diffusion model makes a conditional and an unconditional prediction at each step; their difference is scaled to push generation towards the condition, with no separate classifier
- B. Training a separate classifier on noisy images and adding its gradient
- C. Generating without any condition
- D. Using a GAN discriminator as guide

<details><summary>Answer</summary>

**A.** Classifier guidance uses Bayes' rule and needs a separate classifier's gradient ∇ log p(c|x). CFG gets this by subtracting the unconditional score from the conditional score.

</details>

**Q196. [Week 9: Flow matching]**

How does conditional flow matching work?

- A. Pair random noise samples with data samples, train a network to regress the velocity along the path between them (MSE), then generate by solving the ODE dx/dt = u_θ(x, t)
- B. Train a discriminator to separate real and fake
- C. Add noise and use a random walk (Langevin) to sample
- D. Minimise the KL divergence to a Gaussian prior

<details><summary>Answer</summary>

**A.** Flow matching avoids the random walk. Generation still needs multiple steps because the learned velocity field is non-linear.

</details>

**Q197. [Week 9: Generative summary]**

Which statement correctly contrasts AE/VAE/GAN with diffusion/flow matching?

- A. AE/VAE/GAN: low-dimensional latent, one-shot generation, but many random z give bad samples; diffusion/flow: latent same dimension as x, multi-step generation, any random z gives a good sample
- B. Diffusion generates in one step; VAE needs many steps
- C. GANs use an ODE solver
- D. All of them use the same loss function

<details><summary>Answer</summary>

**A.** That's the summary from the lecture: one-shot decoders vs iterative transport of noise to data.

</details>

**Q198. [Week 9: FID]**

How does the Fréchet Inception Distance (FID) measure generation quality?

- A. Real and generated images pass through a pre-trained Inception-v3; feature means and covariances are compared as Gaussians using the Fréchet distance (lower is better)
- B. It counts the pixels that differ between images
- C. It measures the discriminator loss
- D. It is the reconstruction MSE of a VAE

<details><summary>Answer</summary>

**A.** It's not perfect but widely used. Similar metrics include IS, KID and SWD.

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

</details>

**Q200. [Week 10: Universal approximation]**

What does the universal approximation theorem state?

- A. A network with one hidden layer and enough units can approximate any continuous function arbitrarily closely
- B. Any network trains to the global minimum
- C. Deep networks never overfit
- D. Only convolutional networks can approximate functions

<details><summary>Answer</summary>

**A.** In theory yes. In practice, the more units, the harder the optimisation, so deeper networks are often used.

</details>

**Q201. [Week 10: Sigmoid shaping]**

For a unit f(x) = sigmoid(w·x + b), what are the effects of w and b?

- A. w controls the steepness of the slope (high w → sharp step); b shifts the sigmoid along x
- B. w shifts the sigmoid; b controls the steepness
- C. Both only scale the output
- D. Neither changes the shape

<details><summary>Answer</summary>

**A.** Two sigmoid units with large w and opposite signs make a square (box) function. Many boxes can approximate any 1D function.

</details>

**Q202. [Week 10: SDF]**

What is a signed distance function (SDF) of a shape?

- A. At each location, the closest distance to the shape boundary: positive outside, negative inside
- B. The pixel intensity of the image
- C. The probability that a pixel belongs to a class
- D. The distance between two cameras

<details><summary>Answer</summary>

**A.** Thresholding at 0 gives the binary mask. The value gives the distance to the object, and the gradient gives its direction.

</details>

**Q203. [Week 10: SDF]**

How is the gradient of a learned SDF (MLP) computed?

- A. Using PyTorch autograd to differentiate the network output with respect to the input coordinates
- B. By finite differences on the training labels only
- C. By running k-means on the output
- D. It cannot be computed

<details><summary>Answer</summary>

**A.** The MLP is differentiable with respect to its inputs (x, y), so autograd gives ∇SDF at any location.

</details>

**Q204. [Week 10: Resolution]**

Why is an implicit representation said to have 'infinite resolution'?

- A. The trained MLP is a continuous function of coordinates, so it can be queried at any resolution without changing the number of parameters
- B. It stores every pixel at infinite precision
- C. It uses an infinite number of neurons
- D. It compresses images to zero bytes

<details><summary>Answer</summary>

**A.** The same model generated a 1000×1000 star. That's 100× more pixels with the same number of parameters (~105 floats).

</details>

**Q205. [Week 10: Spectral bias]**

What is spectral bias in coordinate-based MLPs, and how is it addressed?

- A. MLPs are biased towards learning low-frequency functions (blobby results); mapping inputs through high-frequency sinusoids (Fourier features / positional encoding) restores detail
- B. MLPs only learn high frequencies; fix by blurring the input
- C. MLPs ignore colour; fix by adding a colour channel
- D. It is caused by the optimiser; fix by using SGD

<details><summary>Answer</summary>

**A.** Neural Tangent Kernel theory explains why raw (x, y, z) inputs make the MLP act like a low-pass filter.

</details>

**Q206. [Week 10: NeRF]**

What does a NeRF MLP take as input and produce as output?

- A. Input: 3D position (x, y, z) and viewing direction (θ, φ), i.e. 5D; output: colour (RGB) and volume density σ
- B. Input: an image; output: a class label
- C. Input: noise; output: an image
- D. Input: text; output: next token

<details><summary>Answer</summary>

**A.** Colours along each camera ray are integrated with volume rendering to produce each pixel.

</details>

**Q207. [Week 10: NeRF encoding]**

In NeRF, positional encoding with L = 10 is applied to each component of the 3D position. What is the resulting input dimension?

- A. 60 (3 × 2L)
- B. 10
- C. 30
- D. 3

<details><summary>Answer</summary>

**A.** Each scalar maps to 2L sin/cos values: 3 × 20 = 60. For the viewing direction L = 4 is used (24 dims with a unit vector).

</details>

**Q208. [Week 10: NeRF]**

NeRF needs the camera pose for each training image. How is it obtained?

- A. Provided with the data, estimated beforehand with structure-from-motion tools such as COLMAP, or optimised jointly (e.g. BARF)
- B. It is always assumed to be the identity
- C. It is predicted by a GAN discriminator
- D. It is not needed

<details><summary>Answer</summary>

**A.** Without poses, rays cannot be cast through each pixel into the scene.

</details>

**Q209. [Week 10: Volume rendering]**

In NeRF volume rendering, what is the transmittance T(t)?

- A. The probability that the ray travels from the camera to point t without being blocked
- B. The colour of the point at t
- C. The learning rate at step t
- D. The density of the medium at the camera

<details><summary>Answer</summary>

**A.** The pixel colour is a weighted sum of segment colours, with weights from density and transmittance (alpha compositing in discrete form).

</details>

**Q210. [Week 10: Alpha compositing]**

In alpha compositing, what does α = 1 represent?

- A. A completely opaque object that contributes its full colour
- B. A completely transparent object
- C. Half transparency
- D. An invalid value

<details><summary>Answer</summary>

**A.** The final colour is a linear combination of the colours, weighted by their alpha (opacity) values.

</details>

**Q211. [Week 10: Gaussian splatting]**

What is the key idea of 3D Gaussian splatting compared with NeRF?

- A. Represent the radiance field sparsely with 3D Gaussians only where density is non-zero, then project them into 2D Gaussians and sum their contributions directly
- B. Represent every point in space densely with an MLP
- C. Use a GAN to generate 3D scenes
- D. Use a decision tree to store voxels

<details><summary>Answer</summary>

**A.** Natural scenes are mostly air, so a sparse representation is far cheaper to render. Gaussians are culled outside the view frustum.

</details>

**Q212. [Week 10: Gaussian splatting]**

How is Gaussian splatting typically initialised and refined?

- A. Start from a Structure-from-Motion (SfM) point cloud, then iterate with cloning and splitting of Gaussians
- B. Start from random noise and run Langevin dynamics
- C. Start from a pre-trained ResNet
- D. Start from a single Gaussian at the origin and never change

<details><summary>Answer</summary>

**A.** Densification by cloning and splitting adds detail where needed.

</details>

**Q213. [Week 10: Graphics]**

In the camera model, what do the extrinsic [R|T] and intrinsic K matrices do?

- A. [R|T] maps world coordinates to camera coordinates (rotation + translation); K maps camera coordinates to the image (screen) projection
- B. [R|T] stores the pixel colours; K stores the depth
- C. Both are learned by the NeRF MLP
- D. K maps world to camera; [R|T] maps camera to screen

<details><summary>Answer</summary>

**A.** Rays are then defined from the camera centre through each pixel: X = P + t·d, with t > 0.

</details>

**Q214. [Week 10: Materials]**

How do diffuse, specular and mirror materials differ (BRDF)?

- A. Diffuse reflects incoming light equally in all directions; specular reflects more in a particular direction; mirrors reflect only in one direction
- B. All three reflect light equally in all directions
- C. Diffuse materials emit no light at all
- D. Mirrors absorb all light

<details><summary>Answer</summary>

**A.** The BRDF gives the fraction of light from direction s reflected towards direction v. Lambert's law adds the cosine (dot product with the normal) factor.

</details>

**Q215. [Week 10: Deep MLPs]**

A 5-layer, 256-unit ReLU MLP fitted to an image misses details. Which fix did the lecture suggest?

- A. Add batch normalisation to normalise features between layers and thus the gradients in deep networks
- B. Remove all activation functions
- C. Use fewer training pixels
- D. Replace the MLP with k-means

<details><summary>Answer</summary>

**A.** Deep networks can struggle to backpropagate gradients, and normalisation helps. Fourier features also address missing high-frequency detail.

</details>

**Q216. [Week 10: Geometric transforms]**

How can an image represented by an MLP I(x, y) be rotated, scaled or warped?

- A. Transform the input coordinates (e.g. X' = (T + P + S)·R·X) before querying the MLP
- B. Retrain the MLP from scratch for every rotation
- C. Rotate the weights of the first layer only
- D. It cannot be transformed

<details><summary>Answer</summary>

**A.** Since the image is a function of coordinates, warping the coordinates warps the image, including projective transformations.

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

</details>

**Q218. [Week 11: LLM refresher]**

What is the typical pipeline of an LLM forward pass?

- A. Text → tokens → embeddings + positional encoding → N transformer blocks → probabilities for the next token over the vocabulary (~50,000)
- B. Text → pixels → convolutions → class label
- C. Tokens → k-means → decision tree
- D. Noise → generator → discriminator → text

<details><summary>Answer</summary>

**A.** In the lecture example, 9 tokens with 16-dim embeddings produce ~50,000 next-token probabilities.

</details>

**Q219. [Week 11: Attention matrix]**

After applying softmax to QKᵀ, what property does each row of the attention matrix A have?

- A. It sums to 1 (it is a set of weights over the tokens)
- B. It sums to 0
- C. All values are negative
- D. It is always the identity

<details><summary>Answer</summary>

**A.** A = softmax(QKᵀ/√D), then AV mixes the values. With a causal mask the future entries are 0.

</details>

**Q220. [Week 11: DINO]**

Which technique does DINO use to learn image features without labels?

- A. Self-distillation: a student and a teacher (an EMA of the student) see different augmented views; cross-entropy matches their outputs, with centering and sharpening to avoid collapse
- B. A GAN discriminator that classifies real and fake
- C. Supervised training on ImageNet labels
- D. Next-token prediction on text

<details><summary>Answer</summary>

**A.** A stop-gradient (.detach()) is applied to the teacher. Centering prevents mode collapse, where the teacher outputs a constant.

</details>

**Q221. [Week 11: DINO]**

In DINO, what is the purpose of centering the teacher's features?

- A. To avoid mode collapse, where the teacher collapses to a constant output
- B. To increase the learning rate
- C. To make the student a copy of the teacher
- D. To remove data augmentation

<details><summary>Answer</summary>

**A.** Centering and softmax sharpening balance each other. Interestingly, the best classification used kNN on DINO features.

</details>

**Q222. [Week 11: VAE trade-off]**

What latent-size trade-off does a VAE face?

- A. Small latent (e.g. 64): blurry reconstructions and blurry random samples; large latent (e.g. 2048): sharp reconstructions, but random samples are rubbish
- B. Larger latent always gives better random samples
- C. Latent size has no effect
- D. Small latent gives sharp reconstructions

<details><summary>Answer</summary>

**A.** Most locations in a large latent space are invalid, so random z decodes poorly.

</details>

**Q223. [Week 11: Latent diffusion]**

How does Stable Diffusion (latent diffusion) generate images?

- A. A VAE reduces the dimension; diffusion runs in the VAE latent space from Gaussian noise to a valid latent with a UNet (with transformers), conditioned on text; the decoder outputs the image
- B. Diffusion runs directly on full-resolution pixels with no autoencoder
- C. A GAN generates the image in one step from text
- D. An LLM writes the pixel values as text

<details><summary>Answer</summary>

**A.** Conditions (text, images, maps…) are encoded and fed into the UNet features. Diffusion finds the valid latent locations the VAE alone cannot.

</details>

**Q224. [Week 11: Latent diffusion]**

Why combine a VAE with diffusion (or flow matching) in modern generative AI?

- A. Most latent locations of an AE/VAE are invalid; diffusion takes many small steps from low-dimensional noise to valid latents, while working in the latent keeps it cheap
- B. Because VAEs cannot reconstruct images
- C. Because diffusion cannot handle noise
- D. To remove the need for training

<details><summary>Answer</summary>

**A.** Most modern generative AI generates the latent of a VAE, then decodes it.

</details>

**Q225. [Week 11: Architectures]**

Which architectural trend did the lecture highlight for image models?

- A. Transformer blocks inside UNets are now common (e.g. for medical imaging and diffusion denoisers)
- B. UNets have been replaced by decision trees
- C. Transformers cannot be used for images
- D. Skip connections are no longer used

<details><summary>Answer</summary>

**A.** Combining convolutional multiscale structure with attention is the state of the art.

</details>

**Q226. [Week 11: Intelligence]**

What argument do Yann LeCun and others make about intelligence and generative models?

- A. Intelligence does not require a decoder (generating data) but is based on prediction, in a representation (feature) space
- B. Intelligence requires generating perfect images
- C. Only GANs can be intelligent
- D. Intelligence is impossible to model

<details><summary>Answer</summary>

**A.** Humans don't generate perfect images, and many narrow AI tasks work on features (Enc → z → classifier).

</details>

**Q227. [Week 11: JEPA]**

What characterises a joint-embedding predictive architecture (JEPA) compared with contrastive learning?

- A. It predicts the embedding of a target from a context directly in representation space, without negative samples (non-contrastive)
- B. It requires carefully chosen negative pairs
- C. It reconstructs pixels with a decoder
- D. It is a supervised classifier

<details><summary>Answer</summary>

**A.** Contrastive learning is difficult because good positive and negative pairs are hard to get and training is long. JEPA avoids negatives.

</details>

**Q228. [Week 11: Contrastive]**

In contrastive learning of an encoder, what are positive and negative pairs meant to achieve?

- A. Same object → same (close) features; different objects → different (far) features
- B. Same object → far features; different objects → close features
- C. All features should be identical
- D. Features should equal the pixels

<details><summary>Answer</summary>

**A.** The difficulty is generating good pairs and the long training. That's the motivation for non-contrastive methods.

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

</details>

**Q230. [Week 12: Revision: methods]**

Which method defines clusters by conditions on each axis (e.g. X > 1, Y > 0)?

- A. Decision tree
- B. SVM
- C. k-means
- D. Artificial neuron

<details><summary>Answer</summary>

**A.** Each split is an axis-aligned threshold, and a random forest is an ensemble of such trees.

</details>

**Q231. [Week 12: Revision: errors]**

A Type II error (medical term) corresponds to:

- A. A false negative / miss detection: 'not very good at a task'
- B. A false positive / false alarm: 'overdo a task'
- C. A true positive
- D. A correct rejection

<details><summary>Answer</summary>

**A.** Type I = false positive (false alarm). Type II = false negative (miss).

</details>

**Q232. [Week 12: Revision: metrics]**

Why is accuracy rarely a good performance target?

- A. It is an amalgam of the two types of mistakes, which usually have very different costs
- B. It cannot be computed for binary problems
- C. It is always 100% on the training set
- D. It ignores correct predictions

<details><summary>Answer</summary>

**A.** Choose the performance target carefully, e.g. sensitivity for screening and specificity for diagnosis.

</details>

**Q233. [Week 12: Revision: experimental design]**

Ideally, where should the testing data come from?

- A. Secret data kept to benchmark the model after training, ideally not from the same source as training/validation data
- B. The same batch as the training data
- C. Randomly generated noise
- D. The validation data reused

<details><summary>Answer</summary>

**A.** Training and validation data are split (e.g. 90/10) to pick hyperparameters. Testing estimates real-world performance.

</details>

**Q234. [Week 12: Revision: typical approach]**

In the typical ML approach, when is gradient-based optimisation needed instead of an analytical solution?

- A. When no analytical solution exists (e.g. due to activation functions) and the parameter space is too large to inspect exhaustively
- B. Always, even for linear least squares
- C. Never; every model has a closed-form solution
- D. Only when there is no data

<details><summary>Answer</summary>

**A.** Get data → choose the simplest model → analytical solution? → can the parameter space be inspected (grid search)? → otherwise optimise → analyse results on the test set.

</details>

**Q235. [Week 12: Revision: ANN geometry]**

In the geometric view of an ANN, what does each layer do?

- A. It maps the input feature space into another feature space where, ideally, classes become linearly separable for the next layer
- B. It sorts the training samples
- C. It computes the confusion matrix
- D. It normalises labels

<details><summary>Answer</summary>

**A.** In the end the output is a 1D problem where regression or classification is easy.

</details>

**Q236. [Week 12: Revision: generative]**

Given p(x, y) = p(y|x)p(x), when is a classifier's decision uncertain even if p(red|x) is high?

- A. When p(x) is low: the input is far from the training data distribution
- B. When p(x) is high
- C. When p(y|x) equals 1 and p(x) equals 1
- D. Never; a high p(red|x) always means a certain decision

<details><summary>Answer</summary>

**A.** That's why learning p(x) (generative modelling) helps detect out-of-distribution inputs.

</details>

**Q237. [Week 12: Revision: generative]**

Which pairing of method and way of learning p(x) is correct?

- A. GMM: maximum likelihood via EM; VAE: latent variables + approximate q(z|x), maximise ELBO; diffusion: learn the score (SDE); flow matching: learn a velocity field (ODE)
- B. GMM: ELBO; VAE: EM; diffusion: grid search; flow matching: k-means
- C. All four use a GAN discriminator
- D. None of them learn p(x)

<details><summary>Answer</summary>

**A.** That's the 'Finding p(x)' revision map in week 12.

</details>

**Q238. [Week 12: Revision: INR]**

Which summary of implicit neural representations is correct?

- A. An MLP can approximate any function; it can model a 2D image or 3D scan, the density and colour of a 3D scene from 2D projections (NeRF), or a sum of 3D Gaussians (splatting)
- B. INRs only work for text
- C. INRs require labelled classes
- D. INRs are a type of decision tree

<details><summary>Answer</summary>

**A.** That's the week 12 summary of week 10.

</details>

**Q239. [Week 12: Revision: key concepts]**

Which of the following is NOT a method to prevent overfitting in neural networks?

- A. Increasing the number of parameters while training longer on the same small dataset
- B. Dropout
- C. Data augmentation
- D. Early stopping on validation loss

<details><summary>Answer</summary>

**A.** More capacity and longer training on small data increases overfitting.

</details>

**Q240. [Week 12: Revision: safety]**

Which list matches the three 'what could go wrong' failure modes repeated in weeks 1, 5 and 12?

- A. Unknown data (failing to generalise), data drift, and bias
- B. Learning rate, batch size, and epochs
- C. Convolution, pooling, and stride
- D. Precision, recall, and F1

<details><summary>Answer</summary>

**A.** 1) New data unlike training, 2) the data changes after deployment, 3) wrong mapping for subgroups.

</details>

**Q241. [Week 12: Revision: transformers]**

Text, images and other data can all be processed by the same transformer model. What is needed to make this possible?

- A. Convert the data into a sequence of tokens (words, patches…) projected to embeddings with positional encoding
- B. Convert all data to grey-scale images
- C. Use a separate decision tree for each data type
- D. Remove the attention mechanism

<details><summary>Answer</summary>

**A.** Words become tokens, images become patches. The same attention blocks then apply.

</details>

**Q242. [Week 12: Revision: CNN]**

Which set lists ANN/CNN key concepts that are examinable?

- A. Backpropagation, overfitting, dropout, convolution, pooling/stride, batch norm, ResNet/UNet, transfer learning
- B. Only the history of AI
- C. Only Python syntax
- D. Only GAN training

<details><summary>Answer</summary>

**A.** The week 12 revision slide groups these as ANN and CNN key concepts, plus adversarial and backdoor attacks, saliency maps and responsible AI.

</details>

**Q243. [Week 12: Revision: unsupervised]**

In unsupervised k-means, each cluster is represented by:

- A. One point at its centre; each data point is assigned to the closest centre, so the space is divided by straight lines
- B. A full Gaussian with covariance
- C. A set of axis-aligned conditions
- D. A line with a buffer zone

<details><summary>Answer</summary>

**A.** That's the simplest cluster model. GMM generalises it with Gaussians (shape and soft membership).

</details>
