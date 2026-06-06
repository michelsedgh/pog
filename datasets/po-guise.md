Pose-guided token selection for the recognition of activities of daily living
Ricardo Pizarro a ,
∗
, Roberto Valle b
Luis Baumela b
, José M. Buenaposadac
, Luis M. Bergasaa
,
a Departamento de Electrónica, Universidad de Alcalá, Alcalá de Henares, Spain
b Departamento de Inteligencia Artificial, Universidad Politécnica de Madrid, Madrid, Spain
c Departamento de Informática y Estadística, Universidad Rey Juan Carlos, Móstoles, Spain
A R T I C L E I N F O
A B S T R A C T
Keywords:
Activities of daily living recognition
Efficiency in transformers
Token selection
Motion heatmaps
Large pre-trained video transformers are becoming the standard architecture for video processing due to their
exceptional accuracy. However, their computational complexity has been a major obstacle to their practical
application in problems that require the recognition of precise motion patterns in video, such as in the
recognition of Activities of Daily Living (ADL). Techniques like token pruning help mitigate their computational
cost, but overlook some specific aspects of this task such as the actor movement. To address this we propose
an improved token selection method that integrates semantic information from the ADL recognition task with
that of human motion. Our model relies on a multi-task architecture that infers human pose and activity
classification from RGB videos. We show that guiding token pruning with motion information significantly
improves the trade-off between higher efficiency, obtained by reducing the number of tokens, and accuracy
of the classification task. We evaluate our model on three popular ADL recognition benchmarks with their
respective cross-subject and cross-view setups. In our experiments, a video transformer modified with our
proposed modules sets a new state-of-the-art on the ADL recognition task whilst achieving significant reductions
in computational cost.
1. Introduction
Activities of Daily Living (ADL) encompass the fundamental tasks
of daily life, such as eating, cooking, and managing medications. They
play a crucial role in assessing a person’s ability to function indepen-
dently. Their recognition is used to monitor the elderly or people with
disabilities and to evaluate their functional ability in conditions such
as dementia, stroke, or age-related decline. The models and techniques
of computer vision used to recognize them share similarities with the
broader field of human action recognition. However, ADLs present
specific challenges, such as the existence of short and subtle actions
that exhibit a similar visual appearance but differ in motion [1]. This
requires the precise analysis of human body motion patterns within the
video’s spatio-temporal context.
In the recognition of human actions we have seen a transition from
methods using CNNs [2–4] and 3D-CNNs [5–7] or a mixture of both [4]
to transformers [8–10]. Using self-supervised learning techniques and
large-scale datasets, recent video transformer models achieve the high-
est accuracy on the human action recognition problem [11]. A key
limitation in using these models to analyze video is their quadratic com-
plexity, which increases the computational demands as the number of
spatio-temporal tokens grows. Although progress has been made in this
area, there is still considerable room for improvement, especially for
recognizing subtle motions and when the trade-off between accuracy
and efficiency is of practical relevance. Both are crucial ingredients in
making the recognition of ADL a household product. Applications such
as fall detection or ensuring that medication is taken correctly demand
real-time performance, making computationally expensive models im-
practical.
One technique to achieve a better trade-off between accuracy and
efficiency is token selection, where a percentage of tokens are discarded
at certain blocks within the transformer model, reducing the total
number of tokens in the model. Popular techniques include Top-K [12],
where token selection is guided by keeping the K tokens with the
greatest attention to the class token, merging similar tokens [13], or
a mixture of both [14–16]. However, these techniques often lack con-
sideration for factors such as human pose and its temporal dynamics.
This can lead to suboptimal performance in ADL scenarios that require
a nuanced understanding of human actions, resulting in a potential loss
of critical information.
In this paper, we present a token selection method for transformer
models that integrates semantic information from both the activity
∗ Corresponding author.
E-mail address: ricardo.pizarroc@edu.uah.es (R. Pizarro).
https://doi.org/10.1016/j.imavis.2025.105686
Received 19 February 2025; Received in revised form 13 July 2025; Accepted 26 July 2025
Available online 5 August 2025
0262-8856/© 2025 The Authors. Published by Elsevier B.V. This is an open access article under the CC BY-NC license ( http://creativecommons.org/licenses/by-
nc/4.0/ ).
R. Pizarro et al.
Image and Vision Computing 162 (2025) 105686
Fig. 1. Attention maps for the ‘‘Drink.Frombottle’’ action on Toyota-Smarthome (CS) [17]. Colored rectangles represent the attention weight assigned by each visual token to
the classification token, lighter yellow rectangles indicating a low attention from that token. PO-GUISE concentrates attention on task-relevant regions, improving computational
efficiency by discarding irrelevant tokens. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
recognition task and human motion. We aim to improve the atten-
tion of the transformer on the actor’s motion and, at the same time,
reduce computational requirements of the model. Our module can
be integrated on ViT-based architectures such as InternVideo2 [18]
and VideoMAEv2 [11]. These transformer architectures are pre-trained
with a self-supervised strategy and refined with a large human action
database. Our method, called PO-GUISE, is trained in a multi-task fash-
ion using RGB videos. They are converted into spatiotemporal visual
tokens and are processed alongside heatmap tokens representing tem-
poral representations of human poses. We have extended the traditional
heatmap to predict the motion of the keypoints of multiple actors in
video. Our token selection method prunes spatiotemporal visual tokens,
referred to as visual tokens, that do not pay enough attention to semantic
tokens, those relevant to human motion and action recognition. To
ensure that information is not lost during pruning, our merging method
summarizes the pruned tokens by averaging similar dropped tokens.
Fig. 1 shows that our method selects tokens primarily on the actor,
while the baseline model focuses on potentially irrelevant parts of the
scene. To our knowledge, we are the first to improve the accuracy of
transformer models for ADL recognition while reducing its computa-
tional cost using human pose and motion information. Moreover, our
approach does not require an external keypoint detection model. In
summary, we pioneer the introduction of human motion information
into the token selection process in the video transformer architecture.
The contributions of our work are as follows.
• A token selection method guided by human motion and class
information tailored to the recognition of activities of daily living.
Focuses the attention of the model on the motion of the actor and
improves the trade-off between efficiency and accuracy compared
to other methods from the state-of-the-art, even at very low token
keep rates.
• A representation of human motion based on a feature map shared
by all body keypoint temporal heatmaps, that is agnostic of the
number of people in the scene and allows our method to be used
on multi-actor datasets.
• Our method sets a new state-of-the-art in various activities of
daily living RGB video benchmarks, while being much more
efficient than other top performing methods based on video trans-
formers.
2. Related work
In this section, we review the human action recognition and activ-
ities of daily living literature. Recognizing actions in videos requires
considering variations in the location and poses of actors within the
scene, as well as their movement.
Human Action Recognition and ADL. One way to analyze motion
in videos is to compute convolutions in both the image and the time
dimensions with 3D CNNs [5]. A popular approach is the two-stream
CNN [2–4] that uses both RGB and optical flow maps. However, optical
flow only gives short temporal scale information. More recent work
use a Recurrent Neural Network (RNN) [19] on top of a two-stream
network [3] to process a longer but still limited temporal context.
The adoption of video transformers in action recognition allows for
a holistic temporal context to be established [8–10], although with
quadratic complexity in the number of visual tokens.
The human pose and its realization in the form of probability
maps, or heatmaps, corresponding to the location of body keypoints
has proven to be very discriminative in action recognition [20–27].
Many previous studies have used an external human pose estimation
model [21–23,28–31]. This is also the case with recent transformer
based methods [25–27]. Having an external pose estimation model not
only increases the computational cost but also decreases the system
robustness in situations where the external model fails. Few methods
adopt a multi-task strategy to estimate pose and recognize actions in
the same model [19,32]. A recent approach achieves top performance
in the recognition of activities of daily living by combining 2D and 3D
human pose [10]. In our solution we also adopt a multi-task strategy.
However, unlike these approaches, we use human pose to select the
most informative video tokens by guiding the model’s attention to
human motion, while reducing the computational requirements of the
model.
Computational requirements of Video Transformers. The
quadratic complexity in the number of tokens in a transformer is a
fundamental limitation for its use in real-time video analysis. This
problem can be addressed in different ways. Some methods modify
the attention mechanism itself to reduce this quadratic complexity. For
example, one approach is to factorize attention along the spatial and
temporal dimensions [33]. Another is to restrict attention to small local
windows and shift these windows hierarchically [34].
Another approach is token selection, in which a dedicated mecha-
nism prunes or merges the visual tokens processed by the network, dis-
carding those considered irrelevant to the task. This is achieved while
preserving the integrity of the transformer’s weights and underlying
architecture.
Token selection methods can be categorized into pruning or merging
strategies. Token pruning methods focus on identifying and removing
less informative tokens. EViT [14], which uses a Top-K approach,
2
R. Pizarro et al.
selects the K tokens with the highest attention to the class token, where
the non-selected tokens are fused into one token. PPT [35], introduces
a learnable token per body keypoint and uses their attention values to
prune visual tokens. The main limitation of PPT is the fixed number
of keypoint tokens used in training, which limits the number of actors
in the scene. EVAD [9], leverages attention to visual tokens on a key-
frame to determine which tokens to retain. The TPS (Token Pruning
and Squeezing) module [15], is a module for image transformers. It
uses a Top-K token pruning step and a squeeze step that merges the
non-selected tokens into the selected ones via matching and similarity-
based fusing. Another form of guiding pruning from image information
is based on patches, where inter-patch attention and dynamic prun-
ing are applied to take advantage of the rich structure of the patch
relations [36].
Token merging techniques combine similar tokens to reduce redun-
dancy, such as ToMe [13], which merges similar tokens, as dictated
by their cosine similarity, into new ones. DTMFormer [37], which
adaptively clusters tokens into fewer semantic tokens via an attention-
guided mechanism. Another technique is a partitioned token fusion
and pruning strategy. It discards low-correlation background token
information and fuses medium-correlation token. This technique has
been applied to the field of object tracking [16].
Haurum et al. [12] provides a systematic comparison of ten popular
token reduction methods, finding that pruning-based methods such as
Top-K and EViT [14] consistently perform best.
However, a significant limitation of existing token selection meth-
ods is their lack of task-specific considerations. Specifically for the
ADL task, these methods do not account for the human pose and its
temporal dynamics directly, potentially resulting in the loss of crucial
information.
Our proposal. We present a novel token selection method guided
by both temporal human pose heatmaps and ADL. We use a multi-
task strategy, estimating both human motion heatmaps and activity,
which differs from the usual and less efficient approach using exter-
nally provided landmarks [25–27]. While approaches like 𝜋-ViT [10]
also leverage pose, they do so only as a training aid to improve the
base model’s representations, discarding the pose-related modules at
inference and thus not reducing final computational complexity. Our
strategy focuses the attention of the model on the actor’s movements
and reduces the computational complexity of the transformer. Further-
more, our motion heatmap representation inherently supports scenes
with multiple actors, a key advantage over methods like PPT [35],
which are constrained to a predefined number of individuals. As a re-
sult, PO-GUISE maintains or even enhances the accuracy of the baseline
model. In addition, its accuracy decreases much more slowly than that
of other token selection methods at very low computational budgets.
Compared with the baseline model, PO-GUISE in default settings re-
duces computation by a remarkable 30% and improves the accuracy
by 0.55, 1.74 and 3.84 in the NTU60, NTU120 and Toyota-Smarthome
datasets, respectively, in the cross-subject protocol (see Tables 5 and
4).
3. POse-GUIded multi-task video transformer with token SElection
(PO-GUISE)
Our approach incorporates a pre-trained video transformer [11,18]
as its encoding mechanism. The video transformer is fine-tuned in dif-
ferent action recognition datasets. To facilitate human body keypoints
localization and guide our token selection, we have integrated the pose
heatmaps prediction and action classification tasks. Additionally, to
mitigate the computational demands associated with video transformer
models, we introduce the PO-GUISE module, which effectively reduces
the number of visual tokens. A comprehensive visual representation of
our model is given in Fig. 2. In the following sections, we provide a
detailed explanation of each component within our model.
Image and Vision Computing 162 (2025) 105686
3.1. Video transformer and human-pose processing
Consider a video segment, or clip, with dimensions 𝑇 × 𝐶 × 𝐻 × 𝑊
where 𝑇 is the number of frames and 𝐶, 𝐻, 𝑊 are the channels, height,
and width of each frame, respectively. In our experiments, we define
𝑇 = 16, 𝐶 = 3, 𝐻 = 224 and 𝑊 = 224 respectively. To process a clip
with a video transformer [11], we use the joint space–time cube em-
bedding [33]. This technique samples non-overlapping cubes from the
input video clip, which are then fed into the embedding layer. It divides
the input video tensor into cubes of dimension 2 × 𝐶 × 16 × 16, resulting
in a set of 𝑁𝑣𝑖𝑠 = 𝑡⋅ ℎ⋅ 𝑤 visual tokens, where 𝑡 =
𝑇
𝐻
𝑊
2 , ℎ=
16 , 𝑤 =
16. We
then project tokens to 𝐷 dimensions using a linear embedding layer,
resulting in an input tensor with shape 𝑋𝑣𝑖𝑠 ∈ R𝑁𝑣𝑖𝑠 ×𝐷
. Next, we apply
a positional embedding to each token, and a learnable class token,
𝑋𝑐𝑙𝑠 ∈ R1×𝐷
, is concatenated to the sequence. For the computation
of human-pose heatmaps, our model incorporates 𝑁𝑝
= ℎ𝑚𝑟𝑒𝑠⋅ ℎ𝑚𝑟𝑒𝑠
learnable tokens into the input sequence defined as 𝑋𝑝 ∈ R𝑁𝑝 ×𝐷
, where
ℎ𝑚𝑟𝑒𝑠 defines the heatmap feature map resolution and total number of
tokens it is represented by. The complete sequence of tokens, including
the class, pose and visual tokens 𝑋 = (𝑋𝑐𝑙𝑠, 𝑋𝑝, 𝑋𝑣𝑖𝑠) ∈ R𝑁×𝐷 where
𝑁 = 1 + 𝑁𝑝 + 𝑁𝑣𝑖𝑠, is then processed using a standard ViT architecture.
The transformed class token 𝑋𝑐𝑙𝑠 is used in a multilayer perceptron
(MLP) for the classification task, while the 𝑋𝑝 pose tokens are passed
through a heatmaps estimation head to be compared against the ground
truth heatmaps for pose estimation (one heatmap per human body
keypoint).
3.2. Human-pose estimation task
A crucial part of our approach involves the use of temporal
heatmaps, which enhance the training process and facilitate token
selection. These heatmaps are derived from learnable tokens, similar
to those in PPT [35]. However, our method further refines PPT’s
image-only processing by extending its capabilities to handle a vari-
able number of keypoints, video inputs, and multi-person heatmap
predictions.
Heatmap prediction starts with the introduction of additional tokens
to the network, 𝑋𝑝. After passing through the encoder, these tokens
are processed by a lightweight decoder (Heatmap head) to convert the
tokens into heatmaps. The architecture of the Heatmap head consists of
two deconvolution layers followed by a convolution layer with a 1 × 1
kernel and with output channels equal to the number of landmarks
𝐿 [38]. The output of this decoder is then directly compared with the
ground truth heatmaps by measuring the mean-squared error.
While these tokens are inherently capable of predicting heatmaps
for an individual frame within a video clip, we can adapt them to
capture the entire sequence of movements by modifying the ground
truth labels. The use of heatmaps instead of coordinate representations
provides greater flexibility by allowing the incorporation of additional
information directly within the heatmaps, without requiring any struc-
tural changes to the network architecture. We generate time-aware
heatmaps by averaging the spatial heatmaps from the ground-truth
labels, a Gaussian centered at the location of each annotated landmark,
across the whole video clip. It results in a ground truth heatmap where
each keypoint movement within the clip is visible. Likewise, the frame-
work can be extended to predict multi-person heatmaps by combining
detection data from multiple individuals inside a single heatmap. In
Fig. 3 we show an example motion heatmap for the multi-actor case.
3.3. POse-GUIded token SElection module
The use of joint space–time cube embeddings for processing videos
is computationally expensive, which is not ideal for use in environ-
ments with limited computing power. Videos naturally contain repet-
itive information over time and areas with no information for action
3
R. Pizarro et al.
Image and Vision Computing 162 (2025) 105686
Fig. 2. Our architecture consists of 4 stages. An input clip is tokenized and processed by a ViT encoder alongside learnable class and heatmap tokens. Our token selection module
is inserted in the first three stages of the ViT encoder, reducing the number of tokens after each stage. The model outputs both the activity classification and the corresponding
motion heatmaps.
Fig. 3. Motion heatmap generation. We aggregate the movement of the keypoints through time into a single heatmap. The figure shows, from left to right, the left wrist keypoint
at three different time instants and the corresponding aggregated heatmap.
recognition. Thus, we propose the use of token pruning to reduce
computation without losing important content.
We introduce a novel approach named PO-GUISE. This method
leverages the informative content of the class and heatmap tokens to
improve the token selection process. Furthermore, to prevent the loss
of potentially valuable information, PO-GUISE also merges some of the
tokens that were not initially selected during the pruning step. This
merging step is crucial as it compensates for any potentially relevant
data that might not have been identified by the pruning algorithm. Fig.
2 shows an overview of this two-step token selection.
We integrate our token selection module into the transformer net-
work architecture at specific intervals. The ViT base architecture con-
sists of 12 layers, we divide these in 4 stages, where each stage consists
of 4,3,3,2 layers, respectively. We place the module at the output of
each of the first three stages. This results in a total of three token
selection layers within a ViT-base model (see Fig. 2). In doing so, our
goal is to strike a balance between reducing computational load and
maintaining the critical information necessary to efficiently process the
video.
3.3.1. Token pruning
Building upon existing token pruning methods like EVIT [14] and
EVAD [9], our approach introduces a novel integration of spatial in-
formation. Specifically, we leverage heatmap tokens to guide attention
towards visual tokens that correspond to actor locations. Let 𝑀 ∈
R𝑀×𝑁𝑣𝑖𝑠 ×(1+𝑁𝑝 ) be the attention tensor from 𝑀 heads, obtained from
processing the tokens in 𝑋 ∈ R𝑁×𝐷
, and then indexing by the attention
the visual tokens (𝑋𝑣𝑖𝑠) pay to the heatmap (𝑋𝑝) and class (𝑋𝑐𝑙𝑠) tokens.
We average across attention heads to condense it into an 𝑁𝑣𝑖𝑠 × (1 + 𝑁𝑝)
matrix, resulting in 𝑣𝑖𝑠 ∈ R𝑁𝑣𝑖𝑠 ×(1+𝑁𝑝 )
, see Fig. 4. We then multiply
by a small constant factor 𝜅, the class attention scores and by 1 − 𝜅,
the heatmap token attention scores to denote the relative importance
between them. Next, by summing the rows of 𝑣𝑖𝑠, we get a vector
of token scores,  ∈ R𝑁𝑣𝑖𝑠
. Each element in this tensor reflects the
aggregated importance of a visual token influenced by the attention
to the semantic tokens, (𝑋𝑐𝑙𝑠, 𝑋𝑝). The final pruning decision is based
on these aggregated scores, allowing us to retain visual tokens that are
deemed most significant in the context of both global class information
and local spatial heatmap cues. The computed attention score for the
𝑖th visual token can also be formulated as:
 (𝑖) = 𝑣𝑖𝑠(𝑖, 0)⋅ 𝜅 +
⎛ ⎜ ⎜ ⎝
𝑁𝑝
∑
𝑣𝑖𝑠(𝑖, 𝑗)
𝑗=1
⎞ ⎟ ⎟ ⎠⋅ (1 − 𝜅),
where 𝑣𝑖𝑠(𝑖, 𝑗) is the attention score from 𝑖th visual token to 𝑗th
semantic token, and 𝜅 is a constant factor to balance the importance
between class and heatmap tokens.
We use  to select the 𝑁𝑠𝑒𝑙 most significant tokens, based on their
calculated scores. The number of selected tokens is determined by
𝑁𝑠𝑒𝑙 = 𝑁𝑣𝑖𝑠⋅ 𝜌, where the keep rate 𝜌 is a predefined threshold in
the range (0, 1]. Resulting in a set of selected tokens, 𝑋𝑠𝑒𝑙 ∈ R𝑁𝑠𝑒𝑙 ×𝐷
,
and a set of discarded ones, 𝑋𝑑𝑖𝑠𝑐 ∈ R(𝑁𝑣𝑖𝑠−𝑁𝑠𝑒𝑙 )×𝐷
. 𝑋𝑠𝑒𝑙 which will be
processed in the next network block. Fig. 4 illustrates an overview of
the pruning step.
3.3.2. Token merging
The process of token pruning might exclude information that is
important for later processing stages, or information that is not imme-
diately apparent from examining the attention between classes and the
associated heatmaps. To mitigate this, we introduce a token merging
phase for the discarded tokens, 𝑋𝑑𝑖𝑠𝑐. This phase employs cosine sim-
ilarity to identify and merge tokens with highly aligned features. Our
approach adapts the merging strategy of ToMe [13] by implementing
an alternative matching algorithm that is better suited to our context.
Unlike ToMe, which initially partitions tokens into two sets, our algo-
rithm is more flexible, allowing the merging of an arbitrary number
of tokens. The number of output tokens in this phase is controlled by
𝑁𝑚𝑒𝑟𝑔𝑒
= 𝑁𝑑𝑖𝑠𝑐⋅ 𝜆 with 𝜆 being a predefined threshold in the range (0,
1]. Fig. 5 shows an overview of the merging method.
This phase begins with the use of the attention tensor 𝐴𝑑𝑖𝑠𝑐 obtained
from 𝑋𝑑𝑖𝑠𝑐. 𝐴𝑑𝑖𝑠𝑐 contains the attention between the visual tokens 𝑋𝑑𝑖𝑠𝑐.
We then use 𝐴𝑑𝑖𝑠𝑐 to compute the pairwise cosine similarity for these
4
R. Pizarro et al.
Image and Vision Computing 162 (2025) 105686
Fig. 4. Token pruning diagram. The attention obtained from 𝑋𝑣𝑖𝑠 guides the token pruning. Each row in 𝑣𝑖𝑠 corresponds to the attention a visual token (𝐴𝑡𝑡𝑣𝑖𝑠 ) pays to the class
(𝐴𝑡𝑡𝑐𝑙𝑠 ) and heatmap (𝐴𝑡𝑡ℎ𝑚 ) tokens. The Top-K tokens with most attention ( ) are selected as output of the step, while the non-selected go through a merging step.
Fig. 5. Token merging diagram. The discarded tokens from the previous pruning step are merged by their similarity. The similarity between tokens is measured by their attention
to each other (𝐴𝑑𝑖𝑠𝑐 ). The 𝑁𝑚𝑒𝑟𝑔𝑒 most similar tokens are selected and then merged with their corresponding most similar token.
tokens, generating a similarity matrix 𝑆 ∈ R𝑁𝑑𝑖𝑠𝑐 ×𝑁𝑑𝑖𝑠𝑐
. The diagonal
elements of 𝑆 are masked to prevent the tokens from merging with
themselves. Each row of 𝑆 represents the similarity of a specific token
to all other tokens within 𝐴𝑑𝑖𝑠𝑐.
Next, for each token in 𝑋𝑑𝑖𝑠𝑐 , we identify its merge candidate as
the token with the highest cosine similarity, according to the respec-
tive row in 𝑆. Subsequently, we select the 𝑁𝑚𝑒𝑟𝑔𝑒 tokens that exhibit
the strongest similarity to their respective candidates. This selective
aggregation ensures that the information from tokens with substantial
similarity is preserved. These selected tokens are then merged with
their corresponding candidates by averaging their feature vectors, re-
sulting in a new set of tokens, 𝑋𝑚𝑒𝑟𝑔𝑒 ∈ R𝑁𝑚𝑒𝑟𝑔𝑒 ×𝐷
. Finally, 𝑋𝑚𝑒𝑟𝑔𝑒 and
𝑋𝑠𝑒𝑙 are concatenated to be processed by the next network block. This
process ensures that potentially relevant information is not lost and is
passed on to subsequent layers. A detailed description of this module
can be found in Algorithm 1.
4. Experiments
In this section, we evaluate our multi-task video transformer. In all
experiments, HM(P) stands for spatio-temporal heatmaps computed for
multiple-person poses. PR stands for the use of token pruning by: C
using attention to the class token; MF using attention to the middle
frame visual tokens; or P using attention to the tokens used to compute
human motion heatmaps. MG stands for our proposal to merge pruned
tokens. PO-GUISE corresponds to adding +HM(P)+PR(C+P)+MG to the
baseline video transformer. Within each experiment, the results of the
model in the first, second and third positions are shown, respectively,
in bold, underline or double underline.
4.1. Datasets
We use popular ADL recognition datasets for evaluation: NTU60 [5],
NTU120 [39], and Toyota-Smarthome [17]. We employ two standard
evaluation protocols established in the datasets, cross-subject (CS) and
cross-view (CV) or cross-set (CSet). In the CS protocol, the training and
testing sets are split according to the identity of the subject, ensuring
that there is no overlap between actors. In the CV or CSet protocol,
different camera viewpoints are used for training and testing, while
all subjects are included in both sets. We present the overall accuracy
(𝐴𝑐𝑐.) or the average-per-class accuracy (mean class accuracy, 𝑚𝐶𝐴)
when appropriate due to the class imbalance present in some datasets.
NTU120 is a large-scale human action recognition data set for ac-
tivities of daily living. It features 114K videos, multiple camera views,
106 subjects, and 120 different classes. We follow the cross-subject
5
R. Pizarro et al.
Algorithm 1 Token Merging
1: 𝑋𝑑𝑖𝑠𝑐 ∈ R𝑁𝑑𝑖𝑠𝑐 ×𝐷 : Feature tensor of unselected tokens
2: 𝐴𝑑𝑖𝑠𝑐 ∈ R𝑁𝑑𝑖𝑠𝑐 ×𝐷 : Attention tensor of unselected tokens
3: 𝑆 ∈ R𝑁𝑑𝑖𝑠𝑐 ×𝑁𝑑𝑖𝑠𝑐 : Similarity matrix
4: 𝑘: Number of tokens to merge based on similarity
5: 𝑋𝑚𝑒𝑟𝑔𝑒𝑑 ∈ R𝑁𝑚𝑒𝑟𝑔𝑒 ×𝐷 : Merged feature tensor
6: // Compute cosine similarity for discarded tokens
7: for 𝑖 = 1 to 𝑁𝑑𝑖𝑠𝑐 do
8: for 𝑗 = 1 to 𝑁𝑑𝑖𝑠𝑐 do
𝐴𝑑𝑖𝑠𝑐𝑖⋅𝐴𝑑𝑖𝑠𝑐𝑗
9: 𝑆𝑖𝑗 ←
‖𝐴𝑑𝑖𝑠𝑐𝑖 ‖‖𝐴𝑑𝑖𝑠𝑐𝑗 ‖ 10: end for
11: end for
12: 𝑆 ← 𝑆− diag(diag(𝑆)) 13: // Identify merge candidates based on similarity
14: for 𝑖 = 1 to 𝑁𝑑𝑖𝑠𝑐 do
15: 𝑚𝑒𝑟𝑔𝑒_
𝑐𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒[𝑖] ← Max(𝑆𝑖,∶)
16: end for
17: // Select the top-k most similar tokens based on 𝑆
18: 𝑚𝑒𝑟𝑔𝑒_
𝑖𝑛𝑑𝑖𝑐𝑒𝑠 ← argsort(𝑚𝑒𝑟𝑔𝑒_
𝑐𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒)[∶ 𝑘]
19: // Merge source tokens with the selected ones by
20: 𝑋𝑚𝑒𝑟𝑔𝑒𝑑 ← mean(𝑋𝑑𝑖𝑠𝑐 [𝑚𝑒𝑟𝑔𝑒_
𝑖𝑛𝑑𝑖𝑐𝑒𝑠], 𝑎𝑥𝑖𝑠 = 0)
21: return 𝑋𝑚𝑒𝑟𝑔𝑒𝑑
⊳ Cosine similarity
⊳ Set diagonal to zero
protocol (CS), where train-test sets feature different subjects, and cross-
setup (CSet) protocol which uses different camera setups in training and
testing. The NTU60 dataset is a subset that contains only 57K videos,
40 subjects, and 60 classes. We follow the CS and CV protocols. For
both NTU datasets we report the overall accuracy (𝐴𝑐𝑐.).
Toyota-Smarthome is a dataset for activities of daily living per-
formed by seniors. The dataset consists of 16K RGB clips of 31 activity
classes performed by 18 subjects and 7 different camera viewpoints.
We evaluate using the cross-subject (CS) protocol with 31 classes. We
also use two cross-view protocols, CV1 and CV2, both of which use
a 19-class subset and cameras 2 and 5 for testing and validation,
respectively. For training, CV1 uses only camera 1 while CV2 uses
cameras 1, 3, 4, 6, and 7. We report the mean class accuracy (𝑚𝐶𝐴).
4.2. Implementation details
Unless otherwise stated, we use a ViT-base model with pre-trained
weights from VideoMAEv2 [11]. These have been distilled from the pre-
trained ViT-giant model vit_b_k710_dl_from_giant. For classification, we
use cross-entropy loss and log-scaled MSE for heatmap prediction. We
also use Nash-MTL [40] to balance both tasks. Ground truth heatmaps
are created by taking the available landmarks in each dataset and
transforming them into heatmaps using the gaussian UDP heatmap
technique [41]. We set the heatmap resolution to ℎ𝑚𝑟𝑒𝑠 = 8. We use the
AdamW [42] optimizer with a Cosine Annealing learning rate sched-
uler [43]. Data augmentation includes Cutmix [44] (CMx), Mixup [45]
(MxU) and RandAug [46]. For our PO-GUISE model, we set pruning
keep rate to 𝜌 = 0.6 and merge keep rate to 𝜆 = 0.3 in all experiments
unless otherwise stated.
All of our experiments are done on an NVIDIA DGX server with
4 A100-80 GB GPUs. Training is done using Pytorch 2.3 [47], and a
hyperparameter search is done on the learning rates using Wandb [48]
with a Bayesian search on validation loss.
For both NTU120 and NTU60 we follow the official implementation,
discarding the examples where no pose was recorded. The detailed
hyperparameters used for the experiments in NTU60, NTU120, and
Toyota-Smarthome can be seen in Table 1.
At inference we crop the central part of the frame in NTU with full
height, keeping the aspect ratio and resizing it to 224 × 224 pixels
Image and Vision Computing 162 (2025) 105686
Table 1
Training parameters used in the main paper experiments.
Configuration Toyota-SM NTU/Toyota-SM
(CV) All/(CS)
Pre-trained weights vit_b_k710_dl_from_giant
MSE scaling factor 1000
Learning rate backbone 0.00007 0.0001
Learning rate heads 0.0003 0.0006
Optimizer Adamw
Learning rate scheduler Cosine Annealing
RandAug. M 7
RandAug. N 4
label smoothing 0.1
CMx & MxU prob. 1.0
CMx & MxU switch prob. 0.5
Gradient clipping 1.5
accumulate_grad_batches 2
Batch size 16
Merge feat. sim. matrix Attention
Epochs 350
Early stopping 30
#Landmarks 13 25/13
PO-GUISE 𝜌 0.6
PO-GUISE 𝜆 0.3
Table 2
Ablation study. Test results on Toyota-Smarthome (CS) and NTU60 (CS) using different
model configurations. VideoMAEv2-base is the baseline experiment and the rest are
independent experiments adding something to baseline.
Method Toyota-SM NTU60 GFlops
mCA. Acc. (↓)
(↑) (↑)
VideoMAEv2-base (baseline) 73.14 94.29 360
+PR(C) 73.30 93.45 232
+PR(MF) 70.77 94.09 232
+PR(C)+MG 73.89 94.10 232
+HM(P) 76.01 94.47 379
+HM(P)+PR(C) 74.94 93.93 249
+HM(P)+PR(C+P) 75.41 94.57 249
+HM(P)+ToMe 73.80 88.35 190
+HM(P)+PR(C+P)+ToMe 74.65 93.84 249
+HM(P)+PR(C+P)+MG 76.98 94.84 249
and each labeled clip was sampled uniformly over time. With Toyota-
Smarthome we use the same cropping strategy as in NTU. We follow
the official implementation and temporally divide each labeled clip into
4-s samples (128 frames). We reach the final classification by averaging
the logits of the samples from each clip.
4.3. Ablation study
For the ablation experiments (see Table 2), we use the Toyota-
Smarthome and NTU60 datasets following in both cases their cross-
subject procedure. Our baseline result is obtained by fine-tuning a
state-of-the-art video transformer, VideoMAEv2 [11] pre-trained in
Kinetics [4]. The accuracy for the baseline is 73.14 and 94.29 in
Toyota-Smarthome and NTU60, respectively.
4.3.1. Comparison with baseline
First, we test the baseline plus semantic information in the form
of a human pose estimation task, see baseline+HM(P) in Table 2. On
average, it increases the accuracy of all actions by 2.87 and 0.18
points in Toyota-Smarthome and NTU60, respectively. Pose informa-
tion provides a significant improvement in the accuracy of some ac-
tions. A small drawback is the increased computational cost of 5%
more GFLOPS, due to the extra tokens that need to be processed for
the human pose estimation.
6
R. Pizarro et al.
Image and Vision Computing 162 (2025) 105686
Method Toyota-SM GFlops
mCA. (↑) (↓)
VideoMAEv2-base 73.14 360
+PO-GUISE 76.98 249
Internvideo2 75.64 509
+PO-GUISE 77.03 399
To demonstrate the flexibility of PO-GUISE and its ability to be
integrated into other ViT-based backbones, we have performed an
additional experiment using InternVideo2-B/14 [18], see Table 3. In-
ternvideo2 increases the accuracy of VideoMAE by 2.5, but with 41%
more GFLOPS. With this model, the behavior of PO-GUISE is similar. It
reduces the number of GFLOPS by a remarkable 27% while increasing
the accuracy by 1.39. In the rest of the paper we use VideoMAEv2-
base as the backbone due to the low gains obtained by the Internvideo2
backbone.
Fig. 6. Per-class accuracy comparison on Toyota-Smarthome (CS). We show results for the baseline model (VideoMAEv2-base), Top-K pruning (PR(C)), and PO-GUISE. We have
merged some classes for an easier visualization.
We also compare different methods of token selection from the
state-of-the-art on the baseline model while maintaining similar
GFLOPS for each experiment. We test Top-K pruning by attention to the
class token [12], baseline+PR(C), pruning by attention to the middle
frame visual tokens [9], baseline+PR(MF), and adding our token merg-
ing solution to the class token pruning, baseline+PR(C)+MG. We find
that for all configurations there is a loss in accuracy when compared
to the baseline. In Toyota-Smarthome, utilizing PR(MF), similar to the
method in EVAD [9], resulted in a larger loss in accuracy than with
PR(C), −2.37 vs. +0.16. This means that the visual tokens in the middle
frame are not as informative compared to relying only on the class
token for token selection. The use of PR(C)+MG resulted in a small
performance gain of 0.75 in Toyota-Smarthome while in NTU60 we
obtain a small reduction of 0.19. This suggests that merging tokens
is beneficial in preserving valuable information that pruning alone
may not capture. This is crucial for maintaining model accuracy while
increasing computational efficiency. Note here that token pruning
reduces GFLOPs by 35% (360 to 232) and merging does not add a
significant amount of processing.
The last set of experiments in Table 2 assesses the influence of differ-
ent token selection methods in the multi-task model, baseline+HM(P).
The first interesting result is that pruning guided by the class token,
baseline+HM(P)+PR(C), affects the performance of the model, 1.07 and
0.54 less accuracy than baseline+HM(P) for both Toyota-Smarthome
and NTU60. However, we found that our token pruning guided by
class and pose tokens, baseline+HM(P)+PR(C+P), outperforms pruning
based solely on class information, baseline+HM(P)+PR(C), by 0.47
and 0.64. In addition, employing the entire PO-GUISE model (base-
line+HM(P)+PR(C+P)+MG) yields an additional improvement of 2.04
and 0.91 over PR(C). We perform additional experiments to com-
pare with the ToMe merging method [13]. The combination of base-
line+HM(P)+PR(C+P)+ToMe shows a reduction of 2.33 in accuracy
compared to PO-GUISE with our token merging procedure. Lastly, PO-
GUISE model achieves a reduction in GFLOPS around 34% while also
increasing the accuracy by 0.97 and 0.37 over the baseline+HM(P).
These results highlight the effectiveness of pose-guided pruning and
the merging process in efficiently selecting task-relevant tokens. In Fig.
6 we show the per-class-accuracy of our method against the baseline
model and the Top-K (PR(C)) pruning technique. PO-GUISE obtains
an improvement across virtually all classes. The improvement is most
notable in classes that require the recognition of fine-grained actions,
such as ‘‘Use telephone’’, ‘‘Cut bread’’, and ‘‘Make tea’’, where our
method significantly outperforms the baseline.
Table 3
Test results on Toyota-Smarthome (CS) with RGB-only modality at inference.
4.3.2. Efficiency analysis
In this experiment we explore the trade-off between accuracy and
computational cost incurred by different token selection methods ap-
plied on the multi-task model, baseline+HM(P). In Fig. 7 we show
the curves of GFLOPS vs. accuracy obtained by training with differ-
ent values of 𝜌 and 𝜆. For the experiments +HM(P)+PR(C+P) and
+HM(P)+PR(C) 𝜌 ∈ {0.3, 0.4, 0.55, 0.7}. For the +HM(P)+PR(C+P)+MG
experiments, 𝜌 ∈ {0.3, 0.4, 0.45, 0.6} and 𝜆 ∈ {0.1, 0.2, 0.2, 0.3}.
The curve associated with PO-GUISE (baseline+HM(P)+PR(C+P)+
MG) is always on top for different proportions of selected tokens (𝜌).
Interestingly, at 166 GFlops our accuracy is still 94.50, on top of
previous methods. The difference with the same pruning method but
without token merging (PR(C+P)) is significant, while not using pose
tokens in pruning reduces even more the performance in all values of
𝜌.
We have also conducted experiments on a Jetson Orin NX (16 GB)
to evaluate performance in a resource-limited device. The baseline
model VideoMAEv2 processes one sample every 1140 ms with 3608
MB memory usage. This further increases to 1290 ms, and 4125 MB
when incorporating human pose estimation. PO-GUISE at 249 GFLOPS
reduces these to 640 ms and 2973 MB, effectively decreasing by 50%
and 27% the computational time and cost. This gain in performance is
especially important in the Jetson architecture, where the GPU and CPU
7
R. Pizarro et al.
Image and Vision Computing 162 (2025) 105686
Fig. 7. Comparison between GFLOPS and accuracy for different configurations and top methods from SOTA in NTU60 (CS). Circle size represents the number of parameters, either
89M or 121M.
share the same unified memory, meaning that a lower model memory
requirement leaves more space for other secondary CPU tasks. Our
memory usage, 2973 MB, also makes it feasible to implement it on the
lower-end Jetson models with 4 GB of memory.
4.3.3. Feasibility of real-world use
This section evaluates the real-world feasibility of our proposed
system, focusing on its implementation cost and scalability compared
to existing solutions. Our analysis assumes the pre-existence of basic
infrastructure, such as video cameras, as our model represents a single
component within a larger monitoring ecosystem.
For our performance and cost baseline, we utilize the NVIDIA Jetson
Nano, an accessible edge computing device priced between $160 and
$250. On this platform, PO-GUISE achieves a throughput of 33 to 52
frames per second (FPS), corresponding to an inference time of 322 to
478 ms per input clip, depending on the model configuration. Factoring
in system overheads such as data I/O, this performance realistically
enables at least one prediction per second. Given that continuous,
real-time monitoring is not essential for tracking most activities of
daily living, a single Jetson Nano could simultaneously serve multiple
residents in an elderly care facility. A key advantage of this edge-
computing approach is that it is self-contained. By processing data
on-site, it eliminates the need to transmit sensitive video footage over
the internet, preserving user privacy.
To contextualize the cost of our system, recent proposals of real-
world deployments rely mainly on motion sensors and focus on fall
detection [49,50]. A low-cost system based on these sensors costs
around $262 for the entire deployment [50]. It is important to note
that only the motion sensor, which is needed for each individual, costs
$103. As such, the scalability of these types of system is much more
costly than that of camera-based approaches.
4.3.4. Visualizations
In this section we show some qualitative results at low token keep
rates of our improved token selection method PO-GUISE against the
top performer token pruning technique [12], Top-K, and the baseline
VideoMAEv2-base model. For a fair comparison, we have configured
both models to have a similar number of visual tokens and GFLOPS.
Specifically, PO-GUISE uses the keep rates 𝜌 = 0.1, 𝜆 = 0.1 and the Top-
K model uses 𝜌 = 0.2. In Fig. 8 we show some examples, each square
represents a visual token and its normalized attention to class token.
If a visual token was selected more than once in time, its attention
is aggregated. For ease of comparison, we have used the same color
map as in Fig. 1. We can see that PO-GUISE effectively selects the
tokens related to the person, while Top-K and the Baseline tend to select
irrelevant tokens. We believe this is a side-effect from training ViTs. At
inference, these use low-informative background areas of images as a
form of repurposed internal computation [51].
The human pose detection task is well learned by the PO-GUISE as
shown in Figs. 9 and 10. Note that we are learning one motion heatmap
per body joint which consists of the sum of probability maps from the
16 frames of the clip. For ease of visualization, we show in the same
image the motion heatmaps corresponding to all body joints.
4.3.5. Discussion
Our contribution is a token selection procedure guided by human
motion that, at default settings, not only maintains, but improves
the accuracy of a top-performing video transformer. This breaks the
typical trade-off between computational cost and accuracy seen in other
approaches. By guiding the transformer’s attention toward areas with
human motion, we reduce GFLOPs by 30% and simultaneously increase
the final accuracy.
Although our model presents an overall high accuracy, an analysis
of failure cases shows the cause of the remaining errors. For example,
the three lowest accuracy classes in Toyota-Smarthome(CS), CutBread,
Usetelephone, and, Takepills, have 60, 62 and 65 points in accuracy,
respectively. In the case of CutBread the low accuracy can be explained
by the scarcity of data, with only 23 instances for training. Meanwhile,
errors in Usetelephone and Takepills arise from the high visual similarity
with other actions. As shown in Fig. 11, Usetelephone is confused with
8
R. Pizarro et al.
Image and Vision Computing 162 (2025) 105686
Fig. 8. Visual Token Attention and Selection. Brighter colors indicate higher attention from the selected visual tokens to the class token. For Top-K Pruning and PO-GUISE, we
show the attention from the selected tokens at the last stage. For the baseline, the attention maps are obtained from the last layer. (For interpretation of the references to color
in this figure legend, the reader is referred to the web version of this article.)
WatchTV, and Takepills is misclassified as Eat. These errors occur due
to similar poses and motion patterns, creating an ambiguity that could
be resolved in a real-world system by aggregating information over a
longer temporal context.
Table 4
Test results on Toyota-Smarthome over the CS, CV1 and CV2 protocols.
Method CS CV1 CV2 GFlops
mCA. (↑) mCA. (↑) mCA. (↑) (↓)
4.4. Comparison with the state-of-the-art
We compare PO-GUISE with state-of-the-art techniques in different
ADL recognition datasets: NTU60, NTU120 (Table 5), and Toyota-
Smarthome (Table 4).
Our method achieves new state-of-the-art results on the Toyota-
SmartHome dataset (Table 4), surpassing the previous state-of-the-art,
𝜋-ViT [10], by 4.07, 3.77, and 11.32 points in accuracy across all
protocols, respectively. The lower performance observed in the CV1
protocol, compared to other protocols, is consistent with previous
work due to the limited training data available for this challenging
single-camera setting.
In the NTU datasets (Table 5), we also surpass state-of-the-art
performance on all cross-subject benchmarks compared to methods
utilizing only RGB input. PO-GUISE outperforms the prior results of
𝜋-ViT [10] by 0.84, and 1.57 on each dataset’s cross-subject protocol
(CS), respectively. Importantly, we achieve these performance gains
AssembleNet++[52] 63.6 – – –
MotionFormer [53] 65.8 45.2 51.0 369
LTN [54] 65.9 – 54.6 –
TimeSFormer [55] 68.4 50.0 60.6 784
VPN++ [1] 69.0 – 54.9 –
Video Swin [34] 69.8 36.6 48.6 281
𝜋-ViT [10] 72.9 55.2 64.8 785
VideoMAEv2-base 73.14 55.20 67.68 360
+ HM(P) 76.01 57.31 71.82 379
PO-GUISE 76.98 58.98 76.12 249
while simultaneously reducing the computational cost of 𝜋-ViT by 536
GFLOPS.
The difference in performance observed between the Toyota-
SmartHome and NTU datasets for cross-view protocols reflects the dif-
ference in difficulty between these benchmarks. In Toyota-SmartHome,
the test cameras maintain a similar viewpoint to the training cameras,
mostly changing the room the subject is present in. The NTU datasets,
and NTU 60 in particular, present a significantly more challenging
9
R. Pizarro et al.
Image and Vision Computing 162 (2025) 105686
Fig. 9. Sample heatmaps from the NTU120 (CS) dataset test set using PO-GUISE. The first column corresponds to the middle frame of the video clip, the second column displays
the temporal heatmaps used as training labels, and the third column shows the predicted heatmaps.
Table 5
Test results on NTU datasets with RGB-only modality at inference.
Method NTU60 NTU120 GFlops
CS CV CS CSet (↓)
Acc. (↑) Acc. (↑) Acc. (↑) Acc. (↑)
VideoCon [56] 91.4 98.0 85.6 87.5 –
ViewCLR [57] 89.7 94.1 86.2 84.5 –
VPN++ [1] 93.5 99.1 86.7 89.3 –
MotionFormer [53] 85.7 91.6 87.0 87.9 369
TimeSFormer [55] 93.0 97.2 90.6 91.6 784
Video Swin [34] 93.4 96.6 91.4 92.1 281
𝜋-ViT [10] 94.0 97.9 91.9 92.9 785
VideoMAEv2-base 94.29 90.91 91.73 89.64 360
+ HM(P) 94.47 91.27 93.36 91.02 379
PO-GUISE 94.84 92.31 93.47 92.11 249
cross-view scenario, where the cameras used during testing are placed
quite differently compared to those utilized for training. However, the
difference in size between these datasets explains the better accuracy
in NTU. Previous methods have attempted to address this challenge
by incorporating 3D pose information during training, 𝜋 -ViT [10] and
VPN++ [1]. Overall, these results highlight the effectiveness of PO-
GUISE in cross-subject protocols, with the use of 3D pose information
as a promising avenue for future work focused on cross-view protocols.
5. Conclusions
State-of-the-art video transformers for action recognition operate
with a quadratic complexity regarding the number of input tokens,
which presents a significant computational challenge. Although token
pruning offers a promising approach to reduce this computational
burden, existing methods often lead to a decrease in action recognition
accuracy.
Our method addresses this limitation by leveraging human motion
information to selectively retain the most informative tokens for action
recognition. This approach achieves a compelling balance between
accuracy and computational efficiency. Specifically in default settings,
our method reduces the number of visual tokens, resulting in a 30%
reduction in GFLOPS while simultaneously increasing accuracy by up
to 8 points.
Although our method demonstrates notable success on all cross-
subject benchmarks, further research is needed to enhance computa-
tional efficiency and accuracy on more challenging cross-view action
10
R. Pizarro et al.
Image and Vision Computing 162 (2025) 105686
Fig. 10. Sample heatmaps from the Toyota-SmartHome (CS) dataset test set using PO-GUISE. The first column corresponds to the middle frame of the video clip, the second
column displays the temporal heatmaps used as training labels, and the third column shows the predicted heatmaps.
Fig. 11. Failure cases by PO-GUISE. Left: Class Usetelephone, Right: Class TakePills. Some erroneous predictions are attributed to data scarcity and visual similarity between distinct
action classes.
recognition tasks. Our future work will explore the integration of
additional semantic tasks to further improve token selection, as well
as the incorporation of 3D pose information during training.
The code required to reproduce the experiments described in this
paper is available on GitHub at https://github.com/RicardoP0/poguise.
CRediT authorship contribution statement
Ricardo Pizarro: Writing – review & editing, Writing – original
draft, Software, Investigation. Roberto Valle: Writing – review &