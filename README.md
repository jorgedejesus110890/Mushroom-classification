# Mushroom-classification

# 1. Introduction

Mushrooms are the fruiting bodies of fungi and are an important food source. Many species of wild or field mushrooms are considered delicacies in some cultures/countries. However, there are numerous mushrooms that can be toxic if eaten, and it is often difficult to distinguish between "edible" and poisonous species. Also, some "edible" species can be toxic to some humans under certain circumstances that may not be predictable. Poisoning as a consequence of mushroom ingestion continues to be a health problem in many countries, resulting in both morbidity and mortality. The annual global mortality rate from eating mushrooms is unknown, but speculatively may be at least 100 deaths/year, and is probably an underestimate given the approximately 50-100 deaths/year in Europe alone [1].

# 2. Theoretical Framework

Machine learning allows machines to make sense of how to perform actions that irrefutably required a man to do them. The different species of mushrooms constitute a source of food, so it is essential that we have some technique to collect them as poisonous and non-poisonous. Using machine learning models, we can get a really correct classification system that can classify fungi.

## 2.1 Fungi

Biologists currently use the term fungus (fungus=mushroom from Gr. Sphongus=sponge) to designate eukaryotic organisms, carriers of spores, or chlorophyll, which generally reproduce sexually and asexually and whose somatic, branched, and filamentous, they are surrounded by cell walls that contain chitin or cellulose, or both, along with other complex organic molecules. In other words, this means that fungi have true nuclei typical of cells, that they reproduce by means of spores, and that they do not have chlorophyll. Most fungi have a sexual mechanism. There are also some organisms that mycologists have inadvertently studied, which are probably not fungi, but are slime molds or myxomycetes, cellular and plasmodial. Myxomycetes resemble fungi in many ways and are studied by mycologists [2].

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/Mushroom.jpg?raw=true)

Figure 1. Mushroom.

## 2.2 Diversity of Fungi

Fungi are a group of living organisms devoid of chlorophyll. They resemble simple plants in that, in a few exceptions, they have defined cell walls, are usually non-motile, although they do have mobile reproductive cells, and reproduce by means of spores. The classification of fungi presents innumerable difficulties. However, taxonomy has a double objective, first of all, to name organisms, with as little confusion as possible, and secondly, to express current concepts about the relationships of fungi among themselves and with other living organisms. Paleontological studies indicate that fungi constitute a very ancient group that probably dates back to the Precambrian. The taxonomic groups used in the classification of fungi are: super kingdom, kingdom, division, class, order, family, genus and species[3].

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/Morfology.jpg?raw=true)

Figure 2. Morphology of a mushroom.

Figure 2 shows the morphology of a fungus, this composed of:

- Pileus: Also called the hat, it is the part of the fruiting body of the fungus that supports the surface where the spores are housed.
- Lamina: They are the structures under the hat that act as a union of the lamellae with the foot.
- Lamellae: They are the ones that contain the basidia, and these in turn are the ones that generate the spores.
- Hymenium: It is the set of sheets and lamellae, it is the fertile part of the fungus.
- Ring: Only present in some fungi, it is the rest of the partial veil when it is broken to expose the spores. The partial veil is the structure of some fungi to protect the development of the hymenium.
- stipe: Also called peduncle, it is the one that holds the hat, it is made up of hyphal sterile tissue.
- Volva: Only present in some mushrooms, it is the rest left by the universal veil. The universal veil is a cover that completely covers an immature mushroom, in some cases leaving a visible residue on the cap.
- Mycelium: It is the set of hyphae responsible for the nutrition of the fungi.

## 2.3 Classification algorithms

### 2.3.1 Decision Tree

Decision trees are a classification model used in artificial intelligence, whose main characteristic is its visual contribution to decision making. ID3, Iterative Dichotomiser 3 is a decision tree learning algorithm used for classifying objects with the iterative inductive approach. In this algorithm the top-down approach is used. The top node is called the root node and the others are leaf nodes. So it is a traversal from the root node to the leaf nodes. Each node requires some test on the attributes that decide the level of the leaf nodes. These decision trees are mainly used for decision making [4].

### Entropy

Entropy is used to determine how informative a particular input attribute is over the output attribute for a subset of the training data. Entropy is a measure of uncertainty in communication systems.

The classical formula for entropy, whose value is between 0 and 1, is as follows:

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/Entropy.jpg?raw=true)

### Information gain

The information gain is the measure used by the ID3 algorithm to select the most profitable attribute for separation. Also known as mutual information, information gain aims to reduce information uncertainty. In fact, the mutual information of two random variables X and Y measures the dependency relationship between the two variables: the higher the value of the mutual information, the stronger the dependence between X and Y, that is, in our case, the attribute with the highest information gain separates the data set well, which is why it should be chosen. The mathematical expression of this measure has the form:

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/Gain.jpg?raw=true)

### 2.3.2 K–NN

K-Nearest Neighbor (KNN) algorithm for machine learning K-Nearest Neighbor is one of the simplest machine learning algorithms based on the supervised learning technique. The K-NN algorithm assumes the similarity between the new case/data and the available cases and places the new case in the category that is most similar to the available categories. The K-NN algorithm stores all available data and classifies a new data point based on similarity. This means that when new data appears, it can easily be classified into a category of well sets using the K-NN algorithm. The K-NN algorithm can be used for both regression and classification, but it is mainly used for classification problems. K-NN is a non-parametric algorithm, which means that it does not make any assumptions about the underlying data. It is also called a lazy learning algorithm because it does not learn from the training set immediately, but instead stores the data set and, at classification time, performs an action on the data set. The KNN algorithm in the training phase just stores the data set and when it gets new data, it classifies it into a category that is very similar to the new data.

### 2.3.3 PCA

Principal Component Analysis (PCA) is a statistical method that allows you to simplify the complexity of sample spaces with many dimensions while preserving their information. Suppose there is a sample with 𝑛 individuals each with p variables (𝑋1, 𝑋2, …, 𝑋𝑝), that is, the sample space has p dimensions; PCA allows us to find a number of underlying factors (𝑧 < 𝑝) that explain approximately the same as the original p variables. Where before p values were needed to characterize each individual, now z values suffice. Each of these z new variables is called a principal component. PCA belongs to the family of techniques known as unsupervised learning. In this case, the response variable Y is not taken into account since the objective is not to predict Y, but to extract information using the predictors, for example, to identify subgroups. The main problem faced by unsupervised learning methods is the difficulty in validating the results, since there is no response variable available to compare them. The PCA method therefore allows to “condense” the information provided by multiple variables into just a few components. This makes it a very useful method to apply prior to the use of other statistical techniques such as regression, clustering, etc. Even so, we must not forget that it is still necessary to have the value of the original variables to calculate the components.

# 3. Methodology

This section presents the classification methodology used to classify fungi based on behavioral characteristics. The methodology involves three important phases, which are preprocessing, feature selection, and classification, as shown in Figure 3.

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/Metodology.jpg?raw=true)

Figure 3. Methodology.

## 3.1	DataSet

The Mushroom dataset and were obtained from the UCI Machine Learning Repository. This data set includes descriptions of hypothetical specimens corresponding to 23 species of gilled fungi in the family Agaricus and Lepiota.


|Attribute|	Nominal value|
|---------|---------------|
|class	  |e=edible
|         |p=poisonous
|cap-shape|bell=b
|         |conical=c
|         |convex=x
|         |flat=f
|         |knobbed=k
|         |sunken=s
|cap-surface|fibrous=f
|           |grooves=g
|           |scaly=y
|           |smooth=s
|cap-color  |brown=n
|           |buff=b
|           |cinnamon=c
|           |gray=g
|           |green=r
|           |pink=p
|           |purple=u
|           |red=e
|           |white=w
|           |yellow=y
|bruises?	  |bruises =t
|           |no=f
|odor	      |almond=a
|           |anise=l
|           |creosote=c
|           |fishy=y
|           |foul=f
|           |musty=m
|            |none=n
|            |pungent=p
|            |spicy=s
|gill-attachment	|attached=a
|                  |descending=d
|                  |free=f
|                  |notched=n
|gill-spacing	    |close=c
|                  |crowded=w
|                  |distant=d
|gill-size	      |broad=b
|                  |narrow=n
|gill-color	      |black=k
|                  |brown=n
|                  |buff=b
|                  |chocolate=h
|                  |gray=g
|                  |green=r
|                  |orange=o
|                  |pink=p
|                  |purple=u
|                  |red=e
|                  |white=w
|                  |yellow=y
|stalk-shape	    |enlarging=e
|                  |tapering=t
|stalk-root	      |bulbous=b
|                  |club=c
|                  |cup=u
|                  |equal=e 
|                  |rhizomorphs=z
|                  |rooted=r
|                  |missing=?
|stalk-surface-above-ring 	|fibrous=f
|                            |scaly=y
|                            |silky=k
|                            |smooth=s
|stalk-surface-below-ring	  |fibrous=f
|                            |scaly=y
|                            |silky=k
|                            |smooth=s
|stalk-color-above-ring	    |brown=n
|                            |buff=b
|                            |cinnamon=c
|                            |gray=g
|                            |orange=o 
|                            |pink=p
|                            |red=e
|                            |white=w
|                            |yellow=y
|stalk-color-below-ring	    |brown=n
|                            |buff=b
|                            |cinnamon=c
|                            |gray=g
|                            vorange=o
|                            |pink=p
|                            |red=e
|                            |white=w
|                            |yellow=y
|veil-type                  |partial=p
|                            |universal=u
|veil-color	                |brown=n
|                            |orange=o
|                            |white=w
|                            |yellow=y
|ring-number	              |none=n
|                            |one=o
|                            |two=t
|ring-type	                |cobwebby=c
|                            |evanescent=e
|                            |flaring=f
|                            |large=l, none=n
|                            |pendant=p
|                            |sheathing=s
|                            |zone=z
|spore-print-color	        |black=k
|                            |brown=n
|                            |buff=b
|                            |chocolate=h
|                            |green=r
|                            |orange=o
|                            |purple=u
|                            |white=w
|                            |yellow=y
|population	                |abundant=a
|                            |clustered=c
|                            |numerous=n
|                            |scattered=s
|                            |several=v
|                            |solitary=y
|habitat	                  |grasses=g
|                            |leaves=l
|                            |meadows=m
|                            |paths=p
|                            |urban=u
|                            |waste=w
|                            |woods=d

Table 1. Detailed information by attribute

## 3.2 Preprocessing

Preprocessing deals with data that is missing or invalid due to various reasons, including data entry issues. In addition, there are also attributes that are not relevant to the experiment. Irrelevant data should be removed from the data set because their presence may reduce the quality or accuracy of the classification experiment. In this investigation, the data set consists of nominal values. This data must undergo a data transformation, in which nominal values will be transformed into numeric values. By transforming the nominal value into numerical values, the data can now be fed into the classification algorithm.

# 4. Experiments and Results

## Gather

We start by importing the libraries, and displaying a sample of the data being made using the Mushroom database and obtained from the UCI Machine Learning Repository. This data set includes descriptions of hypothetical specimens corresponding to 23 species of gilled fungi in the family Agaricus and Lepiota.

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/Data.jpg?raw=true)

Figure 4. Database instances and attributes.

In the same way, we proceed to visualize quickly and graphically the distribution of the data.

### Data distribution

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/distribution.png?raw=true)

Figure 5. Distribution of data from the Mushroom data set.

### Set up:

Before reading our data to start with the classification, it is necessary to select the characteristics that provide more information.
It is recommended to normalize the data in this case it will be from 0 to 1.

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/Covariance.jpg?raw=true)

Table 2. Values obtained by PCA.

It can be seen in Table 2 that the last 12 components have less amount of data variation, the first 9 components retain more than 86% of the data.
We proceed to divide the data into 4 subsets which we will call X_train, X_test, y_train, y_test, to train and test our classifier.

Analyze:
Next, we read the file and save our actual classifier values.

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/NewDataset.jpg?raw=true)

Figure 7. New dataset with the attributes obtained by PCA. The data was split 80-20.

### Train:

The decision tree classifier is created and trained, obtaining the following tree.

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/Tree.jpg?raw=true)

Figure 8. Mushroom decision tree.
In the same way, a K-NN classifier is created.

### Test:

After performing the prediction of our test set to assess the classification ability of the decision tree.

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/MC_Tree.jpg?raw=true)

Figure 9. Decision tree confusion matrix.

After performing the prediction of our test set to assess the classification ability of K-NN.

![alt text](https://github.com/jorgedejesus110890/Mushroom-classification/blob/main/MC_KNN.jpg?raw=true)

Figure 10. K-NN confusion matrix.

The confusion matrix shows the results of the experiment.

### Comparison of results

|         |Decision Tree|K-NN|
|---------|--------------------|-------------------|
|Accuracy |	0.9993846153846154|	0.9926153846153846|
|Precision	|1.0|	1.0|
|Recall	|0.9987063389391979	|0.9846547314578005|
|F1_score	|0.9993527508090615|	0.9922680412371134|

Table 3. Metrics.

# 5. Conclusions

Classification of these data sets was carried out to classify mushrooms as edible or poisonous based on their behavioral characteristics. The data set contained 22 nominal attribute numbers (characteristics). Statistics from the dataset showed that the edible mushroom has an absolute count of 4,208 at 0.518%, while the poisonous mushroom has an absolute count of 3,916 at 0.482%. The results showed that the decision tree is the one that shows a higher pressure compared to K-NN as shown in Table 3.

# 6. Bibliography

[1] Dadpour B, Tajoddini S, Rajabi M, Afshari R. (2017) Mushroom Poisoning in the Northeast of Iran; a Retro- spective 6-Year Epidemiologic Study. Emergency, 5(1):e23.

[2]Herrera Teófilo, et.al, 1998, El reino de los hongos micología básica y aplicada, Editorial fondo de cultura económica, segunda edición UNAM, Pag: 25-36.


[4] [2] Rokach, L., Maimon, O. (2005). Top-Down Induction of Decision Trees Classifiers: A Survey. IEEE Transaction on Systems, Man, and Cybernetics – Part C: Applications and Reviews, Vol. 35, 4, pp. 476-487.

[5] E. Boa. Overstory# 164-wild edible fungi and livelihoods. [Online]. Available: http://www.agroforestry.net/the-overstory/108-overstory-164- wild-edible-fungi-and-livelihoods.

[6] J. Schlimmer, “Mushroom records drawn from the audubon society field guide to north american mushrooms,” GH Lincoff (Pres), New York, 1981.


