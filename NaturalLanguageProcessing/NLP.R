
dataset_original = read.delim("Restaurant_Reviews.tsv", quote = "", sep = "\t", stringsAsFactors = FALSE)

# install.packages("tm")
# install.packages("SnowballC")
library(SnowballC)
library(tm)
corpus = VCorpus(VectorSource(dataset[, 1]))
as.character(corpus[[1]])
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating bag of words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)  # keep 99.9% of most frequent words

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = factor(dataset_original$Liked, levels = c(0, 1))


library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, 0.9)
train = subset(dataset, split == TRUE)
test = subset(dataset, split == FALSE)

# Fitting the model
# install.packages("rpart")
library(randomForest)
classifier = randomForest(x = train[, -719],
                          y = train[, 719],
                          ntree = 20)


#y_pred = predict(classifier, newdata = test[-3])
#y_pred = ifelse(y_pred[, 2] > 0.5, 1, 0)
y_pred = predict(classifier, newdata = test[-719], type = "class")




# Create confusion matrix
cm = table(test[,719], y_pred)



