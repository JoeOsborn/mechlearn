library(cluster)
library(ggplot2)
library(ggdendro)
library(RColorBrewer)

labelColors = brewer.pal(n = unique(shared$Publisher),name = "Spectral")
# cut dendrogram in 4 clusters
#clusMember = cutree(hc, 4)
publishers <- unique(shared$Publisher)
# function to get color labels
colLab <- function(n) {
  if (is.leaf(n)) {
    a <- attributes(n)
    labCol <- labelColors[publishers == a$Publisher]
    attr(n, "nodePar") <- c(a$nodePar, lab.col = labCol)
  }
  n
}
# using dendrapply

# make plot


shared_agnes <- agnes(x = shared[10:ncol(shared)-3], metric = "euclidian", method = "ward",stand = TRUE)
sagdd <- as.dendrogram(shared_agnes)
ggdendrogram(sagdd,rotate = TRUE)


plot(shared_agnes, main = "Hierarchical clustering of jumps", xlab = " ", horizontal = TRUE)