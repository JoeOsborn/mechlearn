#----------------------------------------------
# kMeans clustering
#----------------------------------------------
#K-means clustering
wss <- (nrow(shared[10:ncol(shared)])-1)*sum(apply(shared[10:ncol(shared)-1],2,var))
for (i in 2:15){
  wss[i] <- sum(kmeans(shared[10:ncol(shared)-1], centers=i)$withinss)
}
rm(i)
plot(1:15, wss, type="b", xlab="shared, number of kMeans Clusters", ylab="Within groups sum of squares")
rm(wss)
# K-Means Cluster Analysis
shared_kMeansFit <- kmeans(shared[10:ncol(shared)-1], 3) # 3 cluster solution
shared_kMedioidsFit <-  pamk(shared[10:ncol(shared)-1])
#clusplot(shared[10:ncol(shared)-1], shared_kMeansFit$cluster, color=TRUE, shade=TRUE, labels=1, lines=0, main="Shared, kMeans clusters")
row.names(shared) <- shared$name
fviz_cluster(shared_kMeansFit,data = shared[10:ncol(shared)-1], geom = "text", repel = TRUE)

# get cluster means
aggregate(shared[10:ncol(shared)-1],by=list(shared_kMeansFit$cluster),FUN=mean)

# append cluster assignment to each game
shared_kMeans_cluster <- shared_kMeansFit$cluster
table(shared_kMeans_cluster)
shared <- data.frame(shared, shared_kMeans_cluster)
shared_kMeans_profile <- shared_kMeansFit$centers
xtable(table(shared$Developer,shared$shared_kMeans_cluster))

#----------------------------------------------
# Archetypal analysis of games
#----------------------------------------------
#Archetypes from HAs
shared_aa <- stepArchetypes(data = shared[10:(ncol(shared)-2)], k = 1:15, verbose = FALSE, nrep = 4)
screeplot(shared_aa)
shared_aa <- bestModel(shared_aa[[7]])
pcplot(shared_aa, shared[10:ncol(shared)-2])

#Find archetype for each game
shared_aa_cluster<-max.col(coef(shared_aa))
#Append to dataset
shared <- data.frame(shared,shared_aa_cluster)
names(shared)[ncol(shared)] <- "Archetype"
#Archetype counts
table(shared_aa_cluster)
#Saving archetype information for plotting
shared_aa_profile<-parameters(shared_aa)

#----------------------------------------------
# Plotting clusters and archetypes
#----------------------------------------------
#plots K-means
plot(shared_pcaFit$ind$coord[,1:2], type="n")
text(shared_pcaFit$ind$coord[,1:2], col=shared_kMeansFit$cluster, labels=shared_kMeansFit$cluster)
arrows(0, 0, 7*shared_pcaFit$var$coord[,1], 7*shared_pcaFit$var$coord[,2], col = "chocolate", angle = 15, length = 0.025)
text(2*shared_pcaFit$var$coord[,1], 2*shared_pcaFit$var$coord[,2], labels=names(shared)[10:ncol(shared)-3])

#plots archetypes
plot(shared_pcaFit$ind$coord[,1:2], type="n")
text(shared_pcaFit$ind$coord[,1:2], col=shared_aa_cluster, labels=shared_aa_cluster)
arrows(0, 0, 7*shared_pcaFit$var$coord[,1], 7*shared_pcaFit$var$coord[,2], col = "chocolate", angle = 15, length = 0.025)
text(2*shared_pcaFit$var$coord[,1], 2*shared_pcaFit$var$coord[,2], labels=names(shared)[10:ncol(shared)-3])

#plots K-means centroids and archetype profiles together
#profiles for K-means and archetypes added to original data as supplementary points in order to map the results in a two-dimensional principal component space
#shared_clusters_archetypes<-rbind(shared[10:ncol(shared)-3],shared_aa_profile,shared_kMeans_profile)
#row.names(shared_clusters_archetypes)<-NULL
#shared_pcaFit <- PCA(X = extractNumerics_clusters_archetypes,7, ind.sup = 37:45, graph = FALSE)
#plot(pcaFit$ind$coord[,1:2], pch=".", cex=6, xlim=(c(-8,5)), ylim=c(-3,3))
#text(pcaFit$ind.sup$coord[,1:2], labels=c("A1","A2","A3","A4","A5","K1","K2","K3","K4"), col=c(rep("blue",5),rep("red",4)),cex=1)